"""Dynamic-policy collection/update connection with an external durable budget."""

import copy

from src.rl.candidate_imitation import decode_example
from src.rl.candidate_pilot_driver import CandidateContinuation
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, assert_dynamic_budget_ancestor
from src.rl.prospective_ddpg_kernel import state_digest


class DynamicCandidateContinuation(CandidateContinuation):
    """Reuse closed-rollout semantics without mutating the historical driver.

    The caller supplies a session factory and already active owner scope. There
    is no patient factory, scientific launcher, or automatic retry in this API.
    """

    def __init__(self, kernel, budget, settings, *, enabled=False, role, scope,
                 seeds, horizon, session_factory):
        if enabled is not True or type(budget) is not DynamicCandidateBudget:
            raise ValueError("explicit dynamic continuation and budget required")
        expected = {"ppo": DynamicCandidatePPOKernel, "bc_continue": DynamicCandidateImitationKernel}
        if role not in expected or type(kernel) is not expected[role] or budget.active != scope:
            raise ValueError("matching kernel and active owner scope required")
        if role == "ppo" and kernel.mode != "online":
            raise ValueError("frozen policy cannot collect a continuation arm")
        required = {"episodes_per_model", "episodes_per_rollout", "max_updates", "epochs", "gae_lambda"}
        if set(settings) != required or not callable(session_factory):
            raise ValueError("explicit fixed-rollout settings and session factory required")
        for key in required - {"gae_lambda"}:
            if type(settings[key]) is not int or settings[key] < 1:
                raise ValueError("positive integer rollout settings required")
        if (type(horizon) is not int or horizon < 1
                or settings["episodes_per_model"] != settings["episodes_per_rollout"] * settings["max_updates"]
                or len(seeds) != settings["episodes_per_model"] or len(set(seeds)) != len(seeds)
                or any(type(seed) is not int or seed < 0 for seed in seeds)
                or settings["max_updates"] != kernel.settings.max_updates
                or settings["gae_lambda"] != 1.0):
            raise ValueError("whole fixed rollouts, unique starts and matching caps required")
        if (role == "ppo" and (settings["epochs"] != kernel.settings.epochs
                                or settings["gae_lambda"] != kernel.settings.gae_lambda
                                or horizon * settings["episodes_per_rollout"] != kernel.settings.max_rollout_steps)):
            raise ValueError("PPO rollout contract mismatch")
        if role == "bc_continue" and kernel.settings.allowed_split != "training":
            raise ValueError("continuation imitation requires the training split")
        self.kernel, self.budget, self.factory = kernel, budget, session_factory
        self.settings, self.role, self.scope, self.seeds = copy.deepcopy(settings), role, scope, tuple(seeds)
        self.horizon = horizon
        self.manifest = {"format": "dynamic-candidate-continuation-v1", "role": role, "scope": scope,
                         "seeds": self.seeds, "settings": self.settings, "horizon": horizon,
                         "initial_kernel_sha256": state_digest(kernel.state_dict())}
        self.episode, self.active, self.examples, self.receipts, self.last_closed = 0, None, [], [], None
        self.failure = None

    def _require_live(self):
        if self.failure is not None or self.kernel._failure is not None or self.budget.failed:
            raise ValueError("failed continuation is terminal")

    def _fail(self, error):
        self.failure = {"error_type": type(error).__name__, "message": str(error),
                        "episode": self.episode, "updates": self.updates}
        self.budget.close()

    def start_episode(self):
        self._require_live()
        return super().start_episode()

    def step(self):
        self._require_live()
        if self.done or self.update_due or self.episode >= len(self.seeds) or self.budget.active != self.scope:
            raise ValueError("collection forbidden at closed/update/wrong-scope boundary")
        try:
            self.start_episode()
            event = self.active.step(before_step=lambda: self.budget.debit_environment("trajectory"))
            if self.active.closed:
                if self.role == "ppo":
                    self.kernel.add_segment(self.active.segment(self.settings["gae_lambda"]))
                else:
                    self.examples.extend(copy.deepcopy(self.active.examples))
                self.last_closed = self.active.state_dict()
                self.receipts.append({"episode": self.episode, "seed": self.seeds[self.episode],
                                      "trajectory_id": self.active.trajectory_id,
                                      "session_sha256": state_digest(self.last_closed), "steps": self.active.index})
                self.episode += 1
                self.active = None
            return event
        except BaseException as error:
            self._fail(error)
            raise

    def update(self):
        self._require_live()
        if not self.update_due or self.budget.active != self.scope:
            raise ValueError("only a complete fixed rollout may update")
        try:
            self.budget.check()
            if self.role == "ppo":
                result = self.kernel.update(before_optimizer_step=self.budget.debit_optimizer,
                                            before_minibatch=self.budget.check_minibatch)
            else:
                result = self.kernel.fit(self.examples, epochs=self.settings["epochs"],
                                         before_step=lambda: self.budget.debit_optimizer("actor"))
                self.examples = []
            self.budget.check()
            return result
        except BaseException as error:
            self._fail(error)
            raise

    def state_dict(self):
        return super().state_dict() | {"failure": copy.deepcopy(self.failure)}

    def load_state_dict(self, state):
        self._require_live()
        if (set(state) != set(self.state_dict()) or state["manifest"] != self.manifest
                or state["failure"] is not None):
            raise ValueError("continuation contract mismatch or terminal failure evidence")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        kernel = copy.deepcopy(self.kernel)
        kernel.load_state_dict(state["kernel"])
        episode = state["episode"]
        if type(episode) is not int or not 0 <= episode <= len(self.seeds) or len(state["receipts"]) != episode:
            raise ValueError("continuation episode/receipt cursor differs")
        updates = kernel.total_updates if self.role == "ppo" else len(kernel.history)
        pending_episodes = episode - updates * self.settings["episodes_per_rollout"]
        if not 0 <= pending_episodes <= self.settings["episodes_per_rollout"]:
            raise ValueError("update and collection cursors disagree")
        for i, receipt in enumerate(state["receipts"]):
            if receipt["episode"] != i or receipt["seed"] != self.seeds[i] or receipt["steps"] != self.horizon:
                raise ValueError("receipt order/seed mismatch")
        if len({r["trajectory_id"] for r in state["receipts"]}) != episode:
            raise ValueError("duplicate trajectory")
        if self.role == "ppo":
            if state["examples"] or len(kernel.pending) != pending_episodes:
                raise ValueError("PPO pending rollout mismatch")
        else:
            expected = pending_episodes * self.horizon
            examples = state["examples"]
            if len(examples) != expected or len({e["identity"] for e in examples}) != expected:
                raise ValueError("imitation pending rollout mismatch")
            for example in examples:
                decode_example(example, kernel.contract)
                if example["split"] != "training":
                    raise ValueError("non-training pending imitation data")
        if episode:
            if state["last_closed"] is None or state_digest(state["last_closed"]) != state["receipts"][-1]["session_sha256"]:
                raise ValueError("closed collector boundary differs")
            last = self._session_from_state(state["last_closed"], episode - 1, kernel)
            if not last.closed or last.trajectory_id != state["receipts"][-1]["trajectory_id"]:
                raise ValueError("last collector is not the declared closed episode")
        elif state["last_closed"] is not None:
            raise ValueError("unexpected closed collector")
        active = None
        if state["active"] is not None:
            if episode >= len(self.seeds) or pending_episodes == self.settings["episodes_per_rollout"]:
                raise ValueError("unexpected active episode at update boundary")
            active = self._session_from_state(state["active"], episode, kernel)
            if active.closed or state_digest(active.learner.state_dict()) != state_digest(kernel.state_dict()):
                raise ValueError("active collector/kernel mismatch")
            kernel = active.learner
        spent = state["budget"]["owner_counts"]
        prefix = f"section/{self.scope}:"
        actors = kernel.total_owner_steps if self.role == "ppo" else kernel.steps
        critics = kernel.total_owner_steps if self.role == "ppo" else 0
        if (spent.get(prefix + "trajectory", 0) < episode * self.horizon + (active.index if active else 0)
                or spent.get(prefix + "actor", 0) < actors or spent.get(prefix + "critic", 0) < critics):
            raise ValueError("collector/optimizer work has no corresponding owner debit")
        self.kernel, self.active, self.episode = kernel, active, episode
        self.examples, self.receipts = copy.deepcopy(state["examples"]), copy.deepcopy(state["receipts"])
        self.last_closed = copy.deepcopy(state["last_closed"])
