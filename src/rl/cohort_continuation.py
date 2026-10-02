"""Fixed prefix-plus-tail batches with no tail actions admitted to learning.

The outer campaign owns scientific admission, persistence and construction.
Recovery factories must rebuild an owned collection without recording/replay;
they receive restore_state explicitly so a live recorder is not opened twice.
"""

import copy

from src.rl.candidate_imitation import decode_example
from src.rl.cohort_collection import CohortCollection
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.dynamic_candidate_continuation import DynamicCandidateContinuation
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_resources import DynamicCandidateBudget, assert_dynamic_budget_ancestor
from src.rl.prospective_ddpg_kernel import state_digest


class CohortContinuation(DynamicCandidateContinuation):
    def __init__(self, kernel, budget, settings, *, enabled=False, role, scope,
                 seeds, prefix_steps, accounting_steps, session_factory, finish_collection):
        expected = {"ppo": CohortPPOKernel, "bc_continue": DynamicCandidateImitationKernel}
        required = {"episodes_per_model", "episodes_per_rollout", "max_updates", "epochs", "gae_lambda"}
        if (enabled is not True or role not in expected or type(kernel) is not expected[role]
                or type(budget) is not DynamicCandidateBudget or budget.active != scope
                or not callable(session_factory) or not callable(finish_collection)
                or set(settings) != required):
            raise ValueError("explicit cohort kernel, owner, settings and persistence callbacks required")
        if (any(type(settings[k]) is not int or settings[k] < 1 for k in required - {"gae_lambda"})
                or settings["gae_lambda"] != 1.
                or any(type(n) is not int or n < 1 for n in (prefix_steps, accounting_steps))
                or settings["episodes_per_model"] != settings["episodes_per_rollout"] * settings["max_updates"]
                or len(seeds) != settings["episodes_per_model"] or len(set(seeds)) != len(seeds)
                or any(type(s) is not int or s < 0 for s in seeds)
                or settings["max_updates"] != kernel.settings.max_updates):
            raise ValueError("whole fixed batches, unique seeds and positive horizons required")
        if role == "ppo":
            if (kernel.mode != "online" or prefix_steps != kernel.episode_horizon
                    or accounting_steps != kernel.accounting_steps
                    or settings["episodes_per_rollout"] != kernel.episodes_per_rollout
                    or settings["epochs"] != kernel.settings.epochs
                    or tuple(seeds) != tuple(r["environment_seed"] for r in kernel.training_manifest)):
                raise ValueError("exact kernel prefix/tail/training contract required")
        elif kernel.settings.allowed_split != "training":
            raise ValueError("only training continuation imitation admitted")
        self.kernel, self.budget, self.factory = kernel, budget, session_factory
        self.finish_collection = finish_collection
        self.settings, self.role, self.scope, self.seeds = copy.deepcopy(settings), role, scope, tuple(seeds)
        self.prefix_steps, self.accounting_steps = prefix_steps, accounting_steps
        self.horizon = prefix_steps + accounting_steps
        self.manifest = dict(format="cohort-continuation-v1", role=role, scope=scope,
            seeds=self.seeds, settings=self.settings, prefix_steps=prefix_steps,
            accounting_steps=accounting_steps, horizon=self.horizon,
            initial_kernel_sha256=state_digest(kernel.state_dict()))
        self.episode, self.active, self.examples, self.receipts, self.last_closed = 0, None, [], [], None
        self.failure = None

    def _validate_collection(self, collection, episode, learner):
        objective = learner.objective if self.role == "ppo" else "none"
        if (type(collection) is not CohortCollection or collection.split != "training"
                or collection.objective != objective or collection.prefix.selection != "sample"
                or collection.learner is not learner
                or collection.env.cohort_spec.enrollment_steps != self.prefix_steps
                or collection.env.cohort_spec.accounting_steps != self.accounting_steps
                or collection.prefix.manifest["environment_seed"] != self.seeds[episode]):
            raise ValueError("matching prefix-only learner, objective, seed and tail required")
        if self.role == "ppo" and collection.trajectory_id != learner.training_manifest[episode]["trajectory_id"]:
            raise ValueError("collector is outside the fixed training inventory")

    def start_episode(self):
        self._require_live()
        if self.done or self.update_due or self.episode >= len(self.seeds) or self.budget.active != self.scope:
            raise ValueError("collection forbidden at closed/update/wrong-owner boundary")
        self.budget.check()
        if self.active is None:
            collection = self.factory(self.kernel, self.episode, self.seeds[self.episode], restore_state=None)
            self._validate_collection(collection, self.episode, self.kernel)
            if collection.index != 0:
                raise ValueError("new collection must start before all prefix/tail calls")
            self.active = collection
        return self.active

    def step(self):
        self._require_live()
        try:
            self.start_episode()
            event = self.active.step(before_step=lambda: self.budget.debit_environment("trajectory"))
            if self.active.closed:
                # Persist and independently recount the tail BEFORE any segment
                # or imitation example becomes eligible for an update.
                evidence = self.finish_collection(self.active)
                if not isinstance(evidence, dict) or not evidence:
                    raise ValueError("complete persisted cohort evidence required")
                closed = self.active.state_dict()
                if self.role == "ppo":
                    self.kernel.add_segment(self.active.segment(self.settings["gae_lambda"]),
                                            cohort=self.active.closure_receipt())
                else:
                    self.examples.extend(copy.deepcopy(self.active.examples))
                self.last_closed = closed
                self.receipts.append(dict(episode=self.episode, seed=self.seeds[self.episode],
                    trajectory_id=self.active.trajectory_id, steps=self.active.index,
                    prefix_steps=self.prefix_steps, evidence=copy.deepcopy(evidence),
                    session_sha256=state_digest(closed)))
                self.episode += 1
                self.active = None
            return event
        except BaseException as error:
            self._fail(error)
            raise

    def _session_from_state(self, saved, episode, kernel):
        prefix = saved["prefix_snapshot"] or saved["prefix_live"]
        learner = copy.deepcopy(kernel)
        learner.load_state_dict(prefix["kernel"])
        learner.sampling_rng.set_state(prefix["initial_rng"])
        collection = self.factory(learner, episode, self.seeds[episode], restore_state=saved)
        self._validate_collection(collection, episode, learner)
        collection.load_state_dict(saved)
        return collection

    def load_state_dict(self, state):
        self._require_live()
        if (set(state) != set(self.state_dict()) or state["manifest"] != self.manifest
                or state["failure"] is not None):
            raise ValueError("cohort continuation contract or failure differs")
        assert_dynamic_budget_ancestor(state["budget"], self.budget)
        episode = state["episode"]
        if (type(episode) is not int or not self.episode <= episode <= len(self.seeds)
                or len(state["receipts"]) != episode
                or state["receipts"][:self.episode] != self.receipts):
            raise ValueError("invalid episode cursor, replaced receipt or rewind")
        kernel = copy.deepcopy(self.kernel)
        kernel.load_state_dict(state["kernel"])
        updates = kernel.total_updates if self.role == "ppo" else len(kernel.history)
        pending = episode - updates * self.settings["episodes_per_rollout"]
        if updates < self.updates or not 0 <= pending <= self.settings["episodes_per_rollout"]:
            raise ValueError("update/prefix cohort progress differs or rewinds")
        for i, receipt in enumerate(state["receipts"]):
            if (receipt["episode"] != i or receipt["seed"] != self.seeds[i]
                    or receipt["steps"] != self.horizon or receipt["prefix_steps"] != self.prefix_steps
                    or not isinstance(receipt["evidence"], dict) or not receipt["evidence"]):
                raise ValueError("fixed complete cohort receipt required")
        if len({r["trajectory_id"] for r in state["receipts"]}) != episode:
            raise ValueError("duplicate cohort trajectory")
        if self.role == "ppo":
            if state["examples"] or len(kernel.pending) != pending or len(kernel.pending_cohorts) != pending:
                raise ValueError("pending prefix/tail PPO batch differs")
            expected = kernel.training_manifest[updates * kernel.episodes_per_rollout:episode]
            if [r["trajectory_id"] for r in kernel.pending_cohorts] != [r["trajectory_id"] for r in expected]:
                raise ValueError("pending tail inventory differs")
        else:
            examples = state["examples"]
            if (len(examples) != pending * self.prefix_steps
                    or len({e["identity"] for e in examples}) != len(examples)):
                raise ValueError("only prefix BC examples may be retained")
            for example in examples:
                decode_example(example, kernel.contract)
                if example["split"] != "training":
                    raise ValueError("nontraining BC example")
        if episode:
            if state["last_closed"] is None or state_digest(state["last_closed"]) != state["receipts"][-1]["session_sha256"]:
                raise ValueError("last closed prefix/tail envelope differs")
            last = self._session_from_state(state["last_closed"], episode - 1, kernel)
            if (not last.closed or last.index != self.horizon
                    or last.trajectory_id != state["receipts"][-1]["trajectory_id"]):
                raise ValueError("incomplete declared closed cohort")
        elif state["last_closed"] is not None:
            raise ValueError("unexpected completed cohort")
        active = None
        if state["active"] is not None:
            if episode >= len(self.seeds) or pending == self.settings["episodes_per_rollout"]:
                raise ValueError("active collection beyond update or final boundary")
            active = self._session_from_state(state["active"], episode, kernel)
            if (active.closed or state_digest(active.learner.state_dict()) != state_digest(kernel.state_dict())
                    or (self.active is not None and self.episode == episode and active.index < self.active.index)):
                raise ValueError("active collector/kernel mismatch or rewind")
            kernel = active.learner
        elif self.active is not None and self.episode == episode:
            raise ValueError("cannot discard a live cohort")
        spent, owner = state["budget"]["owner_counts"], f"section/{self.scope}:"
        actors = kernel.total_owner_steps if self.role == "ppo" else kernel.steps
        critics = actors if self.role == "ppo" else 0
        if (spent.get(owner + "trajectory", 0) < episode * self.horizon + (active.index if active else 0)
                or spent.get(owner + "actor", 0) < actors or spent.get(owner + "critic", 0) < critics):
            raise ValueError("prefix/tail/optimizer work lacks corresponding owner debit")
        self.kernel, self.active, self.episode = kernel, active, episode
        self.examples, self.receipts = copy.deepcopy(state["examples"]), copy.deepcopy(state["receipts"])
        self.last_closed = copy.deepcopy(state["last_closed"])
