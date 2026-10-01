"""Separate-owner PPO transactions for the unregistered dynamic candidate model.

Reuses closed-segment admission, serialization and copy-on-success publication.
One minibatch consumes TWO optimizer calls. External durable accounting is
mandatory; rollback of the model never refunds charges or permits a retry.
"""

import copy
from contextlib import contextmanager
from dataclasses import asdict
import math

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_ppo_kernel import CandidatePPOKernel, CandidatePPOSettings, _keys, _record_id
from src.rl.candidate_ppo_objective import candidate_ppo_loss, normalize_rollout_advantages
from src.rl.candidate_rollout import PreparedCandidateSegment, prepare_candidate_segment, sample_candidate
from src.rl.dynamic_candidate_rollout import (
    evaluate_dynamic_policy, reevaluate_dynamic_decision, verify_dynamic_evaluation,
)
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.training_state import training_contract_sha256


def _no_compute_check():
    pass


@contextmanager
def _compute_checks(kernel, before_compute):
    """Bridge inherited restore hooks without retaining the external budget.

    Forward guards also cover receipt adapters and the independent critic after
    the actor returns. They are removed before publication or failure capture.
    """
    handles = []
    previous = getattr(kernel, "_before_compute", None)
    kernel._before_compute = before_compute

    def check_forward(module, inputs):
        before_compute()

    try:
        for module in (kernel.policy, kernel.policy.actor, kernel.policy.critic):
            handles.append(module.register_forward_pre_hook(check_forward))
        yield
    finally:
        for handle in handles:
            handle.remove()
        if previous is None:
            del kernel._before_compute
        else:
            kernel._before_compute = previous


def validate_adam_state(optimizer, saved, parameters, expected_groups, count, *,
                        before_compute=_no_compute_check):
    """Validate each owner's independent counter and moments before loading."""
    before_compute()
    _keys(saved, ("state", "param_groups"), "optimizer")
    if saved["param_groups"] != expected_groups:
        raise ValueError("Adam hyperparameters or parameter mapping differ")
    ids = [i for group in expected_groups for i in group["params"]]
    parameters = tuple(parameters)
    if len(ids) != len(parameters) or set(saved["state"]) != (set(ids) if count else set()):
        raise ValueError("Adam state coverage differs")
    for index, parameter in zip(ids, parameters):
        if not count:
            break
        before_compute()
        moments = saved["state"][index]
        _keys(moments, ("step", "exp_avg", "exp_avg_sq"), "Adam moments")
        step = moments["step"]
        if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32
                or step.device.type != "cpu" or step.shape != torch.Size([]) or step.item() != count):
            raise ValueError("Adam owner step counter differs")
        for key in ("exp_avg", "exp_avg_sq"):
            moment = moments[key]
            if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                    or moment.dtype != parameter.dtype or moment.device.type != "cpu"):
                raise ValueError("Adam moment shape/dtype/device differs")
        if (moments["exp_avg_sq"] < 0).any().item():
            raise ValueError("negative Adam second moment")
    before_compute()
    state_digest(saved)
    before_compute()
    optimizer.load_state_dict(saved)


def checked_gradient(parameters, other, max_norm):
    parameters, other = tuple(parameters), tuple(other)
    if any(p.grad is not None for p in other):
        raise ValueError("gradient crossed actor/critic ownership")
    if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters):
        raise ValueError("missing/nonfinite gradient")
    return torch.nn.utils.clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True).item()


class DynamicCandidatePPOKernel(CandidatePPOKernel):
    def __init__(self, prototype, contract, settings, *, enabled=False, mode,
                 sampling_seed, shuffle_seed):
        require_torch()
        if (enabled is not True or type(prototype) is not DynamicCandidatePolicy
                or not isinstance(contract, ReplayInputContract)
                or not isinstance(settings, CandidatePPOSettings) or prototype.schema != contract.inputs):
            raise ValueError("explicit matching dynamic policy, contract and settings required")
        if mode not in ("online", "frozen"):
            raise ValueError("explicit online or frozen mode required")
        for seed in (sampling_seed, shuffle_seed):
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("explicit nonnegative CPU RNG seeds required")
        dtype = next(prototype.parameters()).dtype
        if (dtype not in (torch.float32, torch.float64)
                or any(p.device.type != "cpu" or p.dtype != dtype for p in prototype.parameters())):
            raise ValueError("uniform CPU float32/float64 policy required")
        actor, critic = tuple(prototype.actor_parameters()), tuple(prototype.critic_parameters())
        storages = lambda ps: {p.untyped_storage().data_ptr() for p in ps}
        if (not actor or not critic or storages(actor).intersection(storages(critic))
                or {id(p) for p in (*actor, *critic)} != {id(p) for p in prototype.parameters()}):
            raise ValueError("actor and critic must have disjoint complete parameter ownership")
        self.policy = copy.deepcopy(prototype)
        self.contract, self.settings, self.mode = contract, settings, mode
        self._manifest = {
            "format": "dynamic-candidate-ppo-kernel-v1", "contract": asdict(contract),
            "policy": prototype.manifest(), "initial_policy_sha256": prototype.snapshot_sha256(),
            "settings": asdict(settings), "mode": mode, "sampling_seed": sampling_seed,
            "shuffle_seed": shuffle_seed, "calls_per_minibatch": 2,
            "optimizer": {"kind": "Adam", "betas": (.9, .999), "eps": 1e-8,
                          "weight_decay": 0, "foreach": False, "owners": ("actor", "critic")},
        }
        self.manifest_sha256 = training_contract_sha256(self._manifest)
        self.actor_optimizer = self.critic_optimizer = None
        self._owner_groups = {}
        if mode == "online":
            self.actor_optimizer = torch.optim.Adam(self.policy.actor_parameters(),
                                                    lr=settings.learning_rate, foreach=False)
            self.critic_optimizer = torch.optim.Adam(self.policy.critic_parameters(),
                                                     lr=settings.learning_rate, foreach=False)
            self._owner_groups = {name: copy.deepcopy(opt.state_dict()["param_groups"])
                                  for name, opt in self.optimizers.items()}
        self.sampling_rng = torch.Generator(device="cpu").manual_seed(sampling_seed)
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        self.pending, self.history, self.consumed = [], [], []
        self._failure = None
        self._set_modes()
        state_digest(self.policy.state_dict())

    @property
    def optimizers(self):
        return {"actor": self.actor_optimizer, "critic": self.critic_optimizer}

    @property
    def total_owner_steps(self):
        return sum(self.settings.epochs * math.ceil(n / self.settings.batch_size) for n in self.history)

    @property
    def total_optimizer_steps(self):
        return 2 * self.total_owner_steps

    def _required(self, segments):
        return 2 * self.settings.epochs * math.ceil(
            sum(len(s.records) for s in segments) / self.settings.batch_size)

    def decide(self, observation, candidates):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal")
        evaluation = evaluate_dynamic_policy(self.policy, observation, candidates, self.contract)
        return sample_candidate(evaluation, generator=self.sampling_rng)

    def _validate_segment(self, segment):
        before_compute = getattr(self, "_before_compute", _no_compute_check)
        before_compute()
        if not isinstance(segment, PreparedCandidateSegment):
            raise TypeError("prepared closed candidate segment required")
        if segment.gae_lambda != self.settings.gae_lambda:
            raise ValueError("segment GAE lambda differs")
        canonical = prepare_candidate_segment(
            segment.decisions, segment.records, self.contract,
            behavior_sha256=self.policy.snapshot_sha256(), bootstrap=segment.bootstrap,
            gae_lambda=self.settings.gae_lambda, max_steps=self.settings.max_rollout_steps)
        if state_digest(asdict(segment)) != state_digest(asdict(canonical)):
            raise ValueError("segment return/advantage receipt differs")
        for decision in segment.decisions:
            before_compute()
            verify_dynamic_evaluation(self.policy, self._observation(decision.evaluation), decision.evaluation)
        if segment.bootstrap is not None:
            before_compute()
            verify_dynamic_evaluation(self.policy, self._observation(segment.bootstrap), segment.bootstrap)
        return canonical

    def add_segment(self, segment):
        if self._failure is not None or self.mode != "online" or self.total_updates >= self.settings.max_updates:
            raise ValueError("failed/frozen arm or update cap prevents admission")
        proposed = self._validate_pending([*self.pending, segment])
        if self.total_optimizer_steps + self._required(proposed) > self.settings.max_optimizer_steps:
            raise ValueError("two-owner optimizer step cap cannot admit rollout")
        self.pending = proposed

    def _update_in_place(self, before_optimizer_step, before_minibatch, before_compute):
        before_compute()
        self.pending = self._validate_pending(self.pending)
        decisions = [d for s in self.pending for d in s.decisions]
        if not decisions:
            raise ValueError("nonempty closed rollout required")
        if self.total_optimizer_steps + self._required(self.pending) > self.settings.max_optimizer_steps:
            raise ValueError("two-owner optimizer cap exceeded")
        before_compute()
        advantages = normalize_rollout_advantages(torch.tensor(
            [a for s in self.pending for a in s.advantages], dtype=self.dtype),
            enabled=self.settings.normalize_advantages)
        returns = torch.tensor([r for s in self.pending for r in s.returns], dtype=self.dtype)
        old_logp = torch.tensor([d.old_log_prob for d in decisions], dtype=self.dtype)
        observations = []
        for decision in decisions:
            before_compute()
            observations.append(self._observation(decision.evaluation))
        logs = []
        for epoch in range(self.settings.epochs):
            before_compute()
            order = torch.randperm(len(decisions), generator=self.shuffle_rng).tolist()
            for start in range(0, len(order), self.settings.batch_size):
                indices = order[start:start + self.settings.batch_size]
                before_minibatch()  # Check both available slots/deadline without refunding either owner.
                evaluated = []
                for i in indices:
                    before_compute()
                    evaluated.append(reevaluate_dynamic_decision(self.policy, observations[i], decisions[i]))
                before_compute()
                logp, values, entropies = [torch.stack(v) for v in zip(*evaluated)]
                before_compute()
                losses = candidate_ppo_loss(
                    logp, values, entropies, old_logp[indices], advantages[indices], returns[indices],
                    clip_ratio=self.settings.clip_ratio, value_loss_coef=self.settings.value_loss_coef,
                    entropy_coef=self.settings.entropy_coef)
                self.policy.zero_grad(set_to_none=True)
                before_compute()
                (losses.policy - self.settings.entropy_coef * losses.entropy).backward()
                before_compute()
                actor_norm = checked_gradient(self.policy.actor_parameters(), self.policy.critic_parameters(),
                                              self.settings.max_grad_norm)
                before_compute()
                before_optimizer_step("actor")
                before_compute()
                self.actor_optimizer.step()
                before_compute()
                state_digest(self.policy.state_dict())
                state_digest(self.actor_optimizer.state_dict())
                self.policy.zero_grad(set_to_none=True)
                before_compute()
                (self.settings.value_loss_coef * losses.value).backward()
                before_compute()
                critic_norm = checked_gradient(self.policy.critic_parameters(), self.policy.actor_parameters(),
                                               self.settings.max_grad_norm)
                before_compute()
                before_optimizer_step("critic")
                before_compute()
                self.critic_optimizer.step()
                before_compute()
                state_digest(self.policy.state_dict())
                state_digest(self.critic_optimizer.state_dict())
                logs.append({"epoch": epoch, "indices": indices,
                             "policy_loss": losses.policy.item(), "value_loss": losses.value.item(),
                             "entropy": losses.entropy.item(), "clip_fraction": losses.clip_fraction.item(),
                             "actor_grad_norm_before_clip": actor_norm,
                             "critic_grad_norm_before_clip": critic_norm})
        before_compute()
        self.history.append(len(decisions))
        self.consumed.extend(_record_id(r) for s in self.pending for r in s.records)
        self.pending = []
        self._set_modes()
        return {"update": self.total_updates, "optimizer_steps": self.total_optimizer_steps,
                "actor_optimizer_steps": self.total_owner_steps, "critic_optimizer_steps": self.total_owner_steps,
                "rollout_steps": len(decisions), "minibatches": logs}

    def update(self, *, before_optimizer_step, before_minibatch, before_compute=_no_compute_check):
        """Check compute deadlines independently of durable per-owner charging."""
        if not callable(before_optimizer_step) or not callable(before_minibatch):
            raise ValueError("explicit durable charge and pair-cap callbacks required")
        if not callable(before_compute):
            raise ValueError("before_compute must be callable")
        if self._failure is not None or self.mode != "online" or self.total_updates >= self.settings.max_updates:
            raise ValueError("failed/frozen arm or declared update cap prevents update")
        candidate = self
        attempted = []

        def charge(owner):
            before_optimizer_step(owner)
            attempted.append(owner)

        try:
            before_compute()
            candidate = copy.deepcopy(self)
            with _compute_checks(candidate, before_compute):
                before_compute()
                candidate._restore(self.state_dict())
                result = candidate._update_in_place(charge, before_minibatch, before_compute)
                before_compute()
                candidate._restore(candidate.state_dict())
                before_compute()
        except BaseException as error:
            # Failure evidence must remain available even after the deadline.
            self._failure = {"error_type": type(error).__name__, "message": str(error),
                             "acknowledged_charge_owners": attempted,
                             "resource_authority": "external_non_refundable_ledger",
                             "partial_transaction": candidate.state_dict()}
            raise
        self.__dict__.update(candidate.__dict__)
        return result

    def state_dict(self):
        return copy.deepcopy({
            "manifest": self._manifest, "manifest_sha256": self.manifest_sha256,
            "policy": self.policy.state_dict(),
            "optimizer": None if self.mode == "frozen" else {
                owner: optimizer.state_dict() for owner, optimizer in self.optimizers.items()},
            "pending": [asdict(s) for s in self.pending], "history": self.history,
            "consumed": self.consumed, "sampling_rng": self.sampling_rng.get_state(),
            "shuffle_rng": self.shuffle_rng.get_state(), "failure": self._failure,
        })

    def _load_optimizer(self, saved):
        before_compute = getattr(self, "_before_compute", _no_compute_check)
        before_compute()
        if self.mode == "frozen":
            if saved is not None or self.total_optimizer_steps:
                raise ValueError("frozen checkpoint contains optimizers")
            return
        _keys(saved, ("actor", "critic"), "optimizer owners")
        for owner, optimizer in self.optimizers.items():
            parameters = getattr(self.policy, owner + "_parameters")()
            validate_adam_state(optimizer, saved[owner], parameters,
                                self._owner_groups[owner], self.total_owner_steps,
                                before_compute=before_compute)

    def _restore(self, state):
        before_compute = getattr(self, "_before_compute", _no_compute_check)
        before_compute()
        if not isinstance(state, dict) or "failure" not in state:
            raise ValueError("missing transaction failure state")
        failure = state["failure"]
        # Failed partial payloads are evidence, never resumable optimizer states.
        if failure is not None:
            raise ValueError("failed transaction evidence cannot resume")
        base = {k: v for k, v in state.items() if k != "failure"}
        super()._restore(base)
        before_compute()
        if self.pending and self.total_optimizer_steps + self._required(self.pending) > self.settings.max_optimizer_steps:
            raise ValueError("pending two-owner rollout exceeds cap")
        self._failure = None

    def load_state_dict(self, state):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal; cannot restore to retry")
        super().load_state_dict(state)
