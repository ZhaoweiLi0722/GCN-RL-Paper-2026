"""Opt-in, capped CPU categorical PPO transactions on sealed trajectory receipts.

Reuses existing PPO loss/GAE and request contracts. Checkpoints cover closed
rollout/update boundaries, not the environment or an unfinished collection.
No historical agent, collector, scientific runner or automatic retry is wired in.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
import tempfile

from src.models.candidate_policy import CandidatePolicy
from src.rl.candidate_ppo_objective import candidate_ppo_loss, normalize_rollout_advantages
from src.rl.candidate_rollout import (
    CandidateDecision, PolicyEvaluation, PreparedCandidateSegment, evaluate_candidate_policy,
    prepare_candidate_segment, reevaluate_candidate_decision, sample_candidate,
    verify_behavior_evaluation,
)
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract, _decode_actor_state
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import CandidateChoice, RequestCandidates, RoutingRequestSchema
from src.rl.training_state import training_contract_sha256
from src.rl.validated_returns import OneStepRecord


@dataclass(frozen=True)
class CandidatePPOSettings:
    learning_rate: float
    clip_ratio: float
    value_loss_coef: float
    entropy_coef: float
    max_grad_norm: float
    gae_lambda: float
    normalize_advantages: bool
    epochs: int
    batch_size: int
    max_rollout_steps: int
    max_updates: int
    max_optimizer_steps: int

    def __post_init__(self):
        for name in ("learning_rate", "clip_ratio", "value_loss_coef", "entropy_coef",
                     "max_grad_norm", "gae_lambda"):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite nonnegative number")
        if (self.learning_rate <= 0 or self.max_grad_norm <= 0 or not 0 < self.clip_ratio < 1
                or not 0 <= self.gae_lambda <= 1):
            raise ValueError("invalid learning rate, clipping, gradient norm or GAE lambda")
        if type(self.normalize_advantages) is not bool:
            raise ValueError("advantage normalization must be explicit")
        for name in ("epochs", "batch_size", "max_rollout_steps", "max_updates", "max_optimizer_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.max_rollout_steps or self.epochs > self.max_optimizer_steps:
            raise ValueError("batch/epoch settings exceed declared caps")


def _keys(value, expected, name):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"{name} fields differ")


def _record_id(record):
    return (record.source_id, record.trajectory_id, record.step_index)


class CandidatePPOKernel:
    def __init__(self, prototype, contract, settings, *, enabled=False, mode,
                 sampling_seed, shuffle_seed):
        require_torch()
        if enabled is not True:
            raise ValueError("candidate PPO kernel must be explicitly enabled")
        if (type(prototype) is not CandidatePolicy or not isinstance(contract, ReplayInputContract)
                or not isinstance(settings, CandidatePPOSettings) or prototype.schema != contract.inputs):
            raise ValueError("explicit matching candidate policy, contract and settings required")
        if mode not in ("online", "frozen"):
            raise ValueError("explicit online or frozen mode required")
        for seed in (sampling_seed, shuffle_seed):
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("explicit nonnegative CPU RNG seeds required")
        dtype = next(prototype.parameters()).dtype
        if (dtype not in (torch.float32, torch.float64)
                or any(p.device.type != "cpu" or p.dtype != dtype for p in prototype.parameters())):
            raise ValueError("uniform CPU float32/float64 policy required")
        self.policy = copy.deepcopy(prototype)
        self.contract, self.settings, self.mode = contract, settings, mode
        self._manifest = {
            "format": "candidate-ppo-kernel-v1", "contract": asdict(contract),
            "policy": prototype.manifest(), "initial_policy_sha256": prototype.snapshot_sha256(),
            "settings": asdict(settings), "mode": mode,
            "sampling_seed": sampling_seed, "shuffle_seed": shuffle_seed,
            "optimizer": {"kind": "Adam", "betas": (.9, .999), "eps": 1e-8,
                          "weight_decay": 0, "foreach": False},
        }
        self.manifest_sha256 = training_contract_sha256(self._manifest)
        self.optimizer = (torch.optim.Adam(self.policy.parameters(), lr=settings.learning_rate, foreach=False)
                          if mode == "online" else None)
        self._optimizer_groups = None if self.optimizer is None else copy.deepcopy(
            self.optimizer.state_dict()["param_groups"])
        self.sampling_rng = torch.Generator(device="cpu").manual_seed(sampling_seed)
        self.shuffle_rng = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        self.pending = []
        self.history = []
        self.consumed = []
        self._set_modes()
        state_digest(self.policy.state_dict())

    @property
    def total_updates(self):
        return len(self.history)

    @property
    def total_optimizer_steps(self):
        return sum(self.settings.epochs * math.ceil(n / self.settings.batch_size) for n in self.history)

    @property
    def dtype(self):
        return next(self.policy.parameters()).dtype

    def _set_modes(self):
        self.policy.eval()  # This explicit policy has no dropout or batch normalization.
        self.policy.requires_grad_(self.mode == "online")
        self.policy.zero_grad(set_to_none=True)

    def decide(self, observation, candidates):
        evaluation = evaluate_candidate_policy(self.policy, observation, candidates, self.contract)
        return sample_candidate(evaluation, generator=self.sampling_rng)

    def _observation(self, evaluation):
        states = torch.tensor([evaluation.actor_state], dtype=self.dtype, device="cpu")
        return _decode_actor_state(states, self.contract)[0]

    def _validate_segment(self, segment):
        if not isinstance(segment, PreparedCandidateSegment):
            raise TypeError("prepared closed candidate segment required")
        if segment.gae_lambda != self.settings.gae_lambda:
            raise ValueError("segment GAE lambda differs from kernel contract")
        canonical = prepare_candidate_segment(
            segment.decisions, segment.records, self.contract,
            behavior_sha256=self.policy.snapshot_sha256(), bootstrap=segment.bootstrap,
            gae_lambda=self.settings.gae_lambda, max_steps=self.settings.max_rollout_steps)
        if state_digest(asdict(segment)) != state_digest(asdict(canonical)):
            raise ValueError("segment return/advantage receipt differs from recomputed GAE")
        for decision in segment.decisions:
            evaluation = decision.evaluation
            verify_behavior_evaluation(self.policy, self._observation(evaluation), evaluation)
        if segment.bootstrap is not None:
            verify_behavior_evaluation(self.policy, self._observation(segment.bootstrap), segment.bootstrap)
        return canonical

    def _validate_pending(self, segments):
        if self.mode != "online" and segments:
            raise ValueError("frozen arm cannot retain training rollouts")
        if any(not isinstance(s, PreparedCandidateSegment) for s in segments):
            raise TypeError("prepared closed candidate segment required")
        if sum(len(s.records) for s in segments) > self.settings.max_rollout_steps:
            raise ValueError("rollout exceeds declared step cap")
        canonical = [self._validate_segment(segment) for segment in segments]
        ids = [_record_id(row) for segment in canonical for row in segment.records]
        if len(ids) != len(set(ids)) or set(ids).intersection(self.consumed):
            raise ValueError("duplicate or already consumed trajectory receipt")
        return canonical

    def add_segment(self, segment):
        if self.mode != "online" or self.total_updates >= self.settings.max_updates:
            raise ValueError("frozen arm or update cap prevents rollout admission")
        proposed = self._validate_pending([*self.pending, segment])
        required = self.settings.epochs * math.ceil(
            sum(len(s.records) for s in proposed) / self.settings.batch_size)
        if self.total_optimizer_steps + required > self.settings.max_optimizer_steps:
            raise ValueError("optimizer step cap cannot admit this complete rollout")
        self.pending = proposed

    def _update_in_place(self):
        self.pending = self._validate_pending(self.pending)
        decisions = [d for s in self.pending for d in s.decisions]
        if not decisions:
            raise ValueError("nonempty closed rollout required")
        required = self.settings.epochs * math.ceil(len(decisions) / self.settings.batch_size)
        if self.total_optimizer_steps + required > self.settings.max_optimizer_steps:
            raise ValueError("optimizer step cap would be exceeded")
        advantages = normalize_rollout_advantages(torch.tensor(
            [a for s in self.pending for a in s.advantages], dtype=self.dtype),
            enabled=self.settings.normalize_advantages)
        returns = torch.tensor([v for s in self.pending for v in s.returns], dtype=self.dtype)
        old_logp = torch.tensor([d.old_log_prob for d in decisions], dtype=self.dtype)
        observations = [self._observation(d.evaluation) for d in decisions]
        logs = []
        for epoch in range(self.settings.epochs):
            order = torch.randperm(len(decisions), generator=self.shuffle_rng).tolist()
            for start in range(0, len(order), self.settings.batch_size):
                indices = order[start:start + self.settings.batch_size]
                evaluated = [reevaluate_candidate_decision(self.policy, observations[i], decisions[i])
                             for i in indices]
                logp, values, entropies = [torch.stack(items) for items in zip(*evaluated)]
                losses = candidate_ppo_loss(
                    logp, values, entropies, old_logp[indices], advantages[indices], returns[indices],
                    clip_ratio=self.settings.clip_ratio, value_loss_coef=self.settings.value_loss_coef,
                    entropy_coef=self.settings.entropy_coef)
                self.policy.zero_grad(set_to_none=True)
                losses.total.backward()
                if any(p.grad is None or not torch.isfinite(p.grad).all().item()
                       for p in self.policy.parameters()):
                    raise ValueError("missing/nonfinite PPO gradient")
                norm = torch.nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.settings.max_grad_norm, error_if_nonfinite=True)
                self.optimizer.step()
                state_digest(self.policy.state_dict())
                state_digest(self.optimizer.state_dict())
                logs.append({"epoch": epoch, "indices": indices, "total_loss": losses.total.item(),
                             "policy_loss": losses.policy.item(), "value_loss": losses.value.item(),
                             "entropy": losses.entropy.item(), "clip_fraction": losses.clip_fraction.item(),
                             "grad_norm_before_clip": norm.item()})
        self.history.append(len(decisions))
        self.consumed.extend(_record_id(r) for s in self.pending for r in s.records)
        self.pending = []
        self._set_modes()
        return {"update": self.total_updates, "optimizer_steps": self.total_optimizer_steps,
                "rollout_steps": len(decisions), "minibatches": logs}

    def update(self):
        if self.mode != "online" or self.total_updates >= self.settings.max_updates:
            raise ValueError("frozen arm or declared update cap prevents update")
        # Publish only after the entire rollout update and resulting state validate.
        candidate = copy.deepcopy(self)
        candidate._restore(self.state_dict())
        result = candidate._update_in_place()
        candidate._restore(candidate.state_dict())
        self.__dict__.update(candidate.__dict__)
        return result

    def state_dict(self):
        return copy.deepcopy({
            "manifest": self._manifest, "manifest_sha256": self.manifest_sha256,
            "policy": self.policy.state_dict(),
            "optimizer": None if self.optimizer is None else self.optimizer.state_dict(),
            "pending": [asdict(s) for s in self.pending],
            "history": self.history, "consumed": self.consumed,
            "sampling_rng": self.sampling_rng.get_state(), "shuffle_rng": self.shuffle_rng.get_state(),
        })

    def _evaluation_from_state(self, payload):
        data = dict(payload)
        if data.pop("contract") != asdict(self.contract):
            raise ValueError("saved evaluation contract differs")
        bank = data.pop("candidates")
        schema = RoutingRequestSchema(**bank["schema"])
        candidates = RequestCandidates(schema, bank["state_token"], bank["requests"])
        if state_digest(asdict(candidates)) != state_digest(bank):
            raise ValueError("saved candidate support differs from derived request classes")
        return PolicyEvaluation(contract=self.contract, candidates=candidates, **data)

    def _segment_from_state(self, payload):
        _keys(payload, ("decisions", "records", "bootstrap", "gae_lambda", "advantages", "returns"), "segment")
        if (not 1 <= len(payload["records"]) <= self.settings.max_rollout_steps
                or len(payload["decisions"]) != len(payload["records"])):
            raise ValueError("saved segment outside declared cap or missing receipts")
        decisions = []
        for item in payload["decisions"]:
            _keys(item, ("evaluation", "choice"), "decision")
            decisions.append(CandidateDecision(self._evaluation_from_state(item["evaluation"]),
                                               CandidateChoice(**item["choice"])))
        records = []
        for item in payload["records"]:
            data = dict(item)
            if data.pop("semantics") != asdict(self.contract.replay):
                raise ValueError("saved transition semantics differ")
            records.append(OneStepRecord(semantics=self.contract.replay, **data))
        bootstrap = None if payload["bootstrap"] is None else self._evaluation_from_state(payload["bootstrap"])
        return PreparedCandidateSegment(tuple(decisions), tuple(records), bootstrap, payload["gae_lambda"],
                                        tuple(payload["advantages"]), tuple(payload["returns"]))

    def _load_optimizer(self, saved):
        count = self.total_optimizer_steps
        if self.mode == "frozen":
            if saved is not None or count:
                raise ValueError("frozen checkpoint contains optimizer/updates")
            return
        _keys(saved, ("state", "param_groups"), "optimizer")
        if saved["param_groups"] != self._optimizer_groups:
            raise ValueError("Adam hyperparameters or parameter mapping differ")
        ids = [i for group in saved["param_groups"] for i in group["params"]]
        if set(saved["state"]) != (set(ids) if count else set()):
            raise ValueError("Adam moments missing or unexpected")
        for index, parameter in zip(ids, self.policy.parameters()):
            if not count:
                break
            moments = saved["state"][index]
            _keys(moments, ("step", "exp_avg", "exp_avg_sq"), "Adam moments")
            step = moments["step"]
            if (not isinstance(step, torch.Tensor) or step.dtype != torch.float32
                    or step.device.type != "cpu" or step.shape != torch.Size([]) or step.item() != count):
                raise ValueError("Adam step counter differs from completed minibatches")
            for key in ("exp_avg", "exp_avg_sq"):
                moment = moments[key]
                if (not isinstance(moment, torch.Tensor) or moment.shape != parameter.shape
                        or moment.dtype != parameter.dtype or moment.device.type != "cpu"):
                    raise ValueError("Adam moment shape/dtype/device differs")
            if (moments["exp_avg_sq"] < 0).any().item():
                raise ValueError("negative Adam second moment")
        self.optimizer.load_state_dict(saved)

    def _restore(self, state):
        _keys(state, ("manifest", "manifest_sha256", "policy", "optimizer", "pending", "history",
                      "consumed", "sampling_rng", "shuffle_rng"), "checkpoint")
        if (state["manifest"] != self._manifest or state["manifest_sha256"] != self.manifest_sha256
                or training_contract_sha256(state["manifest"]) != self.manifest_sha256
                or self.policy.manifest() != self._manifest["policy"]):
            raise ValueError("checkpoint semantic/model/optimizer manifest mismatch")
        state_digest(state)
        history = state["history"]
        if (not isinstance(history, list) or len(history) > self.settings.max_updates
                or any(type(n) is not int or not 1 <= n <= self.settings.max_rollout_steps for n in history)
                or (self.mode == "frozen" and history)):
            raise ValueError("invalid completed rollout history")
        self.history = list(history)
        if self.total_optimizer_steps > self.settings.max_optimizer_steps:
            raise ValueError("checkpoint exceeds optimizer step cap")
        consumed = state["consumed"]
        if not isinstance(consumed, list) or len(consumed) != sum(history):
            raise ValueError("consumed receipt count differs from history")
        for identity in consumed:
            if (not isinstance(identity, tuple) or len(identity) != 3
                    or any(not isinstance(v, str) or not v.strip() for v in identity[:2])
                    or type(identity[2]) is not int or identity[2] < 0):
                raise ValueError("invalid consumed receipt identity")
        if len(set(consumed)) != len(consumed):
            raise ValueError("duplicate consumed receipt identity")
        reference = self.policy.state_dict()
        _keys(state["policy"], reference, "policy")
        for key, value in state["policy"].items():
            if (not isinstance(value, torch.Tensor) or value.shape != reference[key].shape
                    or value.dtype != reference[key].dtype or value.device.type != "cpu"):
                raise ValueError("checkpoint policy shape/dtype/device differs")
        self.policy.load_state_dict(state["policy"])
        if not history and self.policy.snapshot_sha256() != self._manifest["initial_policy_sha256"]:
            raise ValueError("zero-update policy differs from initial prototype")
        self._load_optimizer(state["optimizer"])
        self.consumed = list(consumed)
        if not isinstance(state["pending"], list) or len(state["pending"]) > self.settings.max_rollout_steps:
            raise ValueError("invalid pending rollout count")
        self.pending = self._validate_pending([self._segment_from_state(s) for s in state["pending"]])
        if self.pending:
            required = self.settings.epochs * math.ceil(
                sum(len(s.records) for s in self.pending) / self.settings.batch_size)
            if (self.total_updates >= self.settings.max_updates
                    or self.total_optimizer_steps + required > self.settings.max_optimizer_steps):
                raise ValueError("pending rollout cannot fit remaining update budget")
        for key, generator in (("sampling_rng", self.sampling_rng), ("shuffle_rng", self.shuffle_rng)):
            value = state[key]
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8 or value.device.type != "cpu"
                    or value.shape != generator.get_state().shape):
                raise ValueError("invalid private CPU RNG state")
            generator.set_state(value)
        self._set_modes()

    def load_state_dict(self, state):
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(state))
        self.__dict__.update(candidate.__dict__)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        state = self.state_dict()
        copy.deepcopy(self)._restore(copy.deepcopy(state))
        envelope = {"state": state, "state_sha256": state_digest(state)}
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".candidate-ppo-", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(envelope, temporary)
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    def load(self, path):
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        _keys(envelope, ("state", "state_sha256"), "checkpoint envelope")
        if state_digest(envelope["state"]) != envelope["state_sha256"]:
            raise ValueError("checkpoint payload checksum mismatch")
        self.load_state_dict(envelope["state"])
