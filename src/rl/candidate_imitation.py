"""Bounded reference-class imitation, separate from return-based PPO updates."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

from src.models.candidate_policy import CandidatePolicy
from src.models.reference_prior_candidate import ReferencePriorCandidatePolicy
from src.rl.networks import torch
from src.rl.candidate_rollout import evaluate_candidate_policy, sample_candidate
from src.rl.prospective_adapter import _decode_actor_state, pack_actor_state
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import RequestCandidates, RoutingRequestSchema


def public_example(observation, candidates, contract, *, split, identity):
    if split not in ("demonstration", "training", "qualification", "test"):
        raise ValueError("explicit data split required")
    if not isinstance(identity, str) or not identity:
        raise ValueError("explicit example identity required")
    anchor = torch.tensor([candidates.requests[1]], dtype=observation.nodes.dtype)
    state = pack_actor_state(observation, anchor, contract)
    return {"split": split, "identity": identity,
            "actor_state": tuple(state[0].tolist()), "candidates": asdict(candidates)}


def decode_example(example, contract):
    if set(example) != {"split", "identity", "actor_state", "candidates"}:
        raise ValueError("public example must not contain outcome labels or hidden metadata")
    if not isinstance(example["identity"], str) or not example["identity"]:
        raise ValueError("example identity missing")
    bank = example["candidates"]
    candidates = RequestCandidates(RoutingRequestSchema(**bank["schema"]), bank["state_token"], bank["requests"])
    if state_digest(asdict(candidates)) != state_digest(bank):
        raise ValueError("candidate payload mismatch")
    state = torch.tensor([example["actor_state"]], dtype=torch.float32)
    observation, anchor = _decode_actor_state(state, contract)
    if not torch.equal(anchor, torch.tensor([candidates.requests[1]], dtype=torch.float32)):
        raise ValueError("example anchor differs from support")
    if candidates.schema.action_schema_id != contract.replay.action_schema_id:
        raise ValueError("example action contract mismatch")
    return observation, candidates


@dataclass(frozen=True)
class ImitationSettings:
    learning_rate: float
    max_grad_norm: float
    batch_size: int
    max_optimizer_steps: int
    max_updates: int
    allowed_split: str

    def __post_init__(self):
        for value in (self.learning_rate, self.max_grad_norm):
            if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite optimizer constants required")
        for value in (self.batch_size, self.max_optimizer_steps, self.max_updates):
            if type(value) is not int or value < 1:
                raise ValueError("positive explicit bounds required")
        if self.allowed_split not in ("demonstration", "training"):
            raise ValueError("qualification/test fitting forbidden")


class CandidateImitationKernel:
    def __init__(self, prototype, contract, settings, *, enabled=False, shuffle_seed, sampling_seed=None):
        if (enabled is not True or type(prototype) not in (CandidatePolicy, ReferencePriorCandidatePolicy)
                or not isinstance(settings, ImitationSettings) or not isinstance(contract, ReplayInputContract)
                or prototype.schema != contract.inputs
                or type(shuffle_seed) is not int or not 0 <= shuffle_seed < 2**63):
            raise ValueError("explicit compatible imitation contract required")
        if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in prototype.parameters()):
            raise ValueError("CPU float32 required")
        self.policy = copy.deepcopy(prototype)
        self.contract, self.settings = contract, settings
        self.manifest = {"format": "candidate-imitation-v1", "contract": asdict(contract),
                         "settings": asdict(settings), "policy": prototype.manifest(),
                         "initial_weights": prototype.snapshot_sha256(), "shuffle_seed": shuffle_seed,
                         "sampling_seed": sampling_seed}
        if sampling_seed is not None and (type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63):
            raise ValueError("invalid private sampling seed")
        self.sampling_rng = None if sampling_seed is None else torch.Generator(device="cpu").manual_seed(sampling_seed)
        self._value_state = copy.deepcopy(self.policy.value_head.state_dict())
        self.policy.requires_grad_(True)
        self.policy.value_head.requires_grad_(False)
        self.policy.eval()
        self.optimizer = torch.optim.Adam(self.trainable(), lr=settings.learning_rate, foreach=False)
        self._groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        self.rng = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        self.steps, self.history = 0, []

    def trainable(self):
        return [p for name, p in self.policy.named_parameters() if not name.startswith("value_head.")]

    def decide(self, observation, candidates):
        if self.sampling_rng is None:
            raise ValueError("initialization kernel has no collection sampler")
        evaluation = evaluate_candidate_policy(self.policy, observation, candidates, self.contract)
        return sample_candidate(evaluation, generator=self.sampling_rng)

    def _fit(self, examples, *, epochs, replacement_steps, before_step):
        if self.settings.allowed_split == "demonstration":
            if epochs is not None or type(replacement_steps) is not int or replacement_steps < 1:
                raise ValueError("demonstration fit requires a fixed replacement-update count")
            count = replacement_steps
        else:
            if replacement_steps is not None or type(epochs) is not int or epochs < 1:
                raise ValueError("continuation requires explicit whole epochs")
            count = epochs * math.ceil(len(examples) / self.settings.batch_size)
        if (not examples or self.steps + count > self.settings.max_optimizer_steps
                or len(self.history) >= self.settings.max_updates):
            raise ValueError("empty data or imitation budget exceeded")
        ids = [e["identity"] for e in examples]
        if len(set(ids)) != len(ids) or any(e["split"] != self.settings.allowed_split for e in examples):
            raise ValueError("mixed/forbidden split or duplicate examples")
        if set(ids).intersection(i for entry in self.history for i in entry["identities"]):
            raise ValueError("already consumed imitation rollout")
        decoded = [decode_example(e, self.contract) for e in examples]
        batches = []
        if replacement_steps is not None:
            batches = [torch.randint(len(examples), (self.settings.batch_size,), generator=self.rng).tolist()
                       for _ in range(replacement_steps)]
        else:
            for _ in range(epochs):
                order = torch.randperm(len(examples), generator=self.rng).tolist()
                batches.extend(order[i:i + self.settings.batch_size] for i in range(0, len(order), self.settings.batch_size))
        logs = []
        for indices in batches:
            losses = []
            for i in indices:
                obs, bank = decoded[i]
                logits = self.policy(obs, bank).logits
                losses.append(-torch.log_softmax(logits, dim=0)[bank.reference_class])
            loss = torch.stack(losses).mean()
            self.policy.zero_grad(set_to_none=True)
            loss.backward()
            params = self.trainable()
            if any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in params):
                raise ValueError("missing/nonfinite imitation gradient")
            norm = torch.nn.utils.clip_grad_norm_(params, self.settings.max_grad_norm, error_if_nonfinite=True)
            before_step()
            self.optimizer.step()
            self.steps += 1
            if any(not torch.isfinite(p).all().item() for p in params):
                raise ValueError("nonfinite imitation parameter")
            logs.append({"indices": indices, "cross_entropy": loss.item(), "grad_norm": norm.item()})
        if state_digest(self.policy.value_head.state_dict()) != state_digest(self._value_state):
            raise ValueError("imitation changed value-head weights")
        self.history.append({"identities": ids, "examples_sha256": state_digest(examples), "steps": count})
        self.policy.zero_grad(set_to_none=True)
        return {"optimizer_steps": self.steps, "minibatches": logs}

    def fit(self, examples, *, epochs=None, replacement_steps=None, before_step=lambda: None):
        candidate = copy.deepcopy(self)
        result = candidate._fit(examples, epochs=epochs, replacement_steps=replacement_steps, before_step=before_step)
        candidate._restore(candidate.state_dict())
        self.__dict__.update(candidate.__dict__)
        return result

    def state_dict(self):
        return copy.deepcopy({"manifest": self.manifest, "policy": self.policy.state_dict(),
                              "optimizer": self.optimizer.state_dict(), "rng": self.rng.get_state(),
                              "sampling_rng": None if self.sampling_rng is None else self.sampling_rng.get_state(),
                              "steps": self.steps, "history": self.history})

    def _restore(self, state):
        if set(state) != {"manifest", "policy", "optimizer", "rng", "sampling_rng", "steps", "history"} or state["manifest"] != self.manifest:
            raise ValueError("imitation state manifest mismatch")
        state_digest(state)
        steps, history = state["steps"], state["history"]
        if (type(steps) is not int or not 0 <= steps <= self.settings.max_optimizer_steps
                or not isinstance(history, list) or len(history) > self.settings.max_updates
                or any(not isinstance(h, dict) or set(h) != {"steps", "identities", "examples_sha256"}
                       or type(h["steps"]) is not int or h["steps"] < 1
                       or not isinstance(h["identities"], list) or not h["identities"]
                       or any(not isinstance(i, str) or not i for i in h["identities"])
                       or not isinstance(h["examples_sha256"], str) or len(h["examples_sha256"]) != 64
                       or any(c not in "0123456789abcdef" for c in h["examples_sha256"]) for h in history)
                or sum(h["steps"] for h in history) != steps):
            raise ValueError("invalid imitation counters")
        ids = [i for h in history for i in h["identities"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate consumed imitation lineage")
        expected = self.policy.state_dict()
        if set(state["policy"]) != set(expected):
            raise ValueError("policy keys differ")
        for key, value in state["policy"].items():
            if (not isinstance(value, torch.Tensor) or value.dtype != expected[key].dtype
                    or value.shape != expected[key].shape or value.device.type != "cpu"):
                raise ValueError("policy tensor contract mismatch")
        self.policy.load_state_dict(state["policy"])
        if state_digest(self.policy.value_head.state_dict()) != state_digest(self._value_state):
            raise ValueError("value head differs from initializer")
        if not steps and self.policy.snapshot_sha256() != self.manifest["initial_weights"]:
            raise ValueError("zero-step weights differ")
        saved = state["optimizer"]
        if set(saved) != {"state", "param_groups"} or saved["param_groups"] != self._groups:
            raise ValueError("Adam settings differ")
        indices = [i for group in self._groups for i in group["params"]]
        if set(saved["state"]) != (set(indices) if steps else set()):
            raise ValueError("Adam state coverage differs")
        for index, parameter in zip(indices, self.trainable()):
            if not steps:
                break
            moments = saved["state"][index]
            if (set(moments) != {"step", "exp_avg", "exp_avg_sq"}
                    or not isinstance(moments["step"], torch.Tensor) or moments["step"].shape != torch.Size([])
                    or moments["step"].dtype != torch.float32 or moments["step"].device.type != "cpu"
                    or moments["step"].item() != steps):
                raise ValueError("Adam step mismatch")
            for name in ("exp_avg", "exp_avg_sq"):
                if (not isinstance(moments[name], torch.Tensor) or moments[name].shape != parameter.shape
                        or moments[name].dtype != parameter.dtype or moments[name].device.type != "cpu"):
                    raise ValueError("Adam moment shape/precision mismatch")
            if (moments["exp_avg_sq"] < 0).any().item():
                raise ValueError("negative second moment")
        self.optimizer.load_state_dict(saved)
        if (not isinstance(state["rng"], torch.Tensor) or state["rng"].dtype != torch.uint8
                or state["rng"].device.type != "cpu" or state["rng"].shape != self.rng.get_state().shape):
            raise ValueError("invalid private RNG")
        self.rng.set_state(state["rng"])
        if self.sampling_rng is None:
            if state["sampling_rng"] is not None:
                raise ValueError("unexpected sampler in initializer")
        else:
            if (not isinstance(state["sampling_rng"], torch.Tensor) or state["sampling_rng"].dtype != torch.uint8
                    or state["sampling_rng"].device.type != "cpu"
                    or state["sampling_rng"].shape != self.sampling_rng.get_state().shape):
                raise ValueError("invalid private sampling RNG")
            self.sampling_rng.set_state(state["sampling_rng"])
        self.steps, self.history = steps, copy.deepcopy(history)
        self.policy.zero_grad(set_to_none=True)

    def load_state_dict(self, state):
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(state))
        self.__dict__.update(candidate.__dict__)
