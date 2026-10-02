"""Unregistered, actor-only full-batch updates over a sealed finite dataset.

Construction only copies a supplied policy and public tensors; it does not load
scientific artifacts, collect branches, or perform a model forward. Each update
is a separate transaction. The caller owns scientific authorization, durable
budget receipts, dataset provenance and process-restart admission.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import math

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import ObservationBatch, build_matched_inputs
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.dynamic_candidate_ppo import _no_compute_check, checked_gradient, validate_adam_state
from src.rl.networks import torch
from src.rl.paired_cohort_objective import PairedCohortState, paired_cohort_objective
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import RequestCandidates


@dataclass(frozen=True)
class PairedCohortActorSettings:
    """Explicit fixed settings; smaller caps/counts support invented unit tests.

    Use ``from_config`` for the prospective numerical package. No defaults or
    adaptive scales are inferred from labels, losses or gradient magnitudes.
    """

    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float
    max_grad_norm: float
    cost_scale: float
    max_optimizer_steps: int
    context_count: int
    future_replications: int

    def __post_init__(self):
        exact = {"learning_rate": 3e-4, "eps": 1e-8, "weight_decay": 0.,
                 "max_grad_norm": .5, "cost_scale": 1e9}
        for name, expected in exact.items():
            value = getattr(self, name)
            if type(value) not in (int, float) or value != expected:
                raise ValueError(f"{name} differs from the fixed paired-cohort protocol")
        if (type(self.betas) is not tuple or self.betas != (.9, .999)
                or any(type(v) not in (int, float) for v in self.betas)):
            raise ValueError("fixed Adam betas required")
        for name, cap in (("max_optimizer_steps", 128), ("context_count", 12)):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= cap:
                raise ValueError(f"{name} exceeds the fixed positive cap")
        if type(self.future_replications) is not int or self.future_replications != 2:
            raise ValueError("exactly two future replications required")

    @classmethod
    def from_config(cls, config):
        optimizer = config["actor_optimizer"]
        if (optimizer["name"] != "Adam" or config["actor_updates_per_arm_block"] != 128
                or config["critic_updates"] != 0 or config["entropy_coefficient"] != 0
                or config["model_dtype"] != "cpu_float32"
                or config["cost_and_objective_dtype"] != "float64_with_differentiable_logit_cast"
                or config["context_cohorts_per_block"] != 4
                or config["context_after_prefix_steps"] != [4, 20, 36]):
            raise ValueError("config differs from fixed actor/full-batch protocol")
        return cls(optimizer["lr"], tuple(optimizer["betas"]), optimizer["eps"],
                   optimizer["weight_decay"], optimizer["gradient_norm_cap"],
                   config["cost_scale"], config["actor_updates_per_arm_block"],
                   config["context_cohorts_per_block"] * len(config["context_after_prefix_steps"]),
                   config["future_replications"])


@dataclass(frozen=True)
class PairedCohortExample:
    """Public inputs and candidate-major paired seed IDs for float64 [R,K] costs.

    The adapter takes detached private copies, seals their content, and never
    exposes those copies. The source's tensors need not be writable or leaf
    tensors. Keys/reference index must exactly match the unfiltered bank.
    """

    observation: ObservationBatch
    bank: RequestCandidates
    raw_costs: torch.Tensor
    replication_seed_ids: tuple[tuple[int | str, ...], ...]
    reference_index: int
    class_keys: tuple


def _sha256(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("fixed lowercase dataset SHA256 required")


def _copy_example(example, schema, settings):
    if type(example) is not PairedCohortExample:
        raise TypeError("explicit PairedCohortExample required")
    obs, bank, costs = example.observation, example.bank, example.raw_costs
    if type(obs) is not ObservationBatch or obs.schema != schema:
        raise ValueError("matching public ObservationBatch required")
    if (type(bank) is not RequestCandidates or bank.schema.max_candidates > 6
            or bank.schema.action_schema_id != schema.definition_id + "/action"
            or bank.schema.num_facilities != len(schema.node_ids)):
        raise ValueError("matching bounded RequestCandidates required")
    canonical = RequestCandidates(bank.schema, bank.state_token, bank.requests)
    if bank != canonical:
        raise ValueError("candidate bank is not canonical")
    if (type(example.class_keys) is not tuple or example.class_keys != bank.class_keys
            or type(example.reference_index) is not int
            or example.reference_index != bank.reference_class):
        raise ValueError("candidate keys or reference index differ from bank")
    if (type(example.replication_seed_ids) is not tuple
            or any(type(row) is not tuple for row in example.replication_seed_ids)):
        raise ValueError("immutable candidate-major paired seed declarations required")
    n = len(schema.node_ids)
    tensors = []
    for name, shape in (("nodes", (1, n, len(schema.node_feature_names))),
                        ("globals", (1, len(schema.global_feature_names))),
                        ("physical_links", (1, n, n))):
        value = getattr(obs, name)
        if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                or value.device.type != "cpu" or value.dtype != torch.float32
                or tuple(value.shape) != shape or not torch.isfinite(value).all().item()):
            raise ValueError(f"finite public CPU float32 {name} of declared shape required")
        tensors.append(value.detach().clone())
    links = tensors[2]
    if ((links < 0).any().item() or not torch.equal(links, links.transpose(1, 2))
            or torch.count_nonzero(links.diagonal(dim1=1, dim2=2)).item()):
        raise ValueError("invalid public physical links")
    if (not isinstance(costs, torch.Tensor) or costs.layout != torch.strided
            or costs.device.type != "cpu" or costs.dtype != torch.float64
            or tuple(costs.shape) != (settings.future_replications, len(bank.class_keys))):
        raise ValueError("raw_costs must be CPU float64 [R,K], without conversion")
    result = PairedCohortExample(ObservationBatch(schema, *tensors), copy.deepcopy(bank),
        costs.detach().clone(), copy.deepcopy(example.replication_seed_ids),
        example.reference_index, copy.deepcopy(example.class_keys))
    # Pure arithmetic validates finite labels, overflow and declared pairing;
    # no scientific model is forwarded during construction or restore.
    paired_cohort_objective((_objective_state(result, torch.zeros(len(bank.class_keys), dtype=torch.float64,
                                                                 device="cpu")),),
                            cost_scale=settings.cost_scale)
    return result


def _objective_state(example, logits):
    return PairedCohortState(logits, example.raw_costs, example.class_keys,
                             example.replication_seed_ids, example.reference_index)


def _examples_digest(examples):
    return state_digest([asdict(example) for example in examples])


def _actor_logits(policy, example):
    """Mirror DynamicCandidatePolicy's public actor path, without a critic call."""
    bank = example.bank
    anchor = torch.tensor([bank.requests[1]], dtype=torch.float32, device="cpu")
    view = build_matched_inputs(example.observation, policy.schema, anchor,
                                role="actor", message_mode=policy.message_mode)
    reference = torch.tensor([bank.requests[0]], dtype=torch.float32, device="cpu")
    features = torch.tensor(bank.class_features, dtype=torch.float32, device="cpu")
    roles = torch.tensor([[i == bank.reference_class, i == bank.anchor_class]
                          for i in range(len(bank.class_keys))], dtype=torch.float32, device="cpu")
    logits = policy.actor(view, reference, features, roles)
    if (logits.device.type != "cpu" or logits.dtype != torch.float32
            or logits.shape != (len(bank.class_keys),) or not torch.isfinite(logits).all().item()):
        raise ValueError("finite CPU float32 actor logits required")
    return logits


class PairedCohortActor:
    """One initializer fork, one fixed dataset, one arm and one hard update cap.

    ``update(before_optimizer_step=hook)`` calls ``hook('actor')`` immediately
    before each attempted Adam step. Attempts include hook failures and cannot
    be refunded by restoring weights. An admitted update exception is terminal.
    Invalid calls/restores are atomic rejections, not new update attempts. A fresh
    process must ALSO reconcile the external non-refundable budget ledger.
    There is no sampler, minibatching, critic optimizer or automatic fit loop.
    """

    def __init__(self, prototype, contract, settings, examples, *, dataset_sha256,
                 arm, enabled=False):
        if (enabled is not True or type(prototype) is not DynamicCandidatePolicy
                or not isinstance(contract, ReplayInputContract) or prototype.schema != contract.inputs
                or type(settings) is not PairedCohortActorSettings):
            raise ValueError("explicit matching dynamic policy, contract and settings required")
        if arm not in ("paired_cost", "bc_continue"):
            raise ValueError("explicit paired_cost or bc_continue arm required")
        _sha256(dataset_sha256)
        if type(examples) is not tuple or len(examples) != settings.context_count:
            raise ValueError("finite immutable tuple containing every configured context required")
        actor, critic = tuple(prototype.actor_parameters()), tuple(prototype.critic_parameters())
        if (not actor or not critic
                or {p.untyped_storage().data_ptr() for p in actor}.intersection(
                    p.untyped_storage().data_ptr() for p in critic)
                or {id(p) for p in (*actor, *critic)} != {id(p) for p in prototype.parameters()}
                or any(p.dtype != torch.float32 or p.device.type != "cpu" for p in (*actor, *critic))):
            raise ValueError("disjoint complete CPU float32 actor/critic ownership required")
        self.settings, self.contract, self.arm = settings, contract, arm
        self._examples = tuple(_copy_example(e, contract.inputs, settings) for e in examples)
        tokens = [e.bank.state_token for e in self._examples]
        if len(tokens) != len(set(tokens)):
            raise ValueError("duplicate training context state tokens")
        self.policy = copy.deepcopy(prototype)
        self._critic_sha256 = state_digest(prototype.critic.state_dict())
        self._manifest = {
            "format": "paired-cohort-actor-v1", "contract": asdict(contract),
            "settings": asdict(settings), "arm": arm, "policy": prototype.manifest(),
            "initial_policy_sha256": prototype.snapshot_sha256(),
            "dataset_sha256": dataset_sha256, "examples_sha256": _examples_digest(self._examples),
            "state_tokens": tokens, "trainable_owner": "actor", "state_weighting": "uniform_full_batch",
            "objective_version": "paired-cohort-objective-v1", "subtract_reference_baseline": True,
            "optimizer": {"kind": "Adam", "foreach": False, "amsgrad": False},
            "rng_scope": "python_numpy_torch_cpu_no_private_sampler",
        }
        self.manifest_sha256 = state_digest(self._manifest)
        self.optimizer = torch.optim.Adam(self.policy.actor_parameters(), lr=settings.learning_rate,
            betas=settings.betas, eps=settings.eps, weight_decay=settings.weight_decay,
            foreach=False, amsgrad=False)
        self._groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        self.steps, self.attempted_steps, self._history, self._failure = 0, 0, [], None
        self._set_modes()

    @property
    def manifest(self):
        return copy.deepcopy(self._manifest)

    def _set_modes(self):
        self.policy.eval()
        self.policy.actor.requires_grad_(True)
        self.policy.critic.requires_grad_(False)
        self.policy.zero_grad(set_to_none=True)

    def _check_binding(self):
        if (state_digest(self._manifest) != self.manifest_sha256
                or asdict(self.settings) != self._manifest["settings"]
                or asdict(self.contract) != self._manifest["contract"] or self.arm != self._manifest["arm"]
                or self.policy.manifest() != self._manifest["policy"]
                or _examples_digest(self._examples) != self._manifest["examples_sha256"]):
            raise ValueError("definition or immutable data binding differs")
        if state_digest(self.policy.critic.state_dict()) != self._critic_sha256:
            raise ValueError("frozen critic differs")
        parameters = [p for group in self.optimizer.param_groups for p in group["params"]]
        if [id(p) for p in parameters] != [id(p) for p in self.policy.actor_parameters()]:
            raise ValueError("optimizer must own only the current actor parameters")
        if self.optimizer.state_dict()["param_groups"] != self._groups:
            raise ValueError("Adam hyperparameters differ")

    def loss(self, *, before_compute=_no_compute_check):
        """Differentiable full-batch loss; no debit or optimizer step.

        This is a model forward and is subject to the caller's scientific
        authorization, just like update. Labels and public inputs are detached.
        """
        if self._failure is not None:
            raise ValueError("failed transaction is terminal")
        before_compute()
        self._check_binding()
        states = []
        for example in self._examples:
            before_compute()
            states.append(_objective_state(example, _actor_logits(self.policy, example).to(torch.float64)))
        before_compute()
        if self.arm == "paired_cost":
            return paired_cohort_objective(tuple(states), cost_scale=self.settings.cost_scale).loss
        loss = torch.stack([-torch.log_softmax(s.logits, dim=0)[s.reference_index] for s in states]).mean()
        if not torch.isfinite(loss).item():
            raise ValueError("nonfinite BC loss")
        return loss

    def update(self, *, before_optimizer_step, before_compute=_no_compute_check):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal; cannot retry")
        if not callable(before_optimizer_step) or not callable(before_compute):
            raise ValueError("explicit durable charge hook and callable compute check required")
        if self.attempted_steps >= self.settings.max_optimizer_steps:
            raise ValueError("actor attempted-step cap exhausted")
        rng = global_rng_state()
        attempted, returned = False, False
        try:
            before_compute()
            candidate = copy.deepcopy(self)
            candidate._check_binding()
            candidate._restore(candidate.state_dict())
            candidate._set_modes()
            loss = candidate.loss(before_compute=before_compute)
            before_compute()
            loss.backward()
            before_compute()
            norm = checked_gradient(candidate.policy.actor_parameters(), candidate.policy.critic_parameters(),
                                    self.settings.max_grad_norm)
            before_compute()
            # Charge before calling the hook: it may persist a debit then raise.
            attempted = True
            self.attempted_steps += 1
            before_optimizer_step("actor")
            before_compute()
            candidate.optimizer.step()
            returned = True
            candidate.steps += 1
            candidate.attempted_steps = self.attempted_steps
            row = {"step": candidate.steps, "loss": loss.item(), "actor_grad_norm_before_clip": norm}
            candidate._history.append(row)
            before_compute()
            candidate._check_binding()
            candidate._set_modes()
            candidate._restore(candidate.state_dict())
            before_compute()
        except BaseException as error:
            self._failure = {"error_type": type(error).__name__, "message": str(error),
                             "attempted_this_update": attempted, "optimizer_step_returned": returned,
                             "attempted_steps": self.attempted_steps, "committed_steps": self.steps,
                             "resource_authority": "external_non_refundable_ledger"}
            restore_global_rng(rng)
            if state_digest(global_rng_state()) != state_digest(rng):
                raise RuntimeError("global RNG rollback mismatch; transaction remains terminal") from error
            raise
        self.__dict__.update(candidate.__dict__)
        return dict(row, optimizer_steps=self.steps, attempted_optimizer_steps=self.attempted_steps,
                    critic_optimizer_steps=0, context_count=len(self._examples), arm=self.arm)

    def state_dict(self):
        """Detached boundary snapshot; input data are rebound by their two hashes."""
        return copy.deepcopy({"manifest": self._manifest, "manifest_sha256": self.manifest_sha256,
            "policy": self.policy.state_dict(), "optimizer": self.optimizer.state_dict(),
            "steps": self.steps, "attempted_steps": self.attempted_steps, "history": self._history,
            "global_rng": global_rng_state(), "failure": self._failure})

    def _restore(self, state):
        keys = {"manifest", "manifest_sha256", "policy", "optimizer", "steps", "attempted_steps",
                "history", "global_rng", "failure"}
        if (type(state) is not dict or set(state) != keys
                or state["manifest_sha256"] != self.manifest_sha256
                or state_digest(state["manifest"]) != self.manifest_sha256):
            raise ValueError("checkpoint definition or data binding differs")
        if state["failure"] is not None:
            raise ValueError("failed checkpoint is evidence only; cannot restore to retry")
        state_digest(state)
        steps, attempted, history = state["steps"], state["attempted_steps"], state["history"]
        if (type(steps) is not int or type(attempted) is not int
                or not 0 <= steps == attempted <= self.settings.max_optimizer_steps
                or steps < self.steps or attempted < self.attempted_steps):
            raise ValueError("invalid counters or non-refundable update rewind")
        if (type(history) is not list or len(history) != steps
                or history[:len(self._history)] != self._history):
            raise ValueError("update history rewind or divergent prefix")
        for index, row in enumerate(history, 1):
            if (type(row) is not dict or set(row) != {"step", "loss", "actor_grad_norm_before_clip"}
                    or type(row["step"]) is not int or row["step"] != index
                    or any(type(row[k]) not in (int, float) or not math.isfinite(row[k])
                           for k in ("loss", "actor_grad_norm_before_clip"))
                    or row["actor_grad_norm_before_clip"] < 0):
                raise ValueError("invalid full-batch update history")
        if steps == self.steps and (state_digest(state["policy"]) != state_digest(self.policy.state_dict())
                                   or state_digest(state["optimizer"]) != state_digest(self.optimizer.state_dict())):
            raise ValueError("same-counter model/optimizer boundary differs")
        expected = self.policy.state_dict()
        if type(state["policy"]) not in (dict, type(expected)) or set(state["policy"]) != set(expected):
            raise ValueError("policy tensor keys differ")
        for name, tensor in state["policy"].items():
            if (not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu"
                    or tensor.layout != torch.strided or tensor.dtype != expected[name].dtype
                    or tensor.shape != expected[name].shape):
                raise ValueError("policy tensor contract differs")
        self.policy.load_state_dict(state["policy"])
        if not steps and self.policy.snapshot_sha256() != self._manifest["initial_policy_sha256"]:
            raise ValueError("zero-step initializer differs")
        validate_adam_state(self.optimizer, state["optimizer"], self.policy.actor_parameters(), self._groups, steps)
        self.steps, self.attempted_steps, self._history = steps, attempted, copy.deepcopy(history)
        self._set_modes()
        self._check_binding()
        if (state_digest(self.policy.state_dict()) != state_digest(state["policy"])
                or state_digest(self.optimizer.state_dict()) != state_digest(state["optimizer"])):
            raise ValueError("model/optimizer restore readback mismatch")
        rng = global_rng_state()
        try:
            if type(state["global_rng"]) is not dict or set(state["global_rng"]) != {"python", "numpy", "torch"}:
                raise ValueError("global RNG contract differs")
            restore_global_rng(state["global_rng"])
            if state_digest(global_rng_state()) != state_digest(state["global_rng"]):
                raise ValueError("global RNG restore readback mismatch")
        finally:
            restore_global_rng(rng)

    def load_state_dict(self, state):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal; cannot restore to retry")
        rng = global_rng_state()
        try:
            state = copy.deepcopy(state)
            self._check_binding()
            candidate = copy.deepcopy(self)
            candidate._restore(state)
            restore_global_rng(state["global_rng"])
            if state_digest(candidate.state_dict()) != state_digest(state):
                raise ValueError("full restore readback mismatch")
        except BaseException:
            restore_global_rng(rng)
            raise
        self.__dict__.update(candidate.__dict__)
