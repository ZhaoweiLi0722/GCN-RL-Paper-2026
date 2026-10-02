"""In-memory, evaluation-only restoration of the four cohort model owners.

The caller verifies the saved envelope/provenance and decodes it first. This
module performs no file I/O, environment construction, fitting or optimizer
steps. Native owner types are retained for CohortPrefixSession's exact-type
checks; their saved training modes describe provenance, not update permission.
"""

import copy
from types import MethodType

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import InputSchema
from src.rl.candidate_imitation import ImitationSettings
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.candidate_rollout import CandidateDecision
from src.rl.cohort_ppo import CohortPPOKernel
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import choose_candidate
from src.rl.training_state import training_contract_sha256
from src.rl.validated_returns import ReplaySemantics


def _evaluation_only(*args, **kwargs):
    raise ValueError("evaluation-only owner forbids sampling, fitting and state changes")


class _SavedOptimizer:
    """Keep validated moments in the native saved format, without parameters."""

    def __init__(self, state):
        self._state = copy.deepcopy(state)

    def state_dict(self):
        return copy.deepcopy(self._state)

    step = load_state_dict = _evaluation_only


def _freeze(self):
    self.policy.eval().requires_grad_(False)
    self.policy.zero_grad(set_to_none=True)


def _same_state_only(self, state):
    # Greedy prefix clone/replay may reload the identical owner, never rewind it.
    if (self._failure is not None or state_digest(state) != self._evaluation_sha256
            or state_digest(self.state_dict()) != self._evaluation_sha256):
        raise ValueError("evaluation-only owner accepts only its unchanged saved state")


def _evaluate(self, observation, candidates):
    if self._failure is not None:
        raise ValueError("failed evaluation owner is terminal")
    return evaluate_dynamic_policy(self.policy, observation, candidates, self.contract)


def _greedy(self, observation, candidates):
    evaluation = self.evaluate(observation, candidates)
    # First maximum matches the collector's numpy.argmax, including ties.
    index = max(range(len(evaluation.log_probs)), key=evaluation.log_probs.__getitem__)
    return CandidateDecision(evaluation, choose_candidate(candidates, index))


def _restore_policy(manifest, state):
    contract = ReplayInputContract(InputSchema(**manifest["contract"]["inputs"]),
                                   ReplaySemantics(**manifest["contract"]["replay"]))
    spec = manifest["policy"]
    dtype = {"torch.float32": torch.float32, "torch.float64": torch.float64}.get(spec["dtype"])
    if dtype is None or spec["device"] != "cpu":
        raise ValueError("saved evaluation policy requires CPU float32/float64")
    init = spec["initialization"]
    policy = DynamicCandidatePolicy(contract.inputs, enabled=True,
        architecture=spec["architecture"], message_mode=spec["message_mode"],
        encoder_width=spec["encoder_width"], head_width=spec["head_width"],
        actor_seed=init["actor_seed"], critic_seed=init["critic_seed"],
        initial_reference_bias=init["initial_reference_bias"]).to(dtype=dtype)
    if policy.manifest() != spec:
        raise ValueError("saved policy definition/schema/runtime differs")
    expected = policy.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError("saved policy tensor names differ")
    for name, value in state.items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != expected[name].dtype or value.shape != expected[name].shape
                or not torch.isfinite(value).all().item()):
            raise ValueError("saved policy tensor contract differs")
    policy.load_state_dict(state, strict=True)
    return policy, contract


def _bind_initial_lineage(owner, manifest, *, imitation):
    field = "initial_weights" if imitation else "initial_policy_sha256"
    original = manifest[field]
    if (not isinstance(original, str) or len(original) != 64
            or any(c not in "0123456789abcdef" for c in original)):
        raise ValueError("saved initial policy SHA256 required")
    expected = copy.deepcopy(owner.manifest if imitation else owner._manifest)
    # Initial fork tensors are not stored separately. Keep their authenticated
    # lineage hash, and validate every reconstructible manifest field normally.
    expected[field] = original
    if expected != manifest:
        raise ValueError("saved owner manifest differs from its reconstructed definition")
    if imitation:
        owner.manifest = expected
    else:
        owner._manifest = expected
        owner.manifest_sha256 = training_contract_sha256(expected)


def restore_evaluation_owner(state):
    """Restore a decoded native model state, returning an exact native owner type.

    Accepted formats: frozen dynamic PPO (own_frozen), cohort PPO with window
    or cohort objective, and training-split dynamic imitation (bc_continue).
    Pass the envelope's decoded *state*, not its path, envelope or session.
    CPU dtype, policy/contract, private RNGs, moments and histories roundtrip
    exactly. Existing load_state_dict validators run without calling step/fit.
    Temporary Adam objects used by those validators are then replaced with
    inert moment holders; no executable optimizer is returned.

    Use with CohortPrefixSession(split="test", selection="greedy"), or call
    owner.greedy(observation, candidates) for the same sealed decision receipt.
    owner.evaluate exposes the usual PolicyEvaluation. Sampling, admission,
    updates and changed-state reloads are disabled. Identical-state reloads and
    deepcopy remain available for deterministic prefix replay/clone checks.

    Envelope authenticity and scientific admission remain caller-owned. The
    saved initial fork hash cannot authenticate absent initial tensors; no
    original prototype or external artifact is loaded to reconstruct them.
    """
    require_torch()
    if (not isinstance(state, dict) or not isinstance(state.get("manifest"), dict)
            or "policy" not in state or "failure" not in state):
        raise ValueError("decoded saved model state required")
    if state["failure"] is not None:
        raise ValueError("failed model evidence cannot become an evaluation owner")
    state = copy.deepcopy(state)
    before = state_digest(state)
    manifest = state["manifest"]
    kind = manifest.get("format")
    imitation = kind == "dynamic-candidate-imitation-v1"
    if kind not in ("dynamic-candidate-ppo-kernel-v1", "cohort-ppo-kernel-v1",
                    "dynamic-candidate-imitation-v1"):
        raise ValueError("unsupported cohort evaluation owner format")
    policy, contract = _restore_policy(manifest, state["policy"])
    common = dict(enabled=True, sampling_seed=manifest["sampling_seed"],
                  shuffle_seed=manifest["shuffle_seed"])
    if imitation:
        settings = ImitationSettings(**manifest["settings"])
        if settings.allowed_split != "training" or common["sampling_seed"] is None:
            raise ValueError("continuation BC owner required, not an initializer")
        owner = DynamicCandidateImitationKernel(policy, contract, settings, **common)
    else:
        settings = CandidatePPOSettings(**manifest["settings"])
        if kind == "dynamic-candidate-ppo-kernel-v1":
            if manifest["mode"] != "frozen":
                raise ValueError("only the frozen dynamic PPO control is supported")
            owner = DynamicCandidatePPOKernel(policy, contract, settings, mode="frozen", **common)
        else:
            if manifest["mode"] != "online":
                raise ValueError("saved cohort PPO training owner required")
            owner = CohortPPOKernel(policy, contract, settings, mode="online", **common,
                **{k: manifest[k] for k in ("objective", "accounting_steps", "training_manifest",
                    "training_source_id", "episode_horizon", "episodes_per_rollout")})
    _bind_initial_lineage(owner, manifest, imitation=imitation)
    owner.load_state_dict(state)
    if state_digest(owner.state_dict()) != before:
        raise ValueError("native owner restoration changed the saved state")
    if imitation:
        owner.optimizer = _SavedOptimizer(owner.optimizer.state_dict())
    else:
        for name, optimizer in owner.optimizers.items():
            if optimizer is not None:
                setattr(owner, name + "_optimizer", _SavedOptimizer(optimizer.state_dict()))
    # Instance bindings preserve the historical collector's exact-type checks
    # without changing any legacy class or its training instances.
    for name in ("decide", "add_segment", "fit", "_fit", "update", "_update_in_place",
                 "load", "save"):
        setattr(owner, name, MethodType(_evaluation_only, owner))
    for name, method in (("_set_modes", _freeze), ("load_state_dict", _same_state_only),
                         ("_restore", _same_state_only), ("evaluate", _evaluate), ("greedy", _greedy)):
        setattr(owner, name, MethodType(method, owner))
    owner.evaluation_only, owner._evaluation_sha256 = True, before
    owner._set_modes()
    if state_digest(owner.state_dict()) != before:
        raise ValueError("evaluation binding changed the saved state")
    return owner
