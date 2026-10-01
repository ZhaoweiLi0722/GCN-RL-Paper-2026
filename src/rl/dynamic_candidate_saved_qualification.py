"""Read-only qualification of saved public examples; no campaign or optimizer.

The execution layer owns permission, input hashes, deadlines and durable debits.
These pure bindings also support invented zero-update engineering fixtures.
"""

import copy
from dataclasses import asdict

import torch

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import InputSchema
from src.rl.candidate_imitation import ImitationSettings, decode_example
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_factory import qualify_dynamic_path, qualify_dynamic_outcomes
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.validated_returns import ReplaySemantics


class SavedPolicyView:
    """Expose actual live weights plus inert saved metadata, never an optimizer."""
    def __init__(self, policy, contract, state):
        self.policy, self.contract = policy, contract
        self.saved = copy.deepcopy(state)

    def state_dict(self):
        result = copy.deepcopy(self.saved)
        result["policy"] = copy.deepcopy(self.policy.state_dict())
        return result


def restore_saved_policy(state, config, streams, block):
    keys = {"manifest", "policy", "optimizer", "rng", "sampling_rng", "steps", "history", "failure"}
    if (set(state) != keys or block not in config["blocks"] or state["failure"] is not None
            or type(state["steps"]) is not int
            or state["steps"] != config["initialization"]["actor_adam_calls_per_block"]):
        raise ValueError("complete nonfailed saved initializer required")
    manifest = state["manifest"]
    contract = ReplayInputContract(InputSchema(**manifest["contract"]["inputs"]),
                                   ReplaySemantics(**manifest["contract"]["replay"]))
    spec, neural = config["model_proposal"], streams["neural"]
    base = f"block{block}/graph/"
    if config["candidate_message_graph"] != "specimen_routes" or spec["graph"] != "specimen_routes":
        raise ValueError("unchanged specimen-routes graph required")
    # Allocate the original architecture, then overwrite every tensor strictly.
    # Construction restores its local RNG context; no optimizer is constructed.
    policy = DynamicCandidatePolicy(contract.inputs, enabled=True, architecture="graph",
        message_mode="physical", encoder_width=spec["encoder_width"], head_width=spec["head_width"],
        actor_seed=neural[base + "actor_initialization"], critic_seed=neural[base + "critic_initialization"],
        initial_reference_bias=spec["initial_reference_bias"])
    opt, init = config["optimizer"], config["initialization"]
    expected_manifest = {"format": "dynamic-candidate-imitation-v1", "contract": asdict(contract),
        "settings": asdict(ImitationSettings(opt["learning_rate"], opt["gradient_norm_cap_each_owner"],
            init["batch_size"], init["actor_adam_calls_per_block"], 1, "demonstration")),
        "policy": policy.manifest(), "initial_weights": policy.snapshot_sha256(),
        "shuffle_seed": neural[base + "bc_init/shuffle"], "sampling_seed": None, "trainable_owner": "actor"}
    if manifest != expected_manifest or state["sampling_rng"] is not None:
        raise ValueError("saved initializer architecture/contract/seed/settings differ")
    expected = policy.state_dict()
    if set(state["policy"]) != set(expected):
        raise ValueError("saved policy tensor names differ")
    for name, value in state["policy"].items():
        if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                or value.dtype != expected[name].dtype or value.shape != expected[name].shape
                or not torch.isfinite(value).all().item()):
            raise ValueError("saved policy tensor contract differs")
        if name.startswith("critic.") and not torch.equal(value, expected[name]):
            raise ValueError("initializer modified the frozen critic")
    policy.load_state_dict(state["policy"], strict=True)
    policy.eval().requires_grad_(False)
    view = SavedPolicyView(policy, contract, state)
    if state_digest(view.state_dict()) != state_digest(state):
        raise ValueError("read-only policy restoration differs")
    return view


def validate_saved_boundary(boundary, initializers, raw_episodes, config, streams):
    """Join saved public examples to already hash-verified qualification rows.

    raw_episodes is a list of {header, rows, files}; rows retain their original
    JSON event wrappers. Only this saved subset is read, no environment restored.
    """
    blocks = config["blocks"]
    roles = ("r4", "initializer_greedy")
    worlds, horizon = config["qualification"]["fresh_worlds_per_block"], config["objective"]["horizon"]
    keys = {f"block{b}/graph" for b in blocks}
    if (boundary["format"] != "dynamic-campaign-v1" or boundary["config_sha256"] != digest(config)
            or boundary["streams_sha256"] != digest(streams) or boundary["failure"] is not None
            or boundary["qualified"] or boundary["models"] or boundary["model_paths"] or boundary["test_index"]
            or any(boundary[name] is not None for name in ("live_session", "clone", "continuation", "recorder"))
            or boundary["work"]["job"] != "qualification" or boundary["work"]["cursor"] != len(blocks) * len(roles) * worlds
            or set(boundary["initializers"]) != keys or set(initializers) != set(blocks)
            or set(boundary["qualification"]) != set(blocks)):
        raise ValueError("exact completed qualification collection boundary required")
    indexed = {}
    for episode in raw_episodes:
        header = episode["header"]
        key = header["block"], header["role"], header["world_index"]
        if header["split"] != "qualification" or key in indexed:
            raise ValueError("duplicate or nonqualification raw episode")
        indexed[key] = episode
    if set(indexed) != {(b, role, w) for b in blocks for role in roles for w in range(worlds)}:
        raise ValueError("exact saved paired qualification matrix required")
    expected_indexes = [episode["files"] for episode in raw_episodes]
    if (sorted(map(digest, boundary["work"]["indexes"])) != sorted(map(digest, expected_indexes))
            or not set(map(digest, expected_indexes)).issubset(map(digest, boundary["raw_index"]))):
        raise ValueError("boundary raw index does not match saved readout")
    examples = boundary["qualification"]
    identities = set()
    for block in blocks:
        view = initializers[block]
        if (state_digest(boundary["initializers"][f"block{block}/graph"]) != state_digest(view.state_dict())
                or set(examples[block]) != set(roles)):
            raise ValueError("boundary initializer or qualification paths differ")
        for role in roles:
            path = examples[block][role]
            if len(path) != worlds * horizon:
                raise ValueError("exact saved qualification path length required")
            for world in range(worlds):
                episode = indexed[block, role, world]
                header, rows = episode["header"], episode["rows"]
                if (len(rows) != horizon or header["seed"] != streams["environment"][str(block)]["qualification"][world]
                        or digest(header["session_manifest"]["contract"]) != digest(asdict(view.contract))
                        or header["selection"] != ("reference" if role == "r4" else "greedy")
                        or header["policy_sha256"] != view.policy.snapshot_sha256()):
                    raise ValueError("saved qualification contract/seed/policy lineage differs")
                for step, row in enumerate(rows):
                    example = path[world * horizon + step]
                    audit = row["event"]["audit"]
                    record, evaluation = audit["record"], audit["decision"]["evaluation"]
                    identity = f"{header['trajectory_id']}/{step}"
                    if (identity in identities or example["identity"] != identity or example["split"] != "qualification"
                            or record["trajectory_id"] != header["trajectory_id"] or record["source_id"] != header["source_id"]
                            or record["step_index"] != step or record["state_token"] != example["candidates"]["state_token"]
                            or digest(example["actor_state"]) != digest(record["state"])
                            or digest(record["state"]) != digest(evaluation["actor_state"])
                            or digest(example["candidates"]) != digest(evaluation["candidates"])):
                        raise ValueError("saved public example identity/state/support differs from raw record")
                    decode_example(example, view.contract)
                    identities.add(identity)
    return examples


def score_saved_qualification(initializers, examples, outcomes, config, *, before_forward, on_path=None):
    results = {}
    for block in config["blocks"]:
        owner = initializers[block]
        paths = {}
        for role in ("r4", "initializer_greedy"):
            paths[role] = qualify_dynamic_path(owner, examples[block][role], config,
                                               before_forward=before_forward)
            if on_path is not None:
                on_path(block, role, paths[role])
        paired = qualify_dynamic_outcomes([row for row in outcomes if row["block"] == block], config, block)
        results[str(block)] = {"paths": paths, "outcomes": paired,
            "passed": paired["passed"] and all(row["passed"] for row in paths.values()),
            "kernel_sha256": state_digest(owner.state_dict())}
    return {"format": "saved-dynamic-qualification-v1", "blocks": results,
        "passed": all(row["passed"] for row in results.values()), "rl_performance_claim": False,
        "clinical_noninferiority_claim": False, "automatic_training_authorized": False,
        "old_s1_attempt_status": "terminal_failed_unchanged"}
