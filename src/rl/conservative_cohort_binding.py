"""Only three existing qualified initializers; no old trained-policy/test loads."""

import copy
import json
from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.conservative_cohort_collection import ConservativeCohortBackend
from src.rl.conservative_cohort_plan import budget_plan, streams, validate_config
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from src.rl.paired_cohort_binding import frozen_owner
from src.rl.paired_cohort_resources import runtime_streams
from src.rl.prospective_ddpg_kernel import state_digest


DIRECTORY = "specs/2026-10-02-conservative-cohort-improvement"
PROPOSAL = "experiments/configs/conservative_cohort_improvement_20261002.json"
RUN = "results/conservative_cohort_improvement_20261002"


def prepare(workspace):
    """Metadata only; inherit the pinned approved initializer contract, not outcomes."""
    from src.rl.paired_cohort_execution import prepare as old_metadata
    workspace = Path(workspace).resolve()
    old = old_metadata(workspace)
    cfg = json.loads((workspace / PROPOSAL).read_text())
    validate_config(cfg)
    inherited = copy.deepcopy(old["inherited"])
    inherited["model_inputs"] = {k: v for k, v in inherited["model_inputs"].items() if k.endswith("/initializer")}
    inherited["backend_config"]["totals"] = dict(backend_layout_builds=3, fresh_episode_builds=195)
    kept = {r["path"] for r in inherited["model_inputs"].values()}
    reference = inherited["backend_config"]["reference"]
    for b in cfg["blocks"]:
        for kind in ("config", "policy"):
            kept.add(reference["directory"] + "/" + reference[kind + "_template"].format(block=b))
    kept.update(old["historical_jsons"])
    return dict(format="conservative-cohort-preparation-v1", scientific_execution_authorized=False,
        source_frozen=False, ready_to_launch=False, workspace=str(workspace),
        branch=old["branch"], result_root=RUN, config=cfg, inherited=inherited,
        **{k: copy.deepcopy(old[k]) for k in ("initializer_config", "initializer_streams",
            "qualified_blocks", "inherited_scientific_config", "historical_jsons")},
        input_files={k: v for k, v in old["input_files"].items() if k in kept},
        streams=streams(cfg, inherited["layout_seeds"]), budget_plan=budget_plan(cfg),
        new_checkpoint_loads=0, new_model_forwards=0, new_environment_calls=0, new_optimizer_calls=0)


def bind_inputs(workspace, root, packet, budget, admission, *, backend_type=ConservativeCohortBackend,
                loader=load_envelope, restore=restore_saved_policy, wrap=frozen_owner, record=file_record):
    cfg, inherited = packet["config"], packet["inherited"]
    runtime = runtime_streams(packet["streams"])
    backend = backend_type(workspace, inherited["backend_config"], runtime, admit_real_calls=admission)
    initializers, models = {}, {}
    for block in cfg["blocks"]:
        budget.check()
        receipt = backend.prepare(block, runtime["environment"][str(block)]["layout"][0])
        write_json_once(Path(root) / f"payload/binding/block{block}.json", receipt)
        expected = inherited["model_inputs"][f"block{block}/initializer"]
        admission("input_hash_verification")
        if record(Path(workspace), expected["path"]) != expected:
            raise ValueError("qualified initializer bytes changed")
        admission("checkpoint_load")
        view = restore(loader(Path(workspace) / expected["path"]), packet["initializer_config"],
                       packet["initializer_streams"], block)
        q = packet["qualified_blocks"][str(block)]
        if q["passed"] is not True or q["kernel_sha256"] != state_digest(view.state_dict()):
            raise ValueError("qualification does not bind this initializer")
        if view.contract != backend.producer(block).contract:
            raise ValueError("saved initializer public layout differs")
        initializers[block] = view
        for role in ("own_frozen", "paired_cost", "bc_continue"):
            seed_role = "paired_cost" if role == "own_frozen" else role
            owner = wrap(view, runtime["neural"][f"block{block}/{seed_role}"])
            if owner.policy.snapshot_sha256() != view.policy.snapshot_sha256():
                raise ValueError("same-start fork changed policy")
            models[f"block{block}/{role}"] = owner
    write_json_once(Path(root) / "payload/binding/models.json", dict(inputs=inherited["model_inputs"],
        models={k: state_digest(v.state_dict()) for k, v in models.items()}, initializer_refitted=False,
        initializer_requalified=False, historical_model_loads=3))
    return backend, initializers, models, runtime
