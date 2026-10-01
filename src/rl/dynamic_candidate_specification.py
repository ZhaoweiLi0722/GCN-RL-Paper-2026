"""Non-executable bindings for the single-GCN mechanism pilot.

Reading a proposal or a static layout never constructs a patient environment,
loads a policy checkpoint, or grants numerical execution permission.
"""

import copy
import json
from pathlib import Path

from src.rl.candidate_pilot_compatibility import inspect_patient_layout, require_supported_layouts
from src.rl.candidate_pilot_driver import file_record
from src.rl.dynamic_candidate_resources import dynamic_budget_plan, dynamic_stream_manifest


DRAFT = "specs/2026-10-01-adaptive-paper-delivery/pilot-budget-draft.json"
REFERENCE_BINDINGS = "experiments/configs/candidate_return_pilot_20260930.json"


def build_dynamic_proposal(workspace):
    """Resolve previously declared inputs and make two prospective choices explicit."""
    root = Path(workspace).resolve()
    draft_record = file_record(root, DRAFT)
    inherited_record = file_record(root, REFERENCE_BINDINGS)
    draft = json.loads((root / DRAFT).read_text())
    inherited = json.loads((root / REFERENCE_BINDINGS).read_text())
    if draft["scientific_execution_authorized"] is not False or draft["ready_to_launch"] is not False:
        raise ValueError("original draft must remain non-executable")
    config = copy.deepcopy(draft)
    config["schema"] = "dynamic-candidate-prospective-binding-v1"
    config["status"] = "PROPOSED_BINDING_NOT_EXECUTABLE_NOT_AUTHORIZED"
    config["source_bindings"] = {"original_draft": draft_record, "reference_input_declarations": inherited_record}
    config["candidate_message_graph"] = "specimen_routes"
    config["objective"].update(raw_state_width=inherited["objective"]["raw_state_width"],
                               transfer_scale=inherited["objective"]["transfer_scale"])
    config["candidate_support"] = copy.deepcopy(inherited["candidate_support"])
    config["reference"]["locks"] = copy.deepcopy(inherited["reference"]["locks"])
    config["model_proposal"].update(graph="specimen_routes", initial_reference_bias=0.0,
        exact_initialization_and_preprocessing="dynamic_candidate_policy_v1_linear_defaults_then_zero_critic_final",
        new_model_restore_acceptance="zero_update_contract_tests_not_scientific_success")
    config["prospective_choices_requiring_packet_approval"] = {
        "graph": {"original_draft": draft["model_proposal"]["graph"], "proposed": "specimen_routes",
                  "reason": "existing public producer preserves heterogeneous environment relations without collapsing them"},
        "initial_reference_bias": {"original_draft": "unresolved", "proposed": 0.0,
                                   "trainable": True, "selection_based_on_fitted_results": False},
        "failed_artificial_prerequisite": "remains_failed_1_of_9_prospective_replacement_requires_approval",
    }
    config["interpretation"] = {
        "question": "simulation_interaction_learning_beyond_own_frozen_and_continued_imitation",
        "scope": "restricted_specimen_coordination_mechanism_not_final_method_ceiling",
        "all_environment_channels_retained": True, "non_specimen_candidate_components_shared": True,
        "isolated_graph_effect_established": False, "full_joint_control_claim": False,
        "deployment_adaptation_claim": False, "clinical_noninferiority_claim": False,
        "historical_r4_graph_representation_identical": False,
    }
    validate_dynamic_proposal(config)
    return config


def validate_dynamic_proposal(config):
    if (config.get("scientific_execution_authorized") is not False
            or config.get("ready_to_launch") is not False
            or config.get("candidate_message_graph") != "specimen_routes"
            or config["model_proposal"]["graph"] != "specimen_routes"):
        raise ValueError("explicit unapproved specimen-routes binding required")
    model, opt, obj = config["model_proposal"], config["optimizer"], config["objective"]
    if (model["device"] != "cpu" or model["dtype"] != "float32"
            or model["initial_reference_bias"] != 0.0 or model["encoder_width"] != 16 or model["head_width"] != 32):
        raise ValueError("one fixed prospective initialization required, no model search")
    if any(opt[k] != value for k, value in {
            "kind": "Adam", "betas": [.9, .999], "eps": 1e-8, "weight_decay": 0, "foreach": False}.items()):
        raise ValueError("unsupported optimizer declaration cannot be silently ignored")
    if (config["blocks"] != [60, 61, 62] or config["representations"] != ["graph"]
            or obj["gamma"] != 1.0 or obj["gae_lambda"] != 1.0
            or obj["terminal_bootstrap"] != 0.0 or obj["terminal_cost_added"] != 0.0):
        raise ValueError("fixed blocks and finite-horizon objective required")
    if (obj["action_width"] != 4 * obj["num_facilities"]
            or config["candidate_support"]["max_original_requests"] != 6
            or config["candidate_support"]["options"] != [
                {"group": "specimen_transfer", "epsilon": epsilon, "sign": sign}
                for epsilon, sign in model["specimen_epsilon_sign_pairs"]]):
        raise ValueError("candidate support differs from the declared proposal")
    dynamic_budget_plan(config)
    dynamic_stream_manifest(config)


def inspect_dynamic_inputs(workspace, config):
    """Hash only consumed reference files and inspect JSON topology, not tensors."""
    root, reports, inputs, environments = Path(workspace).resolve(), [], {}, []
    reference = config["reference"]
    for block in config["blocks"]:
        paths = {key: str(Path(reference["directory"]) / reference[key + "_template"].format(block=block))
                 for key in ("config", "policy")}
        for key, path in paths.items():
            record = file_record(root, path)
            if record["sha256"] != reference["locks"][str(block)][key]:
                raise ValueError("locked R4 input hash mismatch")
            inputs[path] = record
        runtime = json.loads((root / paths["config"]).read_text())
        layout = inspect_patient_layout(runtime, config["objective"],
                                        message_graph=config["candidate_message_graph"])
        require_supported_layouts(layout)
        environments.append(runtime["env"])
        reports.append({"block": block, "layout": layout})
    if any(env != environments[0] for env in environments[1:]):
        raise ValueError("paired block environments differ")
    return {"format": "dynamic-candidate-static-inputs-v1", "passed": True,
            "inputs": inputs, "blocks": reports, "effective_environments_equal": True,
            "patient_environment_builds": 0, "environment_steps": 0, "checkpoint_loads": 0,
            "live_layout_verified": False}


def static_model_counts(num_facilities, node_width, global_width, encoder_width, head_width):
    """Algebra for this fixed architecture; zero model construction or seed use."""
    n, f, g, e, h = num_facilities, node_width, global_width, encoder_width, head_width
    if any(type(v) is not int or v < 1 for v in (n, f, e, h)) or type(g) is not int or g < 0:
        raise ValueError("explicit positive architecture dimensions required")
    action = 4 * n
    base = n * e + g + n * n + 2 * action
    projection = (f + 1) * e
    actor = projection + (base + action + 2 + 1) * h + h + 1 + 1
    critic = projection + (base + 1) * h + h + 1
    return {"actor": actor, "critic": critic, "total": actor + critic, "shared": 0}
