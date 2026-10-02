"""Pure prospective metadata and translations to the existing durable budget.

No model loading, scientific inference, environment construction or execution
authorization is performed by this module. Freshness is checked at freeze.
"""

import copy
import hashlib

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import validate_budget_plan
from src.rl.paired_cohort_plan import context_keys, maximum_accounting
from src.rl.time_baseline_verification import _decimal_seed


def budget_plan(config):
    counts = maximum_accounting(config)
    maximum = config["maximum"]
    names = {"contexts": "context_states", "actor_calls": "optimizer_calls", "critic_calls": "critic_calls"}
    for key in ("contexts", "branch_instances", "branch_calls", "context_calls", "preflight_calls",
                "evaluation_cohorts", "evaluation_calls", "environment_calls", "actor_calls",
                "critic_calls", "clone_instances", "fresh_environment_builds"):
        if maximum[names.get(key, key)] != counts[key]:
            raise ValueError("declared numerical package disagrees with complete branch accounting")
    phases = {}
    for p in config["phase_budgets"]:
        if p["id"] in phases:
            raise ValueError("duplicate phase")
        phases[p["id"]] = dict(trajectory=0, clone=0, actor=0, critic=0, seconds=p["seconds"])
    expected = {"binding", "same_start_preflight", "reference_contexts", "paired_branches",
                "paired_actor", "bc_actor", "seal", "evaluation", "raw_verification", "archive", "closure"}
    if set(phases) != expected:
        raise ValueError("complete serial phase inventory required")
    blocks, end = len(config["blocks"]), config["economic_endpoint"]
    phases["same_start_preflight"].update(trajectory=blocks * end, clone=blocks * end)
    phases["reference_contexts"]["trajectory"] = counts["context_calls"]
    phases["paired_branches"]["clone"] = counts["branch_calls"]
    for name in ("paired_actor", "bc_actor"):
        phases[name]["actor"] = blocks * config["actor_updates_per_arm_block"]
    phases["evaluation"]["trajectory"] = counts["evaluation_calls"]
    owners = {"same_start_preflight": "preflight_block", "reference_contexts": "context_block",
              "paired_branches": "branch_block", "paired_actor": "paired_fit_block",
              "bc_actor": "bc_fit_block", "evaluation": "evaluation_controller_block"}
    sections = {}
    for name, row in phases.items():
        if name not in owners:
            sections[name] = dict(row, phase=name)
            continue
        roles = config["evaluation_controllers"] if name == "evaluation" else [None]
        divisor = blocks * len(roles)
        if any(row[k] % divisor for k in ("trajectory", "clone", "actor", "critic")):
            raise ValueError("indivisible owner resource allocation")
        for block in config["blocks"]:
            for role in roles:
                key = f"{name}/block{block}" + ("/" + role if role is not None else "")
                sections[key] = {k: row[k] // divisor for k in ("trajectory", "clone", "actor", "critic")}
                sections[key].update(phase=name, seconds=config["per_owner_seconds"][owners[name]])
    limits = {k: sum(p[k] for p in phases.values()) for k in ("trajectory", "clone", "actor", "critic")}
    limits["seconds"] = config["global_seconds"]
    result = dict(format="dynamic-candidate-budget-plan-v1", draft_sha256=digest(config),
                  limits=limits, phases=phases, sections=sections)
    validate_budget_plan(result)
    return result


def stream_manifest(config, layout_seeds):
    """Allocate decimal strings; never pass through JSON/NumPy floating point."""
    namespace = config["rng_namespace"]
    if not isinstance(namespace, str) or not namespace or set(layout_seeds) != set(map(str, config["blocks"])):
        raise ValueError("explicit namespace and unchanged per-block layout seeds required")
    environment, future, neural = {}, {}, {}
    allocations = {}
    def allocate(path, bits=128):
        value = int.from_bytes(hashlib.sha256((namespace + "/" + path).encode()).digest(), "big") % 2**bits
        text = str(value)
        allocations[path] = text
        return text
    for block in config["blocks"]:
        label = f"block{block}"
        layout = str(layout_seeds[str(block)]) if type(layout_seeds[str(block)]) is int else layout_seeds[str(block)]
        _decimal_seed(layout)
        environment[str(block)] = dict(layout=[layout], preflight=[allocate(label + "/preflight/0")],
            context=[allocate(f"{label}/context/{i}") for i in range(config["context_cohorts_per_block"])],
            test=[allocate(f"{label}/test/{i}") for i in range(config["test_worlds_per_block"])])
        for arm in config["training_arms"]:
            neural[f"{label}/{arm}"] = allocate(f"neural/{label}/{arm}", 63)
    for block, cohort, t in context_keys(config):
        name = f"block{block}/cohort{cohort}/after{t}"
        future[name] = [allocate(f"conditional/{name}/rep{i}") for i in range(config["future_replications"])]
    analysis = allocate("analysis/bootstrap", 63)
    values = list(allocations.values())
    if len(values) != len(set(values)) or set(values).intersection(str(v) for v in layout_seeds.values()):
        raise ValueError("new streams collide or overlap immutable layout streams")
    actual = dict(preflight=len(config["blocks"]), context=len(config["blocks"]) * config["context_cohorts_per_block"],
                  conditional_future=len(future) * config["future_replications"],
                  test=len(config["blocks"]) * config["test_worlds_per_block"], neural=len(neural), bootstrap=1)
    if actual != config["streams"]:
        raise ValueError("stream counts disagree with numerical package")
    return dict(format="paired-cohort-streams-v1", namespace=namespace, environment=environment,
        conditional_future=future, neural=neural, bootstrap=analysis, allocations=allocations,
        layout_seeds_reused=True, consumed=False, seed_freshness_verified=False,
        scientific_execution_authorized=False)


def runtime_streams(streams):
    """Translate only declared seeds; keep the persisted manifest unchanged."""
    result = copy.deepcopy(streams)
    for splits in result["environment"].values():
        for split, values in splits.items():
            splits[split] = [_decimal_seed(v) for v in values]
        splits["training"] = splits["context"]
    result["conditional_future"] = {k: [_decimal_seed(v) for v in values]
                                    for k, values in result["conditional_future"].items()}
    result["neural"] = {k: _decimal_seed(v) for k, v in result["neural"].items()}
    result["bootstrap"] = _decimal_seed(result["bootstrap"])
    return result


def inherited_metadata(config, initializer_packet, evaluation_packet):
    """Select actual pinned metadata without opening any scientific checkpoint."""
    blocks = config["blocks"]
    old = initializer_packet["scientific_config"]
    if (old != evaluation_packet["scientific_config"] or old["blocks"] != blocks
            or old["candidate_message_graph"] != "specimen_routes"
            or old["cohort_proposal"]["enrollment_steps"] != config["enrollment_steps"]
            or old["cohort_proposal"]["economic_endpoint"] != config["economic_endpoint"]):
        raise ValueError("inherited objective, environment or input contract differs")
    inputs = {}
    for block in blocks:
        inputs[f"block{block}/initializer"] = copy.deepcopy(initializer_packet["initializers"][str(block)])
        inputs[f"block{block}/saved_cohort_ppo"] = copy.deepcopy(evaluation_packet["models"][f"block{block}/graph/cohort_ppo"])
    for record in inputs.values():
        if (set(record) != {"path", "sha256", "bytes"} or not record["path"].startswith("results/")
                or type(record["bytes"]) is not int or record["bytes"] <= 0
                or len(record["sha256"]) != 64):
            raise ValueError("complete immutable scientific input record required")
    base = {k: copy.deepcopy(old[k]) for k in ("blocks", "reference", "candidate_message_graph",
            "model_proposal", "objective", "candidate_support")}
    base["cohort_proposal"] = {k: old["cohort_proposal"][k] for k in
                                ("enrollment_steps", "patient_resolution_steps", "accounting_steps")}
    base["totals"] = dict(backend_layout_builds=len(blocks),
                          fresh_episode_builds=config["maximum"]["fresh_environment_builds"] - len(blocks))
    return dict(backend_config=base, model_inputs=inputs,
                layout_seeds={str(b): initializer_packet["streams"]["environment"][str(b)]["layout"][0] for b in blocks},
                old_packets=[initializer_packet["packet_sha256"], evaluation_packet["packet_sha256"]],
                new_checkpoint_loads=0, scientific_execution_authorized=False)
