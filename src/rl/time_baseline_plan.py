"""Pure JSON planning for the one-shot, six-controller baseline comparison.

No files, models, environments or optimizers are opened. These adapters neither
authorize execution nor certify historical seed freshness. Scientific settings
and the input proposal are never modified.
"""

import hashlib

from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_resources import validate_budget_plan


TRAINING_ARMS = ("current_ppo", "time_baseline_ppo", "bc_continue")
CONTROLLERS = ("own_frozen", *TRAINING_ARMS, "r4", "full_mdl2")
PHASES = ("runtime_binding", "same_start_preflight", *TRAINING_ARMS,
          "all_model_seal", "final_evaluation", "raw_verification",
          "payload_archive", "supervisor_closure")
RESOURCES = ("trajectory", "clone", "actor", "critic", "seconds")
ENCODING = "sha256(namespace)[:12] big-endian <<16 plus ordinal; decimal strings on disk"
# The protocol fixes complete 52-step trajectories; this is not a model setting.
HORIZON = 52


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value):
    _require(type(value) is int and value >= 0, "nonnegative integer required")
    return value


def _inventory(value, keys, label):
    _require(type(value) is dict and set(value) == set(keys), f"{label} inventory mismatch")


def _counts(value, expected, label):
    _inventory(value, expected, label)
    for key, count in expected.items():
        _require(_integer(value[key]) == count, f"{label}/{key} arithmetic mismatch")


def _validate_proposal(proposal):
    required = {"schema", "blocks", "training_arms", "evaluation_controllers",
                "episodes_per_training_arm_block", "episodes_per_rollout",
                "rollouts_per_training_arm_block", "epochs_per_rollout", "minibatch_sizes",
                "evaluation_worlds_per_block", "preflight_worlds_per_block",
                "clone_steps_per_preflight_controller", "initial_historical_loads",
                "layout_builds", "episode_builds", "preflight_clone_instances",
                "same_start_learned_forks", "scheduled_restore_envelope_reads",
                "phase_budgets", "totals", "per_owner", "rng"}
    _require(type(proposal) is dict and required <= set(proposal), "proposal fields missing")
    _require(proposal["schema"] == "time-baseline-comparison-proposal-v1", "unknown proposal schema")
    _require(proposal["training_arms"] == list(TRAINING_ARMS), "training controller inventory mismatch")
    _require(proposal["evaluation_controllers"] == list(CONTROLLERS), "evaluation controller inventory mismatch")
    blocks = proposal["blocks"]
    _require(type(blocks) is list and len(blocks) == 3, "three declared blocks required")
    for block in blocks:
        _integer(block)
    _require(len(set(blocks)) == len(blocks), "duplicate block")
    b, c = len(blocks), len(CONTROLLERS)
    dimensions = {"episodes_per_training_arm_block": 32, "episodes_per_rollout": 4,
                  "rollouts_per_training_arm_block": 8, "epochs_per_rollout": 4,
                  "evaluation_worlds_per_block": 12, "preflight_worlds_per_block": 1,
                  "clone_steps_per_preflight_controller": 4}
    for key, expected in dimensions.items():
        _require(_integer(proposal[key]) == expected, f"v1 {key} differs from protocol")
    sizes = proposal["minibatch_sizes"]
    _require(type(sizes) is list and sizes == [64, 64, 64, 16], "v1 minibatch layout mismatch")
    for size in sizes:
        _integer(size)
    episodes = proposal["episodes_per_training_arm_block"]
    rollout = proposal["episodes_per_rollout"]
    rollouts = proposal["rollouts_per_training_arm_block"]
    _require(rollouts * rollout == episodes and sum(sizes) == rollout * HORIZON,
             "complete rollout arithmetic mismatch")
    calls = rollouts * proposal["epochs_per_rollout"] * len(sizes)
    worlds = proposal["evaluation_worlds_per_block"]
    owners = proposal["per_owner"]
    _inventory(owners, ("ppo_each_arm_block", "bc_each_block", "evaluation_each_controller_block"),
               "per-owner")
    expected_owners = {
        "ppo_each_arm_block": {"episodes": episodes, "trajectory": episodes * HORIZON,
                               "actor": calls, "critic": calls},
        "bc_each_block": {"episodes": episodes, "trajectory": episodes * HORIZON,
                          "actor": calls, "critic": 0},
        "evaluation_each_controller_block": {"episodes": worlds, "trajectory": worlds * HORIZON},
    }
    for name, expected in expected_owners.items():
        row = owners[name]
        _inventory(row, (*expected, "seconds"), name)
        seconds = _integer(row["seconds"])
        _require(seconds > 0, "positive per-owner clock required")
        _counts(row, expected | {"seconds": seconds}, name)

    phase_rows = proposal["phase_budgets"]
    _require(type(phase_rows) is list, "phase list required")
    for row in phase_rows:
        _inventory(row, ("id", "episodes", *RESOURCES), "phase")
    _require([row["id"] for row in phase_rows] == list(PHASES), "phase inventory/order mismatch")
    phases = {}
    for row in phase_rows:
        name = row["id"]
        seconds = _integer(row["seconds"])
        _require(seconds > 0, "positive phase clock required")
        expected = dict.fromkeys(("episodes", "trajectory", "clone", "actor", "critic"), 0)
        if name == "same_start_preflight":
            expected.update(episodes=b * c, trajectory=b * c * HORIZON,
                            clone=b * c * proposal["clone_steps_per_preflight_controller"])
        elif name in TRAINING_ARMS or name == "final_evaluation":
            owner = ("evaluation_each_controller_block" if name == "final_evaluation" else
                     "bc_each_block" if name == "bc_continue" else "ppo_each_arm_block")
            multiplicity = b * c if name == "final_evaluation" else b
            expected.update({key: value * multiplicity for key, value in owners[owner].items()
                             if key != "seconds"})
            _require(seconds == multiplicity * owners[owner]["seconds"], "per-owner clock partition mismatch")
        _counts({k: v for k, v in row.items() if k != "id"},
                expected | {"seconds": seconds}, name)
        phases[name] = {key: row[key] for key in RESOURCES}

    totals = {key: sum(row[key] for row in phase_rows)
              for key in ("episodes", "trajectory", "clone", "actor", "critic")}
    totals.update(environment=totals["trajectory"] + totals["clone"],
                  optimizer=totals["actor"] + totals["critic"],
                  environment_builds=totals["episodes"] + b,
                  phase_seconds=sum(row["seconds"] for row in phase_rows))
    _inventory(proposal["totals"], (*totals, "global_seconds"), "totals")
    global_seconds = _integer(proposal["totals"]["global_seconds"])
    _require(global_seconds >= totals["phase_seconds"], "global clock below phase caps")
    _counts(proposal["totals"], totals | {"global_seconds": global_seconds}, "totals")
    for key, expected in {"initial_historical_loads": b * 2, "layout_builds": b,
                          "episode_builds": totals["episodes"], "preflight_clone_instances": b * c,
                          "same_start_learned_forks": b * (c - 2),
                          "scheduled_restore_envelope_reads": b * c * (1 + worlds)}.items():
        _require(_integer(proposal[key]) == expected, f"{key} arithmetic mismatch")

    rng = proposal["rng"]
    _inventory(rng, ("namespace", "encoding", "roles_by_block", "paired_environment_starts",
                     "paired_future_events_after_divergence_claimed", "freshness_verified"), "rng")
    _require(type(rng["namespace"]) is str and bool(rng["namespace"].strip()), "nonempty namespace required")
    _require(rng["encoding"] == ENCODING, "unknown seed encoding")
    _require(rng["paired_environment_starts"] is True
             and rng["paired_future_events_after_divergence_claimed"] is False,
             "only environment starts may be claimed paired")
    _require(type(rng["freshness_verified"]) is bool, "explicit freshness flag required")
    _inventory(rng["roles_by_block"], map(str, blocks), "ordinal block")
    for index, block in enumerate(blocks):
        expected = {"layout": [index, index], "preflight": [b + index, b + index],
                    "training": [2 * b + index * episodes, 2 * b + (index + 1) * episodes - 1],
                    "test": [2 * b + b * episodes + index * worlds,
                             2 * b + b * episodes + (index + 1) * worlds - 1]}
        actual = rng["roles_by_block"][str(block)]
        _inventory(actual, expected, "ordinal role")
        for role, bounds in expected.items():
            span = actual[role]
            _require(type(span) is list and len(span) == 2, "ordinal endpoint pair required")
            for ordinal in span:
                _require(_integer(ordinal) < 2**16, "ordinal outside encoding")
            _require(span == bounds, f"block{block}/{role} ordinal layout mismatch")
    return phases


def time_baseline_budget_plan(proposal):
    """Return a validated DynamicCandidateBudget v1 plan, without claiming it.

    Preflight retains its declared aggregate cap (no invented per-controller
    clock). Training has nine owners and evaluation has eighteen owners. The
    runtime_input_binding section maps to the runtime_binding phase so the
    existing budget charges startup time from its outer ``started`` clock.
    """
    phases = _validate_proposal(proposal)
    sections = {}
    for phase, caps in phases.items():
        if phase in TRAINING_ARMS or phase == "final_evaluation":
            owner = ("evaluation_each_controller_block" if phase == "final_evaluation" else
                     "bc_each_block" if phase == "bc_continue" else "ppo_each_arm_block")
            row = proposal["per_owner"][owner]
            controllers = CONTROLLERS if phase == "final_evaluation" else (None,)
            for block in proposal["blocks"]:
                for controller in controllers:
                    name = f"{phase}/block{block}" + (f"/{controller}" if controller else "")
                    sections[name] = {"phase": phase, "trajectory": row["trajectory"], "clone": 0,
                                      "actor": row.get("actor", 0), "critic": row.get("critic", 0),
                                      "seconds": row["seconds"]}
        else:
            name = "runtime_input_binding" if phase == "runtime_binding" else phase
            sections[name] = caps | {"phase": phase}
    result = {"format": "dynamic-candidate-budget-plan-v1", "draft_sha256": digest(proposal),
              "limits": {key: proposal["totals"][key] for key in RESOURCES if key != "seconds"}
                        | {"seconds": proposal["totals"]["global_seconds"]},
              "phases": phases, "sections": sections}
    validate_budget_plan(result)
    return result


def time_baseline_serial_sections(proposal):
    """Return ordered section names; the runner must follow this explicit list.

    JSON object order is not an execution contract. All three training arms end
    before all_model_seal, which precedes every six-controller evaluation block.
    DynamicCandidateBudget prevents concurrent/repeated sections, not reordering.
    """
    return list(time_baseline_budget_plan(proposal)["sections"])


def time_baseline_stream_manifest(proposal):
    """Allocate decimal-string seeds and arm-to-purpose path bindings only.

    Convert seeds with ``int`` at the runtime boundary. Each training arm gets
    a separate RNG instance initialized from its referenced sampling seed. PPO
    shuffle starts are paired; BC shuffle is purpose-separated. Preflight uses
    different streams and must never advance any training RNG. No new actor or
    critic initialization streams are allocated for the saved initializers.

    Only environment/neural contain seeds. For the existing scoped collision
    audit, pass those two maps, not count metadata or ordinal endpoints. This
    function checks internal uniqueness, not historical/local collision scope.
    """
    _validate_proposal(proposal)
    rng = proposal["rng"]
    namespace = rng["namespace"]
    base = int.from_bytes(hashlib.sha256(namespace.encode()).digest()[:12], "big") << 16
    environment, neural, bindings = {}, {}, {}
    for block in proposal["blocks"]:
        key = str(block)
        environment[key] = {role: [str(base + ordinal) for ordinal in range(start, end + 1)]
                            for role, (start, end) in rng["roles_by_block"][key].items()}
        bindings[key] = {}
        for split, purpose in (("preflight", "preflight"), ("training", "continuation")):
            prefix = f"block{block}/graph/{purpose}/"
            bindings[key][split] = {
                arm: {"sampling_seed": prefix + "sample",
                      "shuffle_seed": prefix + ("bc_continue/shuffle" if arm == "bc_continue" else "ppo/shuffle")}
                for arm in TRAINING_ARMS}
            for suffix in ("sample", "ppo/shuffle", "bc_continue/shuffle"):
                neural[prefix + suffix] = None
    neural["analysis/bootstrap"] = None
    for path in neural:
        neural[path] = str(int.from_bytes(hashlib.sha256((namespace + "/" + path).encode()).digest()[:8],
                                          "big") % 2**63)
    seeds = [seed for roles in environment.values() for values in roles.values() for seed in values]
    seeds += list(neural.values())
    _require(len(seeds) == len(set(seeds)), "cross-purpose seed collision")
    return {"format": "time-baseline-streams-v1", "proposal_sha256": digest(proposal),
            "namespace": namespace, "environment": environment, "neural": neural,
            "neural_bindings": bindings, "ordinals": {
                str(block): {role: list(bounds) for role, bounds in rng["roles_by_block"][str(block)].items()}
                for block in proposal["blocks"]},
            "unique_worlds": len(seeds) - len(neural), "episode_uses": proposal["totals"]["episodes"],
            "uses_per_world": {"layout": 0, "preflight": len(CONTROLLERS),
                               "training": len(TRAINING_ARMS), "test": len(CONTROLLERS)},
            "layout_builds": len(proposal["blocks"]), "paired_environment_starts": True,
            "paired_future_events_after_divergence_claimed": False,
            "seed_freshness_verified": False, "consumed": False, "scientific_execution_authorized": False}
