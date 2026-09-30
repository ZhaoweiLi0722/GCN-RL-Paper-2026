"""Bounded frozen-policy counterfactual collection, with no learner interface."""

from __future__ import annotations

import copy
from dataclasses import asdict
import gzip
import json
from pathlib import Path
import time

import numpy as np

from src.rl.action_projection import project_action
from src.rl.experiment import COST_COMPONENT_METRICS
from src.rl.patient_replay_collector import evidence_digest, json_value
from src.rl.residual_options import make_explicit_residual_option_specs, residual_option_actions_from_env


LABELS = ("frozen", "mdl2_first", "specimen_minus_005", "specimen_plus_005",
          "specimen_minus_010", "specimen_plus_010")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "xt", encoding="utf-8") as handle:
        json.dump(value, handle, default=json_value, sort_keys=True, allow_nan=False)
        handle.write("\n")


def append_row(handle, value):
    handle.write(json.dumps(value, default=json_value, sort_keys=True, allow_nan=False) + "\n")
    handle.flush()


class Budget:
    def __init__(self, steps, seconds):
        self.limit = int(steps)
        self.deadline = time.monotonic() + float(seconds)
        self.steps = 0

    def check(self):
        if time.monotonic() >= self.deadline:
            raise TimeoutError("Fixed wall-time budget exhausted; do not retry")

    def consume(self):
        self.check()
        if self.steps >= self.limit:
            raise RuntimeError("Fixed simulator-step budget exhausted")
        self.steps += 1


def streams(spec):
    base = int(spec["rng_namespace"])
    result = {}
    for actor_index, seed in enumerate(spec["seeds"]):
        result[f"trajectory/{seed}"] = base + actor_index
        for step_index, step in enumerate(spec["decision_steps"]):
            for block_index, block in enumerate(("discovery", "validation")):
                for draw in range(spec["draws_per_block"]):
                    offset = 3 + (actor_index * 4 + step_index) * 16 + block_index * 8 + draw
                    result[f"{seed}/{step}/{block}/{draw}"] = base + offset
    if len(set(result.values())) != 195:
        raise ValueError("R3 requires 195 distinct fixed RNG starts")
    return result


def assert_scenario(env, config, scenario):
    if (config["env"].get("scenario_name") != scenario or env.scenario_name != scenario
            or env.config.episode_horizon != 52 or not env.config.include_time_state
            or env.config.enable_overtime_control or env.action_size != 80
            or env.observation_size != 561):
        raise ValueError("Nominal-history runtime contract differs")
    # Retain the complete effective dataclass, not just the scenario label.
    return asdict(env.env_config)


def candidates(policy, env, options):
    observation = env.observation()
    frozen = policy.act(observation, env=env)
    agent = policy._agent
    anchor = agent._base_action_from_state_np(observation)
    actions = [frozen] + residual_option_actions_from_env(
        anchor, env, make_explicit_residual_option_specs(options))
    if len(actions) != len(LABELS):
        raise ValueError("Expected frozen, anchor, and four fixed corrections")
    rows = []
    for index, action in enumerate(actions):
        action = project_action(action, env_state=env, action_space_info=env.action_size).action
        scale = agent.residual_scale_vector
        witness = np.divide(action - anchor, scale, out=np.zeros_like(action), where=scale != 0)
        witness_in_box = bool(np.max(np.abs(witness)) <= 1.0 + 1e-6)
        reconstructed = agent._compose_action_np(observation, np.clip(witness, -1, 1))
        error = float(np.max(np.abs(action - reconstructed)))
        certified = index == 0 or (witness_in_box and error <= 1e-6)
        rows.append({"index": index, "label": LABELS[index], "request": action,
                     "deployment_map_witness": witness,
                     "witness_in_box": witness_in_box, "witness_error": error,
                     "map_reachable_certified": certified,
                     "diagnostic_only": not certified,
                     "reachability_scope": "action-map witness only; not network learnability"})
    return rows


def behavioral_digest(snapshot, info):
    physical = copy.deepcopy(snapshot)
    # This diagnostic counter is neither observed nor used by simulator dynamics.
    physical["scalars"].pop("cumulative_blocked_specimen_requests", None)
    return evidence_digest({"state": physical, "cost": info["cost"],
                            "routes": info["specimen_route_events"]})


def recorded_step(env, action, budget, *, horizon=52):
    before_t = int(env.t)
    if before_t >= horizon:
        raise ValueError("Cannot step past objective endpoint")
    before_obs = env.observation()
    budget.consume()
    observation, reward, done, info = env.step(action)
    if env.t != before_t + 1:
        raise ValueError("Non-chronological simulator step")
    components = {key: float(info[key]) for key in COST_COMPONENT_METRICS if key != "base_cost"}
    total = float(sum(components.values()))
    if not np.isclose(total, info["cost"], rtol=1e-12, atol=1e-6) or reward != -info["cost"]:
        raise ValueError("Absolute reward/cost-component contract failed")
    return {"t": before_t, "next_t": int(env.t), "observation_sha256": evidence_digest(before_obs),
            "next_observation_sha256": evidence_digest(observation), "request": action,
            "reward": float(reward), "scaled_reward": float(reward) * 1e-9,
            "cost_components": components, "info": info, "native_done": bool(done),
            "objective_terminal": bool(done or env.t == horizon), "truncated": False}


def obligations(env):
    state = env.state_dict()
    return {"t": int(env.t), "waiting": [len(q) for q in state["patient_queues"]],
            "production": [[len(q) for q in stages] for stages in state["in_production_patients"]],
            "specimen_transits": len(state["specimen_transits"]),
            "product_return_transits": len(state["product_return_transits"]),
            "finished_product_buckets": state["finished_product_buckets"],
            "reagents": state["arrays"]["reagents"], "bioreactors": state["arrays"]["bioreactors"],
            "pipelines": {key: value for key, value in state["arrays"].items() if "pipeline" in key},
            "identity": env.assert_identity_conservation()}


def result_from_steps(rows, env):
    if not rows or not rows[-1]["objective_terminal"]:
        raise ValueError("Interrupted continuation cannot be a completed outcome")
    info = rows[-1]["info"]
    return {"step_count": len(rows), "total_cost": float(sum(r["info"]["cost"] for r in rows)),
            "patients_lost": float(sum(np.sum(r["info"]["patients_lost"]) for r in rows)),
            "patients_completed": float(sum(np.sum(r["info"]["patients_completed"]) for r in rows)),
            "completion_service_level": float(info["completion_service_level"]),
            "manufacturing_loss_rate": float(info["patient_ineligibility_during_manufacturing_rate"]),
            "route_count": float(sum(r["info"]["specimen_route_count"] for r in rows)),
            "objective_terminal": True, "truncated": False, "obligations": obligations(env)}


def collect_draw(snapshot_env, policy, choices, rng_seed, budget, root, stem):
    """Share tails only after exact physical/RNG-state and event equivalence."""
    representatives = {}
    results = []
    for choice in choices:
        env = copy.deepcopy(snapshot_env)
        env.rng = np.random.default_rng(rng_seed)
        before = env.state_dict()
        first = recorded_step(env, np.asarray(choice["request"]), budget)
        first["post_state"] = env.state_dict()
        identity = behavioral_digest(first["post_state"], first["info"])
        relative = f"traces/{stem}/action{choice['index']}.jsonl.gz"
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        prior = representatives.get(identity)
        with gzip.open(path, "xt", encoding="utf-8") as handle:
            append_row(handle, first)
            if prior is not None:
                outcome = copy.deepcopy(prior["outcome"])
                tail_from = prior["trace"]
            else:
                rows = [first]
                while not rows[-1]["objective_terminal"]:
                    action = policy.act(env.observation(), env=env)
                    row = recorded_step(env, action, budget)
                    append_row(handle, row)
                    rows.append(row)
                outcome = result_from_steps(rows, env)
                tail_from = None
        result = {"action_index": choice["index"], "label": choice["label"],
                  "rng_seed": rng_seed, "rng_start_sha256": evidence_digest(before["rng_state"]),
                  "first_full_state_sha256": evidence_digest(before),
                  "execution_identity": identity, "trace": relative, "tail_from": tail_from,
                  "representative_action": prior["action_index"] if prior else choice["index"],
                  "map_reachable_certified": choice["map_reachable_certified"],
                  "outcome": outcome}
        results.append(result)
        if prior is None:
            representatives[identity] = result
    return results


def discovery_choice(rows):
    means = [float(np.mean([r["outcome"]["total_cost"] for r in rows if r["action_index"] == i]))
             for i in range(6)]
    if not np.isfinite(means).all():
        raise ValueError("Invalid discovery costs")
    return {"selected_action": min(range(6), key=lambda i: (means[i], i)), "mean_costs": means,
            "rule": "lowest discovery mean, first-index tie; no clinical filtering"}
