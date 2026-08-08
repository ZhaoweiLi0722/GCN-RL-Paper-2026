"""Layered, read-only diagnostics for the Recovery 7 Windows native crash.

Every invocation runs exactly one layer in its own Python process.  The caller
is responsible for process isolation and Windows Error Reporting collection;
this module enables ``faulthandler``, validates the frozen training contract,
and never writes outside its explicitly supplied result JSON.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import faulthandler
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback
from typing import Any, Callable

import numpy as np

from evaluation.run_full_benchmark import (
    load_benchmark_plan,
    make_training_config,
    resolve_budget,
    select_scenarios,
)
from evaluation.train_multiscenario_network_residual import (
    deep_update,
    merge_multiscenario_env_overrides,
    multiscenario_env_overrides,
)
from evaluation.train_network_residual_history_screen import (
    make_history_screen_config,
)
from src.baselines.heuristics import (
    facility_net_action_from_state,
    heuristic_settings_for_policy,
)
from src.graph.edges import complete_undirected_edges
from src.rl.agents import get_agent_class
from src.rl.config import load_config
from src.rl.networks import require_torch, torch
from src.rl.training_state import (
    FORMAT_VERSION,
    _load_agent_state_dict,
    training_contract_sha256,
)


LAYERS = (
    "complete_undirected_edges",
    "facility_net_action_from_state",
    "batched_base_action",
    "tensor_cpu_numpy",
    "cpu_actor_critic_update",
    "cuda_actor_critic_update",
)
REPLAY_ARRAY_NAMES = (
    "states",
    "actions",
    "rewards",
    "next_states",
    "dones",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_array(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape)).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def replay_array_hashes(replay: dict[str, Any]) -> dict[str, str]:
    return {
        name: sha256_array(replay[name])
        for name in REPLAY_ARRAY_NAMES
    }


def independent_contiguous_numpy(tensor: Any) -> np.ndarray:
    """Return an owned C-contiguous NumPy copy from any torch tensor."""

    cpu_tensor = tensor.detach().to(device="cpu").contiguous()
    view = cpu_tensor.numpy()
    result = np.array(view, dtype=np.float32, order="C", copy=True)
    if not result.flags.c_contiguous or not result.flags.owndata:
        raise RuntimeError("Tensor conversion did not produce owned C-contiguous memory")
    if np.shares_memory(result, view):
        raise RuntimeError("Tensor conversion retained shared NumPy storage")
    return result


def immutable_replay_states(checkpoint: dict[str, Any]) -> np.ndarray:
    replay = dict(checkpoint["agent"]["replay_buffer"])
    states = np.array(replay["states"], dtype=np.float32, order="C", copy=True)
    if states.ndim != 2 or states.shape[0] < 1:
        raise ValueError("Recovery 7 replay states are empty or malformed")
    states.setflags(write=False)
    return states


def load_frozen_training_state(
    path: str | Path,
    *,
    algorithm: str,
    seed: int,
) -> dict[str, Any]:
    require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported Recovery 7 training-state format")
    if str(checkpoint.get("algorithm")) != algorithm:
        raise ValueError("Recovery 7 training-state algorithm mismatch")
    if int(checkpoint.get("seed", -1)) != int(seed):
        raise ValueError("Recovery 7 training-state seed mismatch")
    replay = checkpoint.get("agent", {}).get("replay_buffer", {})
    missing = [name for name in REPLAY_ARRAY_NAMES if name not in replay]
    if missing:
        raise ValueError(f"Recovery 7 replay state is missing arrays: {missing}")
    return checkpoint


def build_locked_smoke_agent_config(
    smoke_config_path: str | Path,
    *,
    algorithm: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reconstruct the exact scientific config used by the frozen Smoke.

    This deliberately mirrors ``train_one_multiscenario_agent`` without
    creating an environment or any output directories.  The training-state
    contract hash is the final authority that the reconstruction is exact.
    """

    run_config = load_config(smoke_config_path)
    plan = load_benchmark_plan(run_config["plan"])
    budget_name = str(run_config["budget"])
    budget = resolve_budget(plan, budget_name)
    scenario_names = tuple(str(value) for value in run_config["scenarios"])
    selected_scenarios = select_scenarios(plan, scenario_names)
    scenarios_by_name = {
        str(scenario["name"]): scenario
        for scenario in selected_scenarios
    }
    scenarios = [scenarios_by_name[name] for name in scenario_names]
    reference_name = str(run_config.get("reference_scenario", scenario_names[0]))
    reference_scenario = select_scenarios(plan, (reference_name,))[0]
    teacher_cache = Path(run_config["teacher_cache"])
    if not teacher_cache.is_file():
        raise FileNotFoundError(teacher_cache)
    demand_history_window = int(run_config.get("demand_history_window", 12))
    online_episodes = int(run_config.get("online_episodes", 0))
    pretrain_epochs_value = run_config.get("pretrain_epochs")
    pretrain_epochs = (
        None if pretrain_epochs_value is None else int(pretrain_epochs_value)
    )
    common_overrides = dict(run_config.get("config_overrides", {}))
    matching_entries = [
        dict(entry)
        for entry in run_config.get("algorithms", ())
        if str(entry["name"]) == algorithm
    ]
    if len(matching_entries) != 1:
        raise ValueError(f"Expected one Smoke entry for {algorithm!r}")
    entry = matching_entries[0]
    entry_seeds = tuple(
        int(value)
        for value in entry.get("seeds", run_config.get("seeds", (0,)))
    )
    if int(seed) not in entry_seeds:
        raise ValueError(f"Smoke entry does not preregister seed {seed}")
    algorithm_overrides = dict(entry.get("config_overrides", {}))

    config = make_history_screen_config(
        plan,
        budget_name=f"multiscenario_{budget_name}",
        budget=budget,
        algorithm=algorithm,
        scenario=reference_scenario,
        seed=int(seed),
        teacher_cache=teacher_cache,
        demand_history_window=demand_history_window,
        online_episodes=online_episodes,
        pretrain_epochs=pretrain_epochs,
    )
    config = deep_update(config, common_overrides)
    config = deep_update(config, algorithm_overrides)
    config["algorithm"] = algorithm
    config["seed"] = int(seed)
    config["num_episodes"] = online_episodes
    config.setdefault("checkpoint_interval", max(online_episodes, 1))
    config["save_pretrain_checkpoint"] = True
    config.setdefault("elite_imitation", {})["enabled"] = False
    config.setdefault("advantage_distillation_pretrain", {})[
        "demonstration_path"
    ] = str(teacher_cache)
    config["multi_scenario_training"] = {
        "scenarios": list(scenario_names),
        "scenario_schedule": "episode_round_robin",
        "scenario_start_index": int(seed) % len(scenarios),
        "scenario_label_in_observation": False,
        "teacher_cache": str(teacher_cache),
        "env_overrides": multiscenario_env_overrides(
            common_overrides=common_overrides,
            algorithm_overrides=algorithm_overrides,
            demand_history_window=demand_history_window,
        ),
    }
    config["env"] = merge_multiscenario_env_overrides(
        make_training_config(
            plan,
            budget_name,
            budget,
            algorithm,
            reference_scenario,
            int(seed),
        )["env"],
        common_overrides=common_overrides,
        algorithm_overrides=algorithm_overrides,
        demand_history_window=demand_history_window,
    )
    metadata = {
        "budget": budget_name,
        "reference_scenario": reference_name,
        "scenarios": list(scenario_names),
        "teacher_cache": str(teacher_cache),
        "online_episodes": online_episodes,
    }
    return config, metadata


def validate_training_contract(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
) -> str:
    actual = training_contract_sha256(config)
    expected = str(checkpoint.get("training_contract_sha256", ""))
    if actual != expected:
        raise ValueError(
            "Recovery 7 training-state scientific contract mismatch: "
            f"expected={expected}, reconstructed={actual}"
        )
    return actual


def _progress(layer: str, iteration: int, total: int) -> None:
    print(
        f"RECOVERY8_PROGRESS layer={layer} iteration={iteration}/{total}",
        flush=True,
    )


def _progress_interval(iterations: int) -> int:
    return max(iterations // 20, 1)


def stress_complete_edges(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    del checkpoint
    num_facilities = int(config["env"]["num_facilities"])
    expected = tuple(
        (i, j)
        for i in range(num_facilities)
        for j in range(i + 1, num_facilities)
    )
    for index in range(iterations):
        if complete_undirected_edges(num_facilities) != expected:
            raise RuntimeError("Complete-edge construction changed under stress")
        if (index + 1) % _progress_interval(iterations) == 0:
            _progress("complete_undirected_edges", index + 1, iterations)
    return {
        "num_facilities": num_facilities,
        "edge_count": len(expected),
        "iterations": iterations,
    }


def stress_facility_action(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    states = immutable_replay_states(checkpoint)
    state_hash = sha256_array(states)
    residual = dict(config.get("residual_action", {}))
    base_policy = str(residual.get("base_policy", ""))
    settings = heuristic_settings_for_policy(
        base_policy,
        dict(residual.get("base_policy_config", {})),
    )
    action_dim = int(np.asarray(checkpoint["agent"]["replay_buffer"]["actions"]).shape[1])
    digest = hashlib.sha256()
    for index in range(iterations):
        action = facility_net_action_from_state(
            states[index % states.shape[0]],
            config["env"],
            settings=settings,
        )
        if action.shape != (action_dim,) or not np.all(np.isfinite(action)):
            raise RuntimeError("MDL-2 facility action became invalid under stress")
        digest.update(np.ascontiguousarray(action).tobytes())
        if (index + 1) % _progress_interval(iterations) == 0:
            _progress("facility_net_action_from_state", index + 1, iterations)
    if sha256_array(states) != state_hash:
        raise RuntimeError("Facility-action stress mutated immutable replay states")
    return {
        "base_policy": base_policy,
        "replay_rows": int(states.shape[0]),
        "action_dim": action_dim,
        "iterations": iterations,
        "output_digest": digest.hexdigest(),
        "replay_states_sha256": state_hash,
    }


def _agent_for_device(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    device: str,
    load_state: bool,
) -> Any:
    diagnostic_config = copy.deepcopy(config)
    diagnostic_config["device"] = device
    replay = checkpoint["agent"]["replay_buffer"]
    state_dim = int(np.asarray(replay["states"]).shape[1])
    action_dim = int(np.asarray(replay["actions"]).shape[1])
    agent = get_agent_class(str(checkpoint["algorithm"]))(
        state_dim,
        action_dim,
        diagnostic_config,
    )
    if load_state:
        agent_state = dict(checkpoint["agent"])
        if device == "cpu":
            agent_state["torch_cuda_rng_states"] = []
        _load_agent_state_dict(agent, agent_state)
    return agent


def stress_batched_base_action(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    states = immutable_replay_states(checkpoint)
    state_hash = sha256_array(states)
    agent = _agent_for_device(
        config,
        checkpoint,
        device="cpu",
        load_state=False,
    )
    batch_size = min(int(config.get("batch_size", 64)), states.shape[0])
    digest = hashlib.sha256()
    for index in range(iterations):
        start = (index * batch_size) % states.shape[0]
        indices = (np.arange(batch_size) + start) % states.shape[0]
        batch = np.array(states[indices], dtype=np.float32, order="C", copy=True)
        state_tensor = torch.as_tensor(batch, dtype=torch.float32, device="cpu")
        actions = agent._base_actions_from_states_tensor(state_tensor)
        actions_np = independent_contiguous_numpy(actions)
        if not np.all(np.isfinite(actions_np)):
            raise RuntimeError("Batched MDL-2 base action became non-finite")
        digest.update(actions_np.tobytes(order="C"))
        if (index + 1) % _progress_interval(iterations) == 0:
            _progress("batched_base_action", index + 1, iterations)
    if sha256_array(states) != state_hash:
        raise RuntimeError("Batched base-action stress mutated replay states")
    return {
        "batch_size": batch_size,
        "iterations": iterations,
        "output_digest": digest.hexdigest(),
        "replay_states_sha256": state_hash,
    }


def stress_tensor_cpu_numpy(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    iterations: int,
) -> dict[str, Any]:
    del config
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for tensor-to-NumPy stress")
    device_name = str(torch.cuda.get_device_name(0))
    if "4090" not in device_name:
        raise RuntimeError(f"Expected RTX 4090, found {device_name!r}")
    states = immutable_replay_states(checkpoint)
    state_hash = sha256_array(states)
    batch_size = min(64, states.shape[0])
    source = torch.as_tensor(
        np.array(states[:batch_size], copy=True, order="C"),
        dtype=torch.float32,
        device="cuda",
    )
    digest = hashlib.sha256()
    for index in range(iterations):
        if (index + 1) % _progress_interval(iterations) == 0:
            print(
                "RECOVERY8_TENSOR_NUMPY "
                f"iteration={index + 1}/{iterations} stage=before_copy",
                flush=True,
            )
        converted = independent_contiguous_numpy(source)
        if not np.all(np.isfinite(converted)):
            raise RuntimeError("Tensor-to-NumPy conversion became non-finite")
        digest.update(converted.tobytes(order="C"))
        if (index + 1) % _progress_interval(iterations) == 0:
            torch.cuda.synchronize()
            _progress("tensor_cpu_numpy", index + 1, iterations)
    if sha256_array(states) != state_hash:
        raise RuntimeError("Tensor-to-NumPy stress mutated replay states")
    return {
        "batch_size": batch_size,
        "device_name": device_name,
        "iterations": iterations,
        "owned_contiguous_copy": True,
        "output_digest": digest.hexdigest(),
        "replay_states_sha256": state_hash,
    }


def stress_actor_critic_update(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
    iterations: int,
    *,
    device: str,
) -> dict[str, Any]:
    if device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for CUDA update stress")
        device_name = str(torch.cuda.get_device_name(0))
        if "4090" not in device_name:
            raise RuntimeError(f"Expected RTX 4090, found {device_name!r}")
    else:
        device_name = "cpu"
    source_replay = checkpoint["agent"]["replay_buffer"]
    source_hashes = replay_array_hashes(source_replay)
    agent = _agent_for_device(
        config,
        checkpoint,
        device=device,
        load_state=True,
    )
    metric_ranges: dict[str, list[float]] = {}
    for index in range(iterations):
        print(
            "RECOVERY8_UPDATE "
            f"device={device} iteration={index + 1}/{iterations} stage=before",
            flush=True,
        )
        metrics = dict(agent.update() or {})
        if not metrics:
            raise RuntimeError("Actor/critic update returned no metrics")
        for name, raw_value in metrics.items():
            value = float(raw_value)
            if not np.isfinite(value):
                raise RuntimeError(f"Update metric {name!r} is non-finite")
            bounds = metric_ranges.setdefault(name, [value, value])
            bounds[0] = min(bounds[0], value)
            bounds[1] = max(bounds[1], value)
        if device == "cuda":
            torch.cuda.synchronize()
        if (index + 1) % _progress_interval(iterations) == 0:
            _progress(f"{device}_actor_critic_update", index + 1, iterations)
    if replay_array_hashes(source_replay) != source_hashes:
        raise RuntimeError("Update stress mutated the loaded checkpoint payload")
    return {
        "device": device,
        "device_name": device_name,
        "iterations": iterations,
        "initial_total_updates": int(checkpoint["agent"]["total_updates"]),
        "final_total_updates": int(agent.total_updates),
        "metric_ranges": {
            name: {"minimum": bounds[0], "maximum": bounds[1]}
            for name, bounds in sorted(metric_ranges.items())
        },
        "source_replay_sha256": source_hashes,
    }


LAYER_FUNCTIONS: dict[
    str,
    Callable[[dict[str, Any], dict[str, Any], int], dict[str, Any]],
] = {
    "complete_undirected_edges": stress_complete_edges,
    "facility_net_action_from_state": stress_facility_action,
    "batched_base_action": stress_batched_base_action,
    "tensor_cpu_numpy": stress_tensor_cpu_numpy,
    "cpu_actor_critic_update": lambda config, checkpoint, iterations: (
        stress_actor_critic_update(
            config,
            checkpoint,
            iterations,
            device="cpu",
        )
    ),
    "cuda_actor_critic_update": lambda config, checkpoint, iterations: (
        stress_actor_critic_update(
            config,
            checkpoint,
            iterations,
            device="cuda",
        )
    ),
}


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic result: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_layer(
    *,
    layer: str,
    training_state: Path,
    smoke_config: Path,
    algorithm: str,
    seed: int,
    iterations: int,
) -> dict[str, Any]:
    if iterations < 1:
        raise ValueError("iterations must be positive")
    input_hashes_before = {
        "training_state": sha256_file(training_state),
        "smoke_config": sha256_file(smoke_config),
    }
    checkpoint = load_frozen_training_state(
        training_state,
        algorithm=algorithm,
        seed=seed,
    )
    config, config_metadata = build_locked_smoke_agent_config(
        smoke_config,
        algorithm=algorithm,
        seed=seed,
    )
    contract_sha256 = validate_training_contract(checkpoint, config)
    replay = checkpoint["agent"]["replay_buffer"]
    replay_hashes_before = replay_array_hashes(replay)
    started = time.perf_counter()
    details = LAYER_FUNCTIONS[layer](config, checkpoint, iterations)
    elapsed = time.perf_counter() - started
    replay_hashes_after = replay_array_hashes(replay)
    input_hashes_after = {
        "training_state": sha256_file(training_state),
        "smoke_config": sha256_file(smoke_config),
    }
    if input_hashes_after != input_hashes_before:
        raise RuntimeError("A frozen diagnostic input changed during execution")
    if replay_hashes_after != replay_hashes_before:
        raise RuntimeError("The in-memory frozen replay payload changed")
    return {
        "schema_version": 1,
        "status": "PASS",
        "layer": layer,
        "iterations": iterations,
        "algorithm": algorithm,
        "seed": int(seed),
        "pid": os.getpid(),
        "ppid": os.getppid(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": str(torch.__version__),
        "cuda_available": bool(torch.cuda.is_available()),
        "training_contract_sha256": contract_sha256,
        "checkpoint_total_updates": int(checkpoint["agent"]["total_updates"]),
        "checkpoint_training": dict(checkpoint.get("training", {})),
        "config": config_metadata,
        "input_sha256_before": input_hashes_before,
        "input_sha256_after": input_hashes_after,
        "replay_sha256_before": replay_hashes_before,
        "replay_sha256_after": replay_hashes_after,
        "elapsed_seconds": elapsed,
        "details": details,
    }


def main() -> None:
    faulthandler.enable(all_threads=True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", choices=LAYERS, required=True)
    parser.add_argument("--training-state", required=True)
    parser.add_argument("--smoke-config", required=True)
    parser.add_argument(
        "--algorithm",
        default="gcn_residual_mdl2_network_ddpg_afd",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite diagnostic result: {output}")
    print(
        "RECOVERY8_LAYER_START "
        f"layer={args.layer} pid={os.getpid()} ppid={os.getppid()} "
        f"iterations={args.iterations}",
        flush=True,
    )
    started_at = utc_now()
    try:
        payload = run_layer(
            layer=args.layer,
            training_state=Path(args.training_state),
            smoke_config=Path(args.smoke_config),
            algorithm=str(args.algorithm),
            seed=int(args.seed),
            iterations=int(args.iterations),
        )
        payload["started_at"] = started_at
        payload["completed_at"] = utc_now()
        atomic_write_json(output, payload)
        print(
            f"RECOVERY8_LAYER_PASS layer={args.layer} output={output}",
            flush=True,
        )
    except BaseException as error:
        failure = {
            "schema_version": 1,
            "status": "FAIL",
            "layer": args.layer,
            "iterations": int(args.iterations),
            "algorithm": str(args.algorithm),
            "seed": int(args.seed),
            "pid": os.getpid(),
            "ppid": os.getppid(),
            "started_at": started_at,
            "completed_at": utc_now(),
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(),
        }
        atomic_write_json(output, failure)
        print(
            f"RECOVERY8_LAYER_FAIL layer={args.layer} error={error}",
            file=sys.stderr,
            flush=True,
        )
        raise


if __name__ == "__main__":
    main()
