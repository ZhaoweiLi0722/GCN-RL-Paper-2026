"""Atomic, full-state checkpoints for resumable off-policy training."""

from __future__ import annotations

import hashlib
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - exercised in CPU-only installs
    torch = None


FORMAT_VERSION = 1
_MODULE_NAMES = (
    "actor",
    "actor_target",
    "actor_reference",
    "critic",
    "critic_target",
    "critic2",
    "critic2_target",
    "pretrain_reference_actor",
    "correction_gate",
    "correction_safety_gate",
)
_OPTIMIZER_NAMES = (
    "actor_optimizer",
    "critic_optimizer",
    "critic2_optimizer",
    "correction_gate_optimizer",
    "correction_safety_gate_optimizer",
)
_IMITATION_TENSOR_NAMES = (
    "imitation_states",
    "imitation_actions",
    "imitation_node_features",
    "imitation_weights",
)
_EXECUTION_ONLY_CONFIG_KEYS = frozenset(
    {
        "checkpoint_dir",
        "checkpoint_interval",
        "config_snapshot_path",
        "preonline_fork_allowed_overrides",
        "preonline_training_state_path",
        "progress_interval",
        "result_csv_path",
        "resume_training_state_path",
        "training_state_checkpoint_interval",
        "training_state_checkpoint_path",
    }
)
_PREONLINE_FORKABLE_CONFIG_PATHS = frozenset(
    {
        "anchor_advantage_actor_loss",
        "actor_update_frequency",
        "critic_teacher_advantage_calibration",
        "critic_teacher_advantage_calibration.enabled",
        "critic_teacher_advantage_calibration.target_scale",
        "critic_teacher_advantage_calibration.updates",
        "exploration_noise.sigma",
        "history_screen.online_episodes",
        "imitation_pretrain.regularization_weight",
        "num_episodes",
        "online_actor_lr",
        "online_advantage_self_imitation",
        "online_advantage_self_imitation.enabled",
        "online_advantage_self_imitation.minimum_return",
        "online_advantage_self_imitation.release_pretrain_reference",
        "online_advantage_self_imitation.require_positive_one_step_return",
        "online_advantage_self_imitation.weight",
        "online_critic_lr",
        "online_imitation_regularization",
        "online_replay_fraction",
        "pretrain_reference_actor_loss.action_space",
        "pretrain_reference_actor_loss.mode",
        "pretrain_reference_actor_loss.q_filter_margin",
        "pretrain_reference_actor_loss.weight",
        "residual_action.correction_gate.align_online_policy",
        "residual_action.correction_gate.hard_actor_policy",
        "residual_action.correction_gate.differentiate_actor_proposal",
        "residual_action.online_reward_mode",
        "residual_action.online_reward_n_step_horizon",
        "specimen_action_quantization",
        "updates_per_update",
    }
)


def training_contract_sha256(config: dict[str, Any]) -> str:
    """Hash scientific settings while ignoring execution-only file controls."""

    scientific = _scientific_training_contract(config)
    payload = json.dumps(
        scientific,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scientific_training_contract(config: dict[str, Any]) -> dict[str, Any]:
    return _jsonable({
        key: value
        for key, value in config.items()
        if key not in _EXECUTION_ONLY_CONFIG_KEYS
    })


def save_off_policy_training_state(
    agent: Any,
    path: str | Path,
    *,
    config: dict[str, Any],
    training: dict[str, Any],
    env: Any | None = None,
) -> Path:
    """Atomically save model, optimizer, replay, RNG, loop, and env state."""

    _require_torch()
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
    training_contract = _scientific_training_contract(config)
    payload = {
        "format_version": FORMAT_VERSION,
        "algorithm": str(agent.algorithm),
        "seed": int(agent.seed),
        "training_contract": training_contract,
        "training_contract_sha256": training_contract_sha256(config),
        "agent": _agent_state_dict(agent),
        "training": dict(training),
        "environment": _environment_state_dict(env),
    }
    try:
        torch.save(payload, temporary)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def load_off_policy_training_state(
    agent: Any,
    path: str | Path,
    *,
    config: dict[str, Any],
    env: Any | None = None,
    restore_environment: bool = True,
) -> dict[str, Any]:
    """Restore a checkpoint and return its training-loop metadata."""

    _require_torch()
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=agent.device,
        weights_only=False,
    )
    if int(checkpoint.get("format_version", -1)) != FORMAT_VERSION:
        raise ValueError("Unsupported off-policy training-state format")
    if str(checkpoint.get("algorithm")) != str(agent.algorithm):
        raise ValueError("Training-state algorithm does not match")
    if int(checkpoint.get("seed", -1)) != int(agent.seed):
        raise ValueError("Training-state seed does not match")
    expected_contract = training_contract_sha256(config)
    fork_overrides: dict[str, dict[str, Any]] = {}
    if checkpoint.get("training_contract_sha256") != expected_contract:
        fork_overrides = _validate_preonline_fork(
            checkpoint,
            config=config,
        )
    _load_agent_state_dict(
        agent,
        checkpoint["agent"],
        fork_overrides=fork_overrides,
    )
    if restore_environment:
        _load_environment_state_dict(env, checkpoint.get("environment"))
    metadata = dict(checkpoint["training"])
    if fork_overrides:
        metadata["preonline_fork_overrides"] = fork_overrides
    return metadata


def _validate_preonline_fork(
    checkpoint: dict[str, Any],
    *,
    config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    allowed = tuple(
        str(path)
        for path in config.get("preonline_fork_allowed_overrides", ())
    )
    if not allowed:
        raise ValueError(
            "Training-state scientific contract does not match the current config"
        )
    unsupported = sorted(
        set(allowed) - _PREONLINE_FORKABLE_CONFIG_PATHS
    )
    if unsupported:
        raise ValueError(
            "Unsupported pre-online fork override paths: "
            + ", ".join(unsupported)
        )
    training = dict(checkpoint.get("training", {}))
    if int(training.get("next_episode", -1)) != 0:
        raise ValueError(
            "Training-state scientific contract may be forked only at "
            "the pre-online episode-0 boundary"
        )
    checkpoint_contract = checkpoint.get("training_contract")
    if not isinstance(checkpoint_contract, dict):
        raise ValueError(
            "Training-state scientific contract lacks forkable contract metadata"
        )
    current_contract = _scientific_training_contract(config)
    differences = _contract_differences(
        checkpoint_contract,
        current_contract,
    )
    disallowed = sorted(set(differences) - set(allowed))
    if not differences or disallowed:
        detail = ", ".join(disallowed) if disallowed else "none"
        raise ValueError(
            "Training-state scientific contract has disallowed pre-online "
            f"fork differences: {detail}"
        )
    return differences


def _contract_differences(
    checkpoint_value: Any,
    current_value: Any,
    path: str = "",
) -> dict[str, dict[str, Any]]:
    if isinstance(checkpoint_value, dict) and isinstance(current_value, dict):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(checkpoint_value) | set(current_value)):
            child_path = f"{path}.{key}" if path else str(key)
            if key not in checkpoint_value or key not in current_value:
                differences[child_path] = {
                    "checkpoint": checkpoint_value.get(key),
                    "current": current_value.get(key),
                }
                continue
            differences.update(
                _contract_differences(
                    checkpoint_value[key],
                    current_value[key],
                    child_path,
                )
            )
        return differences
    if checkpoint_value == current_value:
        return {}
    return {
        path: {
            "checkpoint": checkpoint_value,
            "current": current_value,
        }
    }


def _environment_state_dict(env: Any | None) -> dict[str, Any] | None:
    if env is None:
        return None
    snapshot = getattr(env, "state_dict", None)
    if not callable(snapshot):
        return None
    return dict(snapshot())


def _load_environment_state_dict(
    env: Any | None,
    state: dict[str, Any] | None,
) -> None:
    if state is None:
        return
    if env is None:
        raise ValueError("Training state contains environment state but no env was supplied")
    restore = getattr(env, "load_state_dict", None)
    if not callable(restore):
        raise ValueError("Environment cannot restore the saved training state")
    restore(dict(state))


def _agent_state_dict(agent: Any) -> dict[str, Any]:
    modules = {}
    module_modes = {}
    for name in _MODULE_NAMES:
        module = getattr(agent, name, None)
        if module is not None:
            modules[name] = module.state_dict()
            module_modes[name] = bool(module.training)
    optimizers = {
        name: optimizer.state_dict()
        for name in _OPTIMIZER_NAMES
        if (optimizer := getattr(agent, name, None)) is not None
    }
    imitation_tensors = {}
    for name in _IMITATION_TENSOR_NAMES:
        value = getattr(agent, name, None)
        imitation_tensors[name] = (
            None if value is None else value.detach().cpu()
        )
    cuda_rng_states = []
    if torch.cuda.is_available():
        cuda_rng_states = [state.cpu() for state in torch.cuda.get_rng_state_all()]
    return {
        "modules": modules,
        "module_modes": module_modes,
        "optimizers": optimizers,
        "total_updates": int(getattr(agent, "total_updates", 0)),
        "replay_buffer": agent.replay_buffer.state_dict(),
        "noise": agent.noise.state_dict(),
        "imitation_tensors": imitation_tensors,
        "imitation_rng_state": agent.imitation_rng.bit_generator.state,
        "critic_teacher_advantage_rng_state": (
            None
            if getattr(agent, "critic_teacher_advantage_rng", None) is None
            else agent.critic_teacher_advantage_rng.bit_generator.state
        ),
        "python_rng_state": random.getstate(),
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state().cpu(),
        "torch_cuda_rng_states": cuda_rng_states,
    }


def _load_agent_state_dict(
    agent: Any,
    state: dict[str, Any],
    *,
    fork_overrides: dict[str, dict[str, Any]] | None = None,
) -> None:
    modules = dict(state["modules"])
    for name, module_state in modules.items():
        module = getattr(agent, name, None)
        if module is None:
            raise ValueError(f"Training-state module is unavailable: {name}")
        module.load_state_dict(module_state)
    for name, is_training in dict(state.get("module_modes", {})).items():
        module = getattr(agent, name, None)
        if module is not None:
            module.train(bool(is_training))
    for name, optimizer_state in dict(state["optimizers"]).items():
        optimizer = getattr(agent, name, None)
        if optimizer is None:
            raise ValueError(f"Training-state optimizer is unavailable: {name}")
        optimizer.load_state_dict(optimizer_state)
        _move_optimizer_state(optimizer, agent.device)
    agent.total_updates = int(state["total_updates"])
    agent.replay_buffer.load_state_dict(state["replay_buffer"])
    noise_state = dict(state["noise"])
    if "exploration_noise.sigma" in (fork_overrides or {}):
        saved_noise = np.asarray(noise_state["state"], dtype=np.float32)
        saved_mean = np.asarray(noise_state["mu"], dtype=np.float32)
        if not np.array_equal(saved_noise, saved_mean):
            raise ValueError(
                "Exploration sigma may be forked only from a reset "
                "episode-0 OU-noise state"
            )
        noise_state["sigma"] = float(agent.noise.sigma)
    agent.noise.load_state_dict(noise_state)
    for name, value in dict(state["imitation_tensors"]).items():
        setattr(
            agent,
            name,
            None if value is None else value.to(agent.device),
        )
    agent.imitation_rng.bit_generator.state = state["imitation_rng_state"]
    critic_advantage_rng_state = state.get(
        "critic_teacher_advantage_rng_state"
    )
    critic_advantage_rng = getattr(
        agent,
        "critic_teacher_advantage_rng",
        None,
    )
    if (
        critic_advantage_rng_state is not None
        and critic_advantage_rng is not None
    ):
        critic_advantage_rng.bit_generator.state = (
            critic_advantage_rng_state
        )
    random.setstate(state["python_rng_state"])
    np.random.set_state(state["numpy_rng_state"])
    torch.set_rng_state(state["torch_rng_state"].cpu())
    cuda_rng_states = list(state.get("torch_cuda_rng_states", ()))
    if cuda_rng_states:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA RNG state cannot be restored without CUDA")
        torch.cuda.set_rng_state_all(cuda_rng_states)


def _move_optimizer_state(optimizer: Any, device: Any) -> None:
    for optimizer_state in optimizer.state.values():
        for key, value in optimizer_state.items():
            if torch.is_tensor(value):
                optimizer_state[key] = value.to(device)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def _require_torch() -> None:
    if torch is None:
        raise RuntimeError("PyTorch is required for off-policy training state")
