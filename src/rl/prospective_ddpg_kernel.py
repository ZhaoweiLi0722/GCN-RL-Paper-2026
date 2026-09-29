"""Opt-in, capped DDPG update kernel for typed replay; not a campaign runner.

Owns a cloned network, Adam, a small typed-window ring and its sampling RNG.
Checkpoints cover update boundaries only, never environment/collector state.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from src.models.matched_inputs import build_matched_inputs
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch, require_torch
from src.rl.prospective_adapter import prepare_replay_batch
from src.rl.training_state import training_contract_sha256
from src.rl.validated_returns import OneStepRecord, ReplaySemantics


def _json_tensor(value):
    if isinstance(value, torch.Tensor):
        return {"tensor_dtype": str(value.dtype), "shape": list(value.shape),
                "values": value.detach().cpu().tolist()}
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported kernel-state type: {type(value).__name__}")


def state_digest(value):
    data = json.dumps(value, default=_json_tensor, sort_keys=True,
                      separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class KernelSettings:
    actor_lr: float
    critic_lr: float
    tau: float
    batch_size: int
    replay_capacity: int
    max_updates: int
    max_return_steps: int
    differentiate_gate_proposal: bool

    def __post_init__(self):
        for name in ("actor_lr", "critic_lr", "tau"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite number")
        if self.actor_lr <= 0 or self.critic_lr <= 0 or not 0 <= self.tau <= 1:
            raise ValueError("positive learning rates and tau in [0,1] required")
        for name in ("batch_size", "replay_capacity", "max_updates", "max_return_steps"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch size exceeds replay capacity")
        if type(self.differentiate_gate_proposal) is not bool:
            raise ValueError("gate proposal gradient policy must be explicit")


def deployed_request(agent, actor_view, *, differentiate_gate_proposal):
    """Differentiable counterpart of the unchanged N4 inference transform.

    Equivalence with N4's no-grad path is tested. Fixed gate parameters do not
    imply detaching its input: that distinct choice is explicit here.
    """
    if type(differentiate_gate_proposal) is not bool:
        raise ValueError("explicit gate proposal policy required")
    observation, anchor = agent._parts(actor_view)
    proposal = torch.clamp(anchor + agent.residual_scales * agent.actor(actor_view), -1, 1)
    gate_view = build_matched_inputs(observation, agent.contract.inputs, anchor, role="gate",
                                     message_mode=agent.message_mode, action=proposal,
                                     detach_proposal=not differentiate_gate_proposal)
    gate = agent.gate(gate_view)
    return torch.clamp(anchor + gate * (proposal - anchor), -1, 1)


def _finite_grad_norm(module):
    grads = [p.grad for p in module.parameters()]
    if any(g is None or not torch.isfinite(g).all().item() for g in grads):
        raise ValueError("missing/nonfinite gradient")
    norm = torch.linalg.vector_norm(torch.cat([g.detach().flatten() for g in grads])).item()
    if not math.isfinite(norm):
        raise ValueError("nonfinite gradient norm")
    return norm


class ProspectiveDDPGKernel:
    def __init__(self, prototype, settings, *, enabled=False, mode, replay_seed=0):
        require_torch()
        if enabled is not True:
            raise ValueError("kernel must be explicitly enabled")
        if type(prototype) is not ProspectiveForwardAgent or not isinstance(settings, KernelSettings):
            raise TypeError("prospective prototype and explicit settings required")
        if mode not in ("online", "frozen") or type(replay_seed) is not int or replay_seed < 0:
            raise ValueError("explicit mode and nonnegative replay seed required")
        if any(p.device.type != "cpu" or p.dtype != torch.float32 for p in prototype.parameters()):
            raise ValueError("this engineering kernel supports CPU float32 only")
        self.agent = copy.deepcopy(prototype)
        self.settings, self.mode = settings, mode
        self.contract = prototype.contract
        self._manifest = {
            "format": "prospective-ddpg-kernel-v1", "contract": asdict(self.contract),
            "settings": asdict(settings), "mode": mode, "replay_seed": replay_seed,
            "message_mode": prototype.message_mode,
            "initial_weights_sha256": prototype.weights_digest(),
            "optimizer": {"kind": "Adam", "betas": [.9, .999], "eps": 1e-8,
                          "weight_decay": 0, "foreach": False}, "device": "cpu", "dtype": "float32",
        }
        self.manifest_sha256 = training_contract_sha256(self._manifest)
        self._fixed_gate = state_digest(prototype.gate.state_dict())
        self._fixed_target_gate = state_digest(prototype.target_gate.state_dict())
        self._fixed_scales = prototype.residual_scales.clone()
        self._set_modes()
        self.actor_optimizer = self.critic_optimizer = None
        if mode == "online":
            self.actor_optimizer = torch.optim.Adam(self.agent.actor.parameters(), lr=settings.actor_lr, foreach=False)
            self.critic_optimizer = torch.optim.Adam(self.agent.critic.parameters(), lr=settings.critic_lr, foreach=False)
        self.windows = []
        self.position, self.total_updates = 0, 0
        self.rng = np.random.default_rng(replay_seed)

    def _set_modes(self):
        self.agent.eval()
        self.agent.requires_grad_(False)
        self.agent.zero_grad(set_to_none=True)
        if self.mode == "online":
            self.agent.actor.requires_grad_(True)
            self.agent.critic.requires_grad_(True)

    def _prepare(self, windows):
        for window in windows:
            if not 1 <= len(window) <= self.settings.max_return_steps:
                raise ValueError("return length outside declared kernel contract")
            if any(not isinstance(record, OneStepRecord) for record in window):
                raise TypeError("typed OneStepRecord required")
            if any(any(abs(value) > 1 for value in record.action) for record in window):
                raise ValueError("replay request outside normalized action coordinates")
        return prepare_replay_batch(windows, self.contract, dtype=torch.float32,
                                     device="cpu", message_mode=self.agent.message_mode)

    def add_windows(self, windows):
        windows = tuple(tuple(window) for window in windows)
        self._prepare(windows)
        for window in windows:
            if len(self.windows) < self.settings.replay_capacity:
                self.windows.append(window)
            else:
                self.windows[self.position] = window
            self.position = (self.position + 1) % self.settings.replay_capacity

    def _actor_loss(self, batch):
        request = deployed_request(self.agent, batch.current_actor,
                                   differentiate_gate_proposal=self.settings.differentiate_gate_proposal)
        observation, anchor = self.agent._parts(batch.current_actor)
        view = build_matched_inputs(observation, self.contract.inputs, anchor, role="critic",
                                    message_mode=self.agent.message_mode, action=request)
        return -self.agent.critic(view).mean()

    def _polyak(self):
        with torch.no_grad():
            for source, target in ((self.agent.actor, self.agent.target_actor),
                                   (self.agent.critic, self.agent.target_critic)):
                for current, previous in zip(source.parameters(), target.parameters()):
                    previous.mul_(1 - self.settings.tau).add_(current, alpha=self.settings.tau)

    def update(self):
        if self.mode != "online":
            raise ValueError("frozen arm cannot update")
        if self.total_updates >= self.settings.max_updates:
            raise ValueError("declared update cap reached")
        if len(self.windows) < self.settings.batch_size:
            raise ValueError("insufficient typed replay")
        # A failed actor step must not leave an advanced critic or sampling RNG.
        before = self.state_dict()
        try:
            indices = self.rng.choice(len(self.windows), self.settings.batch_size, replace=False).tolist()
            batch = self._prepare([self.windows[i] for i in indices])
            _, next_q, targets = self.agent.replay_forward(batch)
            if targets.requires_grad or next_q.requires_grad:
                raise ValueError("bootstrap graph was not detached")
            self.agent.zero_grad(set_to_none=True)
            critic_loss = torch.nn.functional.mse_loss(self.agent.critic(batch.current_critic), targets)
            if not torch.isfinite(critic_loss).item():
                raise ValueError("nonfinite critic loss")
            critic_loss.backward()
            critic_norm = _finite_grad_norm(self.agent.critic)
            if any(p.grad is not None for p in self.agent.actor.parameters()):
                raise ValueError("critic update leaked into actor")
            self.critic_optimizer.step()
            self.critic_optimizer.zero_grad(set_to_none=True)
            critic_after = state_digest(self.agent.critic.state_dict())
            self.agent.critic.requires_grad_(False)
            actor_loss = self._actor_loss(batch)
            if not torch.isfinite(actor_loss).item():
                raise ValueError("nonfinite actor loss")
            actor_loss.backward()
            actor_norm = _finite_grad_norm(self.agent.actor)
            if any(p.grad is not None for p in self.agent.critic.parameters()):
                raise ValueError("actor differentiation accumulated critic gradients")
            self.actor_optimizer.step()
            if critic_after != state_digest(self.agent.critic.state_dict()):
                raise ValueError("actor update changed critic parameters")
            self._polyak()
            self.total_updates += 1
            self._set_modes()
            self._check_fixed_parts()
            state_digest(self.state_dict())  # Reject nonfinite parameters/moments after either step.
            return {"update": self.total_updates, "sample_indices": indices,
                    "critic_loss": critic_loss.item(), "actor_loss": actor_loss.item(),
                    "critic_grad_norm": critic_norm, "actor_grad_norm": actor_norm,
                    "targets": targets[:, 0].tolist(), "bootstrap_detached": True,
                    "critic_fixed_during_actor": True}
        except Exception:
            self._restore(before)
            raise

    def _check_fixed_parts(self):
        if (state_digest(self.agent.gate.state_dict()) != self._fixed_gate
                or state_digest(self.agent.target_gate.state_dict()) != self._fixed_target_gate
                or not torch.equal(self.agent.residual_scales, self._fixed_scales)):
            raise ValueError("fixed gate or action scale changed")

    def state_dict(self):
        return copy.deepcopy({
            "manifest": self._manifest, "manifest_sha256": self.manifest_sha256,
            "modules": self.agent.state_dict(), "total_updates": self.total_updates,
            "actor_optimizer": None if self.actor_optimizer is None else self.actor_optimizer.state_dict(),
            "critic_optimizer": None if self.critic_optimizer is None else self.critic_optimizer.state_dict(),
            "windows": [[asdict(record) for record in window] for window in self.windows],
            "position": self.position, "rng": self.rng.bit_generator.state,
        })

    def _restore(self, state):
        if set(state) != {"manifest", "manifest_sha256", "modules", "total_updates",
                          "actor_optimizer", "critic_optimizer", "windows", "position", "rng"}:
            raise ValueError("checkpoint fields differ")
        if (state["manifest"] != self._manifest or state["manifest_sha256"] != self.manifest_sha256
                or training_contract_sha256(state["manifest"]) != self.manifest_sha256):
            raise ValueError("checkpoint semantic/model/optimizer manifest mismatch")
        state_digest(state)
        count, position = state["total_updates"], state["position"]
        if type(count) is not int or not 0 <= count <= self.settings.max_updates:
            raise ValueError("invalid update count")
        size = len(state["windows"])
        if (type(position) is not int or not 0 <= position < self.settings.replay_capacity
                or size > self.settings.replay_capacity or (size < self.settings.replay_capacity and position != size)):
            raise ValueError("invalid replay ring position/size")
        reference = self.agent.state_dict()
        if set(state["modules"]) != set(reference):
            raise ValueError("checkpoint module keys differ")
        for key, value in state["modules"].items():
            if not isinstance(value, torch.Tensor) or value.shape != reference[key].shape or value.dtype != reference[key].dtype:
                raise ValueError("checkpoint module shape/dtype differs")
        windows = []
        for window in state["windows"]:
            rows = []
            for record in window:
                data = dict(record)
                data["semantics"] = ReplaySemantics(**data["semantics"])
                rows.append(OneStepRecord(**data))
            windows.append(tuple(rows))
        if windows:
            self._prepare(windows)
        if count and size < self.settings.batch_size:
            raise ValueError("updated checkpoint lacks its replay")
        for name in ("actor_optimizer", "critic_optimizer"):
            optimizer = getattr(self, name)
            saved = state[name]
            if self.mode == "frozen":
                if saved is not None or count:
                    raise ValueError("frozen checkpoint contains updates")
                continue
            current_groups = optimizer.state_dict()["param_groups"]
            if len(saved["param_groups"]) != len(current_groups):
                raise ValueError("optimizer groups differ")
            for group, original in zip(saved["param_groups"], current_groups):
                if group != original:
                    raise ValueError("optimizer hyperparameters or parameter map differ")
            parameters = [p for g in optimizer.param_groups for p in g["params"]]
            ids = [i for g in saved["param_groups"] for i in g["params"]]
            if set(saved["state"]) != (set(ids) if count else set()):
                raise ValueError("optimizer moments missing or unexpected")
            for index, parameter in zip(ids, parameters):
                if not count:
                    break
                moments = saved["state"][index]
                if set(moments) != {"step", "exp_avg", "exp_avg_sq"}:
                    raise ValueError("invalid Adam moments")
                if (not isinstance(moments["step"], torch.Tensor) or moments["step"].shape != torch.Size([])
                        or moments["step"].dtype != torch.float32 or moments["step"].item() != count):
                    raise ValueError("Adam step counter differs")
                for field in ("exp_avg", "exp_avg_sq"):
                    if moments[field].shape != parameter.shape or moments[field].dtype != parameter.dtype:
                        raise ValueError("Adam moment shape/dtype differs")
                if (moments["exp_avg_sq"] < 0).any().item():
                    raise ValueError("negative Adam second moment")
            optimizer.load_state_dict(saved)
        self.agent.load_state_dict(state["modules"])
        self._check_fixed_parts()
        if not count and self.agent.weights_digest() != self._manifest["initial_weights_sha256"]:
            raise ValueError("zero-update checkpoint differs from its initial prototype")
        self.windows, self.position, self.total_updates = windows, position, count
        self.rng.bit_generator.state = copy.deepcopy(state["rng"])
        self._set_modes()

    def load_state_dict(self, state):
        # Validate on a private copy so malformed files never partially mutate the live kernel.
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(state))
        self.__dict__.update(candidate.__dict__)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        state = self.state_dict()
        envelope = {"state": state, "state_sha256": state_digest(state)}
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".kernel-", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(envelope, temporary)
            os.link(temporary, path)  # Atomic publication without replacing an existing file.
        finally:
            os.unlink(temporary)

    def load(self, path):
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        if set(envelope) != {"state", "state_sha256"} or state_digest(envelope["state"]) != envelope["state_sha256"]:
            raise ValueError("checkpoint payload checksum mismatch")
        self.load_state_dict(envelope["state"])
