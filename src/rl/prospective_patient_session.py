"""Opt-in single-episode engineering session; not a performance campaign driver."""

from __future__ import annotations

import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from src.models.matched_inputs import build_matched_inputs
from src.models.prospective_forward_agent import ForwardDecision
from src.rl.networks import torch
from src.rl.noise import OUNoise
from src.rl.patient_replay_collector import PatientReplayCollector, evidence_digest, json_value
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_adapter import pack_actor_state
from src.rl.validated_returns import OneStepRecord, ReplaySemantics, make_return


def encode_arrays(value):
    """Preserve ndarray dtype/shape without unsafe NumPy pickle globals."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind not in "biuf":
            raise ValueError("unsupported array dtype")
        return {"ndarray_dtype": value.dtype.str, "tensor": torch.from_numpy(value.copy())}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: encode_arrays(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(encode_arrays(item) for item in value)
    return copy.deepcopy(value)


def decode_arrays(value):
    if isinstance(value, dict):
        if set(value) == {"ndarray_dtype", "tensor"}:
            tensor = value["tensor"]
            if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu":
                raise ValueError("invalid encoded ndarray")
            array = tensor.detach().numpy().copy()
            if array.dtype.str != value["ndarray_dtype"] or array.dtype.kind not in "biuf":
                raise ValueError("encoded ndarray dtype mismatch")
            return array
        return {key: decode_arrays(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(decode_arrays(item) for item in value)
    return copy.deepcopy(value)


def restore_record(payload):
    data = dict(payload)
    data["semantics"] = ReplaySemantics(**data["semantics"])
    return OneStepRecord(**data)


def scheduled_windows(records, semantics, n_steps, closed):
    """A bounded reference schedule retains each start position exactly once."""
    records = tuple(records)
    if not records:
        return (), ()
    make_return(records, semantics)  # Check lineage across the entire finite segment.
    if bool(records[-1].terminated or records[-1].truncated) != closed:
        raise ValueError("record boundary differs from collector closure")
    count = len(records) if closed else max(0, len(records) - n_steps + 1)
    return tuple(records[i:i + n_steps] for i in range(count)), records[count:]


class PatientLearningSession:
    def __init__(self, env, producer, learner, *, enabled=False, trajectory_id,
                 max_steps, return_steps, horizon_end, noise_options, noise_seed=0):
        if enabled is not True:
            raise ValueError("live engineering session must be explicitly enabled")
        if (type(max_steps) is not int or not 1 <= max_steps <= 6
                or type(return_steps) is not int or not 1 <= return_steps <= 3):
            raise ValueError("bounded six-step/three-return engineering session required")
        if (learner.contract != producer.contract or learner.settings.max_return_steps != return_steps
                or learner.total_updates or learner.windows or learner.settings.max_updates > 3):
            raise ValueError("fresh compatible bounded kernel required")
        if (set(noise_options) != {"mu", "theta", "sigma"}
                or any(type(v) not in (int, float) or not np.isfinite(v) for v in noise_options.values())
                or noise_options["theta"] < 0 or noise_options["sigma"] < 0
                or type(noise_seed) is not int or noise_seed < 0):
            raise ValueError("explicit finite OU contract required")
        self.collector = PatientReplayCollector(env, producer, enabled=True, trajectory_id=trajectory_id,
                                                max_steps=max_steps, horizon_end=horizon_end, verify_clone=True)
        self.learner, self.return_steps = learner, return_steps
        self.noise = OUNoise(len(producer.contract.inputs.node_ids), seed=noise_seed, **noise_options)
        self.events, self.pending, self.emitted_count = [], (), 0
        self.manifest = {
            "format": "patient-learning-session-v1", "environment_config": asdict(env.env_config),
            "initial_environment_sha256": evidence_digest(env.state_dict()),
            "kernel_manifest_sha256": learner.manifest_sha256, "trajectory_id": trajectory_id,
            "max_steps": max_steps, "return_steps": return_steps, "horizon_end": horizon_end,
            "noise_options": dict(noise_options), "noise_seed": noise_seed,
            "exploration_location": "actor_specimen_residual_before_fixed_gate",
        }
        self.manifest_sha256 = state_digest(self.manifest)

    @torch.no_grad()
    def decision(self, noise):
        agent, producer = self.learner.agent, self.collector.producer
        observation, anchor = producer.observe(self.collector.env.observation())
        view = build_matched_inputs(observation, producer.contract.inputs, anchor,
                                    role="actor", message_mode=agent.message_mode)
        deterministic = agent.policy_tensors(view)[2][0].numpy().copy()
        if noise.shape != self.noise.state.shape or not np.isfinite(noise).all():
            raise ValueError("invalid exploration vector")
        offsets = torch.zeros_like(anchor)
        offsets[0, :noise.size] = torch.from_numpy(noise)
        proposal = torch.clamp(anchor + agent.residual_scales * (agent.actor(view) + offsets), -1, 1)
        gate_view = build_matched_inputs(observation, producer.contract.inputs, anchor, role="gate",
                                         message_mode=agent.message_mode, action=proposal, detach_proposal=True)
        gate = agent.gate(gate_view)
        request = torch.clamp(anchor + gate * (proposal - anchor), -1, 1)
        return ForwardDecision(*(t[0].numpy().copy() for t in (proposal, gate, request))), deterministic

    def step(self):
        if self.collector.closed:
            raise ValueError("session is closed; no reset or automatic second episode")
        self.collector.producer.check_environment(self.collector.env)
        if evidence_digest(self.collector.env.state_dict()) != self.collector.expected_token:
            raise ValueError("environment drift before session transaction")
        before = self.state_dict()
        try:
            noise = self.noise.sample().copy()
            decision, deterministic = self.decision(noise)
            receipt = self.collector.step(decision)
            payload = json.loads(json.dumps(asdict(receipt), default=json_value, allow_nan=False))
            records = [restore_record(event["receipt"]["record"]) for event in self.events] + [receipt.record]
            windows, pending = scheduled_windows(records, self.learner.contract.replay,
                                                  self.return_steps, self.collector.closed)
            if len(windows) > self.emitted_count:
                self.learner.add_windows(windows[self.emitted_count:])
            self.pending, self.emitted_count = pending, len(windows)
            update = None
            if self.learner.mode == "online" and len(self.learner.windows) >= self.learner.settings.batch_size:
                update = self.learner.update()
            event = {"receipt": payload, "noise": noise.tolist(), "deterministic_request": deterministic.tolist(),
                     "update": update, "kernel_state_sha256": state_digest(self.learner.state_dict())}
            self.events.append(event)
            state_digest(self.state_dict())
            return copy.deepcopy(event)
        except Exception:
            self.load_state_dict(before)
            raise

    def state_dict(self):
        collector = self.collector
        return encode_arrays({
            "manifest": self.manifest, "manifest_sha256": self.manifest_sha256,
            "environment": collector.env.state_dict(), "noise": self.noise.state_dict(),
            "kernel": self.learner.state_dict(), "events": self.events,
            "collector": {"index": collector.index, "closed": collector.closed,
                          "expected_token": collector.expected_token},
            "pending": [asdict(record) for record in self.pending], "emitted_count": self.emitted_count,
        })

    def _restore(self, encoded):
        if set(encoded) != {"manifest", "manifest_sha256", "environment", "noise", "kernel",
                            "events", "collector", "pending", "emitted_count"}:
            raise ValueError("session checkpoint fields differ")
        if (encoded["manifest"] != self.manifest or encoded["manifest_sha256"] != self.manifest_sha256
                or state_digest(encoded["manifest"]) != self.manifest_sha256):
            raise ValueError("session manifest mismatch")
        state_digest(encoded)
        state = decode_arrays(encoded)
        collector = self.collector
        collector.env.load_state_dict(state["environment"])
        collector.producer.check_environment(collector.env)
        if state_digest(encode_arrays(collector.env.state_dict())) != state_digest(encoded["environment"]):
            raise ValueError("environment snapshot did not round trip exactly")
        self.learner.load_state_dict(state["kernel"])
        cursor = state["collector"]
        index, closed = cursor["index"], cursor["closed"]
        if (set(cursor) != {"index", "closed", "expected_token"}
                or type(index) is not int or not 0 <= index <= collector.max_steps
                or type(closed) is not bool or len(state["events"]) != index
                or type(collector.env.t) is not int or collector.env.t != index
                or closed != bool(index >= min(collector.max_steps, collector.env.config.episode_horizon))
                or cursor["expected_token"] != evidence_digest(collector.env.state_dict())):
            raise ValueError("collector cursor/environment/closure mismatch")
        records = [restore_record(event["receipt"]["record"]) for event in state["events"]]
        token = self.manifest["initial_environment_sha256"]
        expected_noise = OUNoise(self.noise.action_dim, seed=self.manifest["noise_seed"],
                                 **self.manifest["noise_options"])
        expected_updates = 0
        for i, (record, event) in enumerate(zip(records, state["events"])):
            if (record.source_id != collector.source_id or record.trajectory_id != collector.trajectory_id
                    or record.step_index != i or record.state_token != token or record.origin != "trajectory"
                    or event["receipt"]["clone_verified"] is not True
                    or record.raw_reward != -event["receipt"]["execution"]["cost"]):
                raise ValueError("receipt lineage/reward mismatch")
            batch = self.learner._prepare([(record,)])
            _, anchor = self.learner.agent._parts(batch.current_actor)
            proposal = np.asarray(event["receipt"]["proposal"], dtype=np.float32)
            gate = np.asarray(event["receipt"]["gate"], dtype=np.float32)
            action = np.asarray(record.action, dtype=np.float32)
            if (proposal.shape != action.shape or gate.shape != action.shape
                    or np.any(np.abs(proposal) > 1) or np.any((gate < 0) | (gate > 1))
                    or not np.array_equal(action, np.clip(anchor[0].numpy() + gate * (proposal - anchor[0].numpy()), -1, 1))):
                raise ValueError("receipt action/proposal/gate mismatch")
            if not np.array_equal(expected_noise.sample(), np.asarray(event["noise"], dtype=np.float32)):
                raise ValueError("exploration draw sequence mismatch")
            step_closed = bool(record.terminated or record.truncated)
            expected_terminal = bool(i + 1 >= collector.env.config.episode_horizon
                                     and collector.horizon_end == "terminal")
            if record.terminated != expected_terminal or step_closed != bool(closed and i == index - 1):
                raise ValueError("record terminal/truncation mapping mismatch")
            prefix_windows, _ = scheduled_windows(records[:i + 1], self.learner.contract.replay,
                                                    self.return_steps, step_closed)
            should_update = (self.learner.mode == "online"
                             and len(prefix_windows) >= self.learner.settings.batch_size)
            if (event["update"] is not None) != should_update:
                raise ValueError("update schedule mismatch")
            expected_updates += int(should_update)
            if should_update and event["update"]["update"] != expected_updates:
                raise ValueError("update trace counter mismatch")
            token = record.next_state_token
        if token != cursor["expected_token"] or expected_updates != self.learner.total_updates:
            raise ValueError("last receipt/kernel count mismatch")
        observation, anchor = collector.producer.observe(collector.env.observation())
        endpoint = tuple(pack_actor_state(observation, anchor, self.learner.contract)[0].tolist())
        if records and records[-1].next_state != endpoint:
            raise ValueError("last observation differs from restored environment")
        if state_digest(encode_arrays(expected_noise.state_dict())) != state_digest(encoded["noise"]):
            raise ValueError("OU state/RNG does not match the saved draw history")
        self.noise.load_state_dict(state["noise"])
        windows, pending = scheduled_windows(records, self.learner.contract.replay, self.return_steps, closed)
        if (type(state["emitted_count"]) is not int or state["emitted_count"] != len(windows)
                or state["pending"] != [asdict(record) for record in pending]):
            raise ValueError("pending tail or published window count mismatch")
        ring, position = [], 0
        for window in windows:
            if len(ring) < self.learner.settings.replay_capacity:
                ring.append(window)
            else:
                ring[position] = window
            position = (position + 1) % self.learner.settings.replay_capacity
        if ring != self.learner.windows or position != self.learner.position:
            raise ValueError("kernel replay lost or duplicated a collected window")
        if state["events"] and state["events"][-1]["kernel_state_sha256"] != state_digest(self.learner.state_dict()):
            raise ValueError("last event/kernel state mismatch")
        collector.index, collector.closed, collector.expected_token = index, closed, token
        self.events, self.pending, self.emitted_count = state["events"], pending, len(windows)

    def load_state_dict(self, state):
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(state))
        self.__dict__.update(candidate.__dict__)

    def save(self, path):
        path = Path(path)
        if path.exists():
            raise FileExistsError(path)
        state = self.state_dict()
        payload = {"state": state, "state_sha256": state_digest(state)}
        path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(prefix=".session-", suffix=".tmp", dir=path.parent)
        os.close(descriptor)
        try:
            torch.save(payload, temporary)
            os.link(temporary, path)
        finally:
            os.unlink(temporary)

    def load(self, path):
        envelope = torch.load(path, map_location="cpu", weights_only=True)
        if set(envelope) != {"state", "state_sha256"} or state_digest(envelope["state"]) != envelope["state_sha256"]:
            raise ValueError("session payload checksum mismatch")
        self.load_state_dict(envelope["state"])
