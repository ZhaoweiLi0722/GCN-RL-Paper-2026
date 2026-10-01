"""Explicitly enabled real-environment collector; no learner or legacy migration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from src.baselines.heuristics import facility_net_action_from_state, heuristic_settings_for_policy
from src.env.patient_capacity_planning import PatientConditionCapacityEnv
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.networks import torch, require_torch
from src.rl.prospective_adapter import ReplayInputContract, pack_actor_state
from src.rl.validated_returns import OneStepRecord, ReplaySemantics


def json_value(value):
    """Strict JSON conversion for current execution receipts, not checkpoint loading."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"unsupported receipt type: {type(value).__name__}")


def evidence_digest(value):
    payload = json.dumps(value, default=json_value, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _edge_sets(env):
    return tuple(tuple(sorted(tuple(sorted(edge)) for edge in getattr(env, key)))
                 for key in ("specimen_edges", "resource_edges", "capacity_edges", "information_edges"))


class PatientObservationProducer:
    """Lossless raw-observation layout plus a causal, state-derived MDL-2 anchor.

    Supports the basic four-group facility-net action only. Full environment
    snapshots are never accepted by observe(), so hidden state cannot be passed
    to the policy by the collector's metadata path.
    """

    def __init__(self, env, *, enabled=False, gamma, reward_scale, message_graph="shared_relations"):
        require_torch()
        if enabled is not True:
            raise ValueError("prospective collection must be explicitly enabled")
        if type(env) is not PatientConditionCapacityEnv:
            raise TypeError("only the declared patient environment is supported")
        base = env.config
        if message_graph not in ("shared_relations", "specimen_routes"):
            raise ValueError("explicit supported message graph required")
        if base.action_mode != "facility_net":
            raise ValueError("facility_net required")
        if (base.enable_overtime_control or (base.include_central_capacity_hub and message_graph == "shared_relations")
                or base.include_on_order_state or base.enable_stochastic_procurement
                or base.reagent_purchase_lead_time):
            raise ValueError("overtime, hub and procurement layouts are not supported")
        edges = _edge_sets(env)
        if message_graph == "shared_relations" and any(relation != edges[0] for relation in edges[1:]):
            raise ValueError("heterogeneous relations cannot be collapsed into one physical graph")
        self.edges, self.message_graph = edges, message_graph
        self.config_digest = evidence_digest(asdict(env.env_config))
        self.anchor_config = asdict(base)
        patient = asdict(env.env_config)
        patient.pop("base")
        self.anchor_config.update(patient, env_type="patient_condition")
        self.settings = heuristic_settings_for_policy("mdl2")
        n = base.num_facilities
        names = ["demand", "waiting_specimens", "reagents"]
        names += [f"bioreactor_slot_{i}" for i in range(base.production_lead_time)]
        if base.include_supplier_state:
            names += ["supplier_available"]
        if base.include_demand_forecast_state:
            names += ["demand_forecast"]
        if base.include_transfer_pipeline_state:
            names += ["pending_specimens", "pending_reagents", "pending_capacity"]
        if base.include_demand_history_state:
            names += ["demand_history_mean", "demand_trend", "forecast_error_mean"]
        if base.include_demand_sequence_state:
            names += [f"{kind}_{i}" for kind in ("demand_sequence", "error_sequence", "sequence_mask")
                      for i in range(base.demand_sequence_length)]
        self.base_width = len(names)
        if self.base_width != env.features_per_facility:
            raise ValueError("unsupported base observation width")
        names += ["waiting_count", "waiting_mean_survival", "near_expiry_count",
                  "production_count", "production_mean_survival", "production_at_risk_count"]
        names += [f"waiting_survival_bucket_{i}" for i in range(len(patient["survival_bucket_edges"]) + 1)]
        if env.env_config.include_specimen_routing_state:
            names += ["in_transit_count", "in_transit_mean_survival",
                      "transferred_waiting_count", "route_eligible_waiting_count"]
        self.summary_width = len(names) - self.base_width
        if self.summary_width != env.summary_width or env.action_size != 4 * n:
            raise ValueError("unsupported patient/action layout")
        definition_data = {"env": self.config_digest, "edges": edges, "anchor": asdict(self.settings)}
        if message_graph == "specimen_routes":
            # A declared projection, not a union or a change to physical transfers.
            # Raw observations have facility nodes only, even for hub-aware R4.
            definition_data["message_graph"] = message_graph
        definition = "patient-raw-request-v1/" + evidence_digest(definition_data)
        schema = InputSchema(definition, tuple(f"facility_{i}" for i in range(n)), tuple(names),
                             ("normalized_time",) if base.include_time_state else (),
                             tuple(f"{group}_{i}" for group in
                                   ("specimen_net", "reagent_net", "capacity_net", "replenishment")
                                   for i in range(n)))
        self.raw_width = env.observation_size
        self.links = np.zeros((n, n), dtype=np.float32)
        for a, b in edges[0]:
            self.links[a, b] = self.links[b, a] = 1.
        self.contract = ReplayInputContract(schema, ReplaySemantics(
            reward_kind="absolute_environment", reward_definition_id="patient-negative-total-cost/" + self.config_digest,
            reward_scale=reward_scale, gamma=gamma, state_schema_id=definition + "/actor-flat",
            action_schema_id=definition + "/action", state_dim=self.raw_width + n * n + 4 * n,
            action_dim=4 * n, bootstrap_on_truncation=True))

    def check_environment(self, env):
        if (type(env) is not PatientConditionCapacityEnv
                or evidence_digest(asdict(env.env_config)) != self.config_digest
                or env.config != env.env_config.base
                or _edge_sets(env) != self.edges):
            raise ValueError("environment config/topology changed after contract creation")

    def observe(self, raw):
        if not isinstance(raw, np.ndarray) or raw.dtype != np.float32:
            raise TypeError("raw observation must be the environment float32 array")
        if raw.shape != (self.raw_width,) or not np.isfinite(raw).all():
            raise ValueError("raw observation width or finiteness mismatch")
        schema = self.contract.inputs
        n = len(schema.node_ids)
        split = n * self.base_width
        end = split + n * self.summary_width
        nodes = np.concatenate((raw[:split].reshape(n, self.base_width),
                                raw[split:end].reshape(n, self.summary_width)), axis=1)
        observation = ObservationBatch(schema, torch.tensor(nodes[None]),
                                       torch.tensor(raw[None, end:]), torch.tensor(self.links[None]))
        anchor_np = facility_net_action_from_state(raw, self.anchor_config, settings=self.settings)
        if anchor_np.shape != (len(schema.action_names),) or not np.isfinite(anchor_np).all():
            raise ValueError("invalid MDL-2 anchor")
        if np.any(np.abs(anchor_np) > 1):
            raise ValueError("anchor outside normalized action coordinates")
        return observation, torch.tensor(anchor_np[None], dtype=torch.float32)

    def unpack_raw(self, observation):
        if observation.schema != self.contract.inputs:
            raise ValueError("observation schema mismatch")
        nodes = observation.nodes.detach().cpu().numpy()
        return np.concatenate((nodes[:, :, :self.base_width].reshape(nodes.shape[0], -1),
                               nodes[:, :, self.base_width:].reshape(nodes.shape[0], -1),
                               observation.globals.detach().cpu().numpy()), axis=1)


@dataclass(frozen=True)
class CollectedStep:
    record: OneStepRecord
    proposal: tuple[float, ...]
    gate: tuple[float, ...]
    execution: dict
    clone_verified: bool


class PatientReplayCollector:
    """Own one fresh, closed segment; reject drift/reset rather than guessing resume.

    The env's horizon-only done is mapped according to an explicit caller choice.
    There is intentionally no reset/resume method or optimizer/update callback.
    """

    def __init__(self, env, producer, *, enabled=False, trajectory_id, max_steps,
                 horizon_end, verify_clone=True):
        if enabled is not True:
            raise ValueError("prospective collection must be explicitly enabled")
        if type(max_steps) is not int or max_steps < 1:
            raise ValueError("positive max_steps required")
        if horizon_end not in ("terminal", "truncation"):
            raise ValueError("explicit horizon_end mapping required")
        if not isinstance(trajectory_id, str) or not trajectory_id.strip():
            raise ValueError("trajectory_id required")
        if type(verify_clone) is not bool:
            raise ValueError("verify_clone must be boolean")
        producer.check_environment(env)
        if env.t != 0:
            raise ValueError("collector requires a fresh episode; resume not supported")
        self.env, self.producer = env, producer
        self.max_steps, self.horizon_end = max_steps, horizon_end
        self.trajectory_id = trajectory_id
        self.source_id = "live-patient-collector-v1/" + producer.config_digest
        self.index, self.closed = 0, False
        self.expected_token = evidence_digest(env.state_dict())
        self.clone = PatientConditionCapacityEnv(env.env_config, seed=0) if verify_clone else None

    def step(self, decision):
        if self.closed:
            raise ValueError("segment is closed")
        self.producer.check_environment(self.env)
        before = self.env.state_dict()
        token = evidence_digest(before)
        if token != self.expected_token or self.env.t != self.index:
            raise ValueError("environment state/RNG drift or reset between records")
        size = self.producer.contract.replay.action_dim
        arrays = []
        for name in ("proposal", "gate", "request"):
            array = np.asarray(getattr(decision, name), dtype=np.float32)
            if array.shape != (size,) or not np.isfinite(array).all():
                raise ValueError(f"invalid decision {name}")
            bound = (array < 0) | (array > 1) if name == "gate" else np.abs(array) > 1
            if bound.any():
                raise ValueError(f"decision {name} out of bounds")
            arrays.append(array.copy())
        proposal, gate, request = arrays
        raw = self.env.observation()
        observation, anchor = self.producer.observe(raw)
        if not np.array_equal(self.producer.unpack_raw(observation)[0], raw):
            raise ValueError("raw observation round trip failed")
        expected_request = np.clip(anchor[0].numpy() + gate * (proposal - anchor[0].numpy()), -1, 1)
        if not np.array_equal(request, expected_request):
            raise ValueError("request is not the declared post-gate proposal")
        state = tuple(pack_actor_state(observation, anchor, self.producer.contract)[0].tolist())
        next_raw, reward, done, info = self.env.step(request)
        self.closed = bool(done or self.index + 1 >= self.max_steps)
        following = self.env.state_dict()
        next_token = evidence_digest(following)
        if self.clone is not None:
            self.clone.load_state_dict(before)
            clone_raw, clone_reward, clone_done, clone_info = self.clone.step(request)
            if (not np.array_equal(next_raw, clone_raw) or reward != clone_reward or done != clone_done
                    or evidence_digest(info) != evidence_digest(clone_info)
                    or next_token != evidence_digest(self.clone.state_dict())):
                self.closed = True
                raise ValueError("cloned real step did not reproduce exactly")
        next_observation, next_anchor = self.producer.observe(next_raw)
        next_state = tuple(pack_actor_state(next_observation, next_anchor, self.producer.contract)[0].tolist())
        terminated = bool(done and self.horizon_end == "terminal")
        truncated = bool(self.closed and not terminated)
        record = OneStepRecord(self.producer.contract.replay, self.source_id, "trajectory",
                               self.trajectory_id, self.index, token, next_token, state,
                               tuple(request.tolist()), float(reward), next_state, terminated, truncated)
        self.index += 1
        self.expected_token = next_token
        self.env.assert_identity_conservation()
        return CollectedStep(record, tuple(proposal.tolist()), tuple(gate.tolist()), info,
                             self.clone is not None)
