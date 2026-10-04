"""Versioned weight-only forks and bounded variable-tail value updates."""

import copy
import hashlib
import io

import numpy as np

from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_learner import td_rows as observed_td_rows
from src.rl.capacity_planner_tail_targets import mc_rows, td_rows
from src.rl.networks import torch


class CapacityPlannerTailLearner(CapacityValueComparisonLearner):
    format = "capacity-planner-tail-learner-v1"

    def __init__(self, config, *, ancestor_sha256, **kwargs):
        if (config.get("method") not in ("observed_td", "planner_tail_td", "planner_tail_mc")
                or config.get("architecture") != "graph" or config.get("batch_size") != 64
                or config.get("updates_per_world") != 32 or config.get("max_new_updates") != 768
                or any(config.get(k) != v for k, v in dict(width=32, lr=.0003,
                    gradient_norm_cap=5., cost_scale=1000000., td_horizon=8, gamma=1.).items())
                or not isinstance(ancestor_sha256, str) or len(ancestor_sha256) != 64
                or any(c not in "0123456789abcdef" for c in ancestor_sha256)):
            raise ValueError("explicit method, fixed update cap and ancestor hash required")
        super().__init__(config, **kwargs)
        self.ancestor_sha256 = ancestor_sha256

    @classmethod
    def fork_weights(cls, raw, config, *, seed, expected_sha256, before_forward, before_optimizer):
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError("ancestor bytes do not match declared model")
        historical = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if (historical.get("format") != "capacity-value-comparison-td-v1"
                or historical["config"].get("architecture") != "graph"
                or historical["config"]["width"] != config["width"]
                or historical["feature_dim"] != 31 or historical["updates"] != 1536
                or historical["pending"] is not None):
            raise ValueError("only declared completed graph ancestors may fork")
        out = cls(config, seed=seed, feature_dim=31, ancestor_sha256=expected_sha256,
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.model.load_state_dict(historical["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in out.model.parameters()):
            raise ValueError("nonfinite ancestor model")
        # Do not import Adam moments, RNG, pending targets or historical counters.
        return out

    def _admit(self, states, targets, source_hashes):
        x, y = np.array(states, dtype=np.float32, copy=True), np.array(targets, dtype=np.float64, copy=True)
        if (self.pending is not None or self.updates + 32 > self.config["max_new_updates"]
                or x.ndim != 3 or x.shape[1:] != (4, 31) or not 1 <= len(x) <= 4224
                or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all()
                or not source_hashes or any(not isinstance(s, str) or len(s) != 64 for s in source_hashes)):
            raise ValueError("invalid complete frozen-target admission")
        self.pending = dict(states=x, targets=y, completed=0, source_hashes=list(source_hashes))

    def admit_observed(self, data):
        if self.config["method"] != "observed_td" or self.pending is not None or self.updates + 32 > 768:
            raise ValueError("wrong method or pending data")
        rows = observed_td_rows(**data, horizon=8, scale=1000000.)
        with torch.no_grad():
            future = self._forward(rows["next_states"]).numpy().astype(np.float64)
        target = rows["observed_cost"] + np.where(rows["done"], 0., rows["bootstrap_base"] + future) - rows["base"]
        digest = hashlib.sha256()
        for key in ("features", "heuristics", "costs"):
            digest.update(np.asarray(data[key], dtype="<f8").tobytes())
        self._admit(rows["states"], target, [digest.hexdigest()])

    def admit_tails(self, records):
        method = self.config["method"]
        if (method not in ("planner_tail_td", "planner_tail_mc") or self.pending is not None
                or self.updates + 32 > 768 or len(records) != 96):
            raise ValueError("exactly one shared 96-tail reference world required")
        validated = [mc_rows(r["features"], r["heuristics"], r["costs"], epochs=r["epochs"]) for r in records]
        states = np.concatenate([r["states"] for r in validated])
        if len(states) > 4224:
            raise ValueError("tail world exceeds frozen row cap")
        hashes = [r["metadata"]["source_data_sha256"] for r in validated]
        if method == "planner_tail_mc":
            targets = np.concatenate([r["targets"] for r in validated])
        else:
            # Freeze all cohort bootstraps before any update, including short tails.
            values = np.concatenate([self.residuals(states[i:i+64]) for i in range(0, len(states), 64)])
            outputs, offset = [], 0
            for raw, row in zip(records, validated):
                length = len(row["states"])
                residual = np.r_[values[offset:offset+length], 0.]
                output = td_rows(raw["features"], raw["heuristics"], raw["costs"],
                                 epochs=raw["epochs"], frozen_residuals=residual)
                outputs.append(output["targets"])
                offset += length
            targets = np.concatenate(outputs)
        self._admit(states, targets, hashes)

    def update(self):
        if self.pending is None or self.pending["completed"] >= 32 or self.updates >= 768:
            raise RuntimeError("no admitted bounded update")
        prior = self.state_dict()
        try:
            indices = self.rng.integers(0, len(self.pending["states"]), size=64)
            prediction = self._forward(self.pending["states"][indices]).to(torch.float64)
            target = torch.from_numpy(self.pending["targets"][indices])
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite value loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_norm_cap"], error_if_nonfinite=True)
            self.before_optimizer("value", 64)
            self.counts["optimizer_steps"] += 1
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("nonfinite updated parameters")
            self.updates += 1
            self.pending["completed"] += 1
            receipt = dict(loss=float(loss.detach()), gradient_norm=float(norm), update=self.updates,
                           cohort_update=self.pending["completed"], indices=indices.tolist(),
                           method=self.config["method"], ancestor_sha256=self.ancestor_sha256)
            if self.pending["completed"] == 32:
                self.pending = None
            return receipt
        except BaseException:
            charged = copy.deepcopy(self.counts)
            self.load_state_dict(prior)
            self.counts = charged
            raise

    def state_dict(self):
        state = super().state_dict()
        state["ancestor_sha256"] = self.ancestor_sha256
        return state

    def load_state_dict(self, state):
        if (state.get("format") != self.format or state["config"] != self.config
                or state["feature_dim"] != 31 or state.get("ancestor_sha256") != self.ancestor_sha256):
            raise ValueError("learner/method/ancestor mismatch")
        updates, counts, pending = state["updates"], copy.deepcopy(state["counts"]), copy.deepcopy(state["pending"])
        if (type(updates) is not int or not 0 <= updates <= 768
                or set(counts) != {"forwards", "optimizer_steps"}
                or any(type(v) is not int or v < 0 for v in counts.values())
                or counts["optimizer_steps"] < updates):
            raise ValueError("invalid update counters")
        if pending is not None:
            x, y = np.asarray(pending["states"]), np.asarray(pending["targets"])
            if (set(pending) != {"states", "targets", "completed", "source_hashes"}
                    or x.ndim != 3 or x.shape[1:] != (4,31) or not 1 <= len(x) <= 4224
                    or y.shape != (len(x),) or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or type(pending["completed"]) is not int or not 0 <= pending["completed"] < 32
                    or updates % 32 != pending["completed"] or updates >= 768
                    or not pending["source_hashes"]
                    or any(not isinstance(s, str) or len(s) != 64 for s in pending["source_hashes"])):
                raise ValueError("invalid pending target boundary")
        elif updates % 32:
            raise ValueError("incomplete fit lacks pending targets")
        candidate = copy.deepcopy(self.model)
        candidate.load_state_dict(state["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in candidate.parameters()):
            raise ValueError("invalid model state")
        optimizer = torch.optim.Adam(candidate.parameters(), lr=self.config["lr"], foreach=False)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
               for entry in optimizer.state.values() for v in entry.values()):
            raise ValueError("invalid optimizer state")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.model, self.optimizer, self.rng = candidate, optimizer, rng
        self.updates, self.counts, self.pending = updates, counts, pending

    @classmethod
    def from_bytes(cls, raw, *, before_forward, before_optimizer):
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        out = cls(state["config"], seed=0, feature_dim=31, ancestor_sha256=state["ancestor_sha256"],
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.load_state_dict(state)
        return out
