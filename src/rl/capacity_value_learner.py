"""Frozen-behavior eight-step TD evaluation, followed by MPC policy improvement."""

import copy
import hashlib
import io

import numpy as np

from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.networks import torch


def td_rows(features, heuristics, costs, *, horizon=8, scale=1000000.):
    if type(horizon) is not int or not 1 <= horizon <= 64 or not np.isfinite(scale) or scale <= 0:
        raise ValueError("finite positive scale and bounded integer TD horizon required")
    if len(features) != 65 or len(heuristics) != 65 or len(costs) != 64:
        raise ValueError("one complete 64-step settled cohort required")
    x = np.asarray(features, dtype=np.float32)
    h, c = np.asarray(heuristics, dtype=np.float64), np.asarray(costs, dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(h).all() or not np.isfinite(c).all() or (c < 0).any():
        raise ValueError("nonfinite/negative recorded cost")
    ends = np.minimum(np.arange(64)+horizon, 64)
    sums = np.asarray([np.sum(c[t:end], dtype=np.float64) for t, end in enumerate(ends)])
    return dict(states=x[:-1], next_states=x[ends], base=h[:-1]/scale,
                bootstrap_base=h[ends]/scale, observed_cost=sums/scale, done=ends == 64)


class CapacityValueLearner:
    format = "capacity-terminal-value-td-v1"

    def __init__(self, config, *, seed, feature_dim, before_forward, before_optimizer):
        self.config = copy.deepcopy(config)
        self.feature_dim = feature_dim
        self.before_forward, self.before_optimizer = before_forward, before_optimizer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            self.model = CapacityTerminalValue(feature_dim, width=config["width"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], foreach=False)
        self.rng = np.random.default_rng(int(seed)+1)
        self.updates = 0
        self.counts = dict(forwards=0, optimizer_steps=0)
        self.pending = None

    def _forward(self, x):
        self.before_forward("value", len(x))
        self.counts["forwards"] += 1
        return self.model(torch.as_tensor(x, dtype=torch.float32))

    def residuals(self, x):
        with torch.no_grad():
            values = self._forward(x).numpy().astype(np.float64)
        if not np.isfinite(values).all():
            raise ValueError("nonfinite terminal value")
        return values*self.config["cost_scale"]

    def admit_episode(self, rows):
        if self.pending is not None:
            raise RuntimeError("previous cohort update boundary incomplete")
        with torch.no_grad():
            future = self._forward(rows["next_states"]).numpy().astype(np.float64)
        target = rows["observed_cost"] + np.where(rows["done"], 0., rows["bootstrap_base"]+future) - rows["base"]
        if not np.isfinite(target).all():
            raise ValueError("nonfinite eight-step TD targets")
        converted = target.astype(np.float32)
        if not np.isfinite(converted).all():
            raise ValueError("TD targets overflow float32")
        self.pending = dict(states=rows["states"].copy(), targets=converted, completed=0)

    def update(self):
        if self.pending is None or self.pending["completed"] >= self.config["updates_per_world"]:
            raise RuntimeError("no pending bounded value update")
        prior = self.state_dict()
        try:
            indices = self.rng.integers(0, 64, size=self.config["batch_size"])
            prediction = self._forward(self.pending["states"][indices])
            target = torch.from_numpy(self.pending["targets"][indices])
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite TD loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config["gradient_norm_cap"], error_if_nonfinite=True)
            self.before_optimizer("value", len(indices))
            self.counts["optimizer_steps"] += 1
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("nonfinite value parameters")
            self.updates += 1
            self.pending["completed"] += 1
            receipt = dict(loss=float(loss.detach()), gradient_norm=float(norm), update=self.updates,
                           cohort_update=self.pending["completed"], indices=indices.tolist())
            if self.pending["completed"] == self.config["updates_per_world"]:
                self.pending = None
            return receipt
        except BaseException:
            charged = copy.deepcopy(self.counts)
            self.load_state_dict(prior)
            self.counts = charged
            raise

    def state_dict(self):
        return copy.deepcopy(dict(format=self.format, config=self.config, feature_dim=self.feature_dim,
            model=self.model.state_dict(), optimizer=self.optimizer.state_dict(),
            rng=self.rng.bit_generator.state, updates=self.updates, counts=self.counts, pending=self.pending))

    def load_state_dict(self, state):
        if (state.get("format") != self.format or state["config"] != self.config
                or state["feature_dim"] != self.feature_dim):
            raise ValueError("value snapshot metadata mismatch")
        updates, counts, pending = state["updates"], copy.deepcopy(state["counts"]), copy.deepcopy(state["pending"])
        if (type(updates) is not int or updates < 0 or set(counts) != {"forwards", "optimizer_steps"}
                or any(type(v) is not int or v < 0 for v in counts.values())
                or counts["optimizer_steps"] < updates):
            raise ValueError("invalid value snapshot counters")
        if pending is not None:
            if (set(pending) != {"states", "targets", "completed"}
                    or np.asarray(pending["states"]).shape != (64, 4, self.feature_dim)
                    or np.asarray(pending["targets"]).shape != (64,)
                    or not np.isfinite(pending["states"]).all() or not np.isfinite(pending["targets"]).all()
                    or type(pending["completed"]) is not int
                    or not 0 <= pending["completed"] < self.config["updates_per_world"]):
                raise ValueError("invalid pending value update boundary")
        candidate = copy.deepcopy(self.model)
        candidate.load_state_dict(state["model"], strict=True)
        if any(not torch.isfinite(p).all() for p in candidate.parameters()):
            raise ValueError("nonfinite value snapshot")
        optimizer = torch.optim.Adam(candidate.parameters(), lr=self.config["lr"], foreach=False)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        if any(isinstance(v, torch.Tensor) and not torch.isfinite(v).all()
               for entry in optimizer.state.values() for v in entry.values()):
            raise ValueError("nonfinite value optimizer snapshot")
        rng = np.random.default_rng()
        rng.bit_generator.state = copy.deepcopy(state["rng"])
        self.model, self.optimizer, self.rng = candidate, optimizer, rng
        self.updates, self.counts, self.pending = updates, counts, pending

    def snapshot(self):
        buffer = io.BytesIO()
        torch.save(self.state_dict(), buffer)
        raw = buffer.getvalue()
        return raw, hashlib.sha256(raw).hexdigest()

    @classmethod
    def from_bytes(cls, raw, *, before_forward, before_optimizer):
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        out = cls(state["config"], seed=0, feature_dim=state["feature_dim"],
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.load_state_dict(state)
        return out
