"""Fresh matched observed-TD warmup and corresponding native TD8/MC forks.

Warm config defaults to method=observed_td and requires max_new_updates=1536; tail config uses
planner_tail_td/planner_tail_mc and max_new_updates=768, with the existing native
tail schema, two branches and frozen_mpc policy. Both require architecture=graph
or self_only and the unchanged width/lr/batch/scale/horizon/gradient settings.

The runner owns shared plainH8 warm data, common frozen graph-parent collection,
paired seeds, and execution authorization. Hashes bind declarations and bytes,
not proof of collection history. Only trusted local snapshot bytes may be read.
"""

import copy
import hashlib
import io
import json

import numpy as np

from src.models.capacity_confirmation_value import CapacityConfirmationValue
from src.rl.capacity_native_tail_targets import (
    TAIL_SCHEMA, TAILS_PER_WORLD, mc_rows, td_rows, validate_metadata, validate_records,
)
from src.rl.capacity_value_learner import CapacityValueLearner, td_rows as observed_td_rows
from src.rl.networks import torch


def _sha(value):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("lowercase SHA256 required")
    return value


def _array(value, dtype, shape, name):
    if (type(value) is not np.ndarray or value.dtype != np.dtype(dtype)
            or value.shape != shape or not np.isfinite(value).all()):
        raise ValueError(f"{name} must be finite {dtype} with shape {shape}")
    return np.array(value, dtype=dtype, order="C", copy=True)


def _digest(header, arrays):
    digest = hashlib.sha256(b"capacity-confirmation-v1\0")
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii") + b"\0")
    for name, value in arrays.items():
        digest.update(name.encode("ascii") + b"\0" + value.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
        digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


def _configuration(config, *, tail):
    if not isinstance(config, dict):
        raise ValueError("explicit confirmation configuration required")
    config = copy.deepcopy(config)
    if not tail:
        config.setdefault("method", "observed_td")
    fixed = dict(width=32, lr=.0003, gradient_norm_cap=5., cost_scale=1000000.,
                 td_horizon=8, gamma=1., batch_size=64, updates_per_world=32,
                 max_new_updates=768 if tail else 1536)
    if (config.get("architecture") not in ("graph", "self_only")
            or config.get("method") not in (("planner_tail_td", "planner_tail_mc")
                                            if tail else ("observed_td",))
            or any(type(config.get(k)) not in (int, float) or config[k] != v
                   or isinstance(v, int) and type(config[k]) is not int
                   for k, v in fixed.items())):
        raise ValueError("explicit matched architecture, method and frozen hyperparameters required")
    if tail:
        if (config.get("tail_policy") != "frozen_mpc" or config.get("tail_schema") != TAIL_SCHEMA
                or type(config.get("tails_per_world")) is not int
                or config["tails_per_world"] != TAILS_PER_WORLD):
            raise ValueError("two native frozen_mpc branches required")
        _sha(config.get("continuation_sha256"))
    elif any(k in config for k in ("tail_policy", "tail_schema", "tails_per_world", "continuation_sha256")):
        raise ValueError("fresh warmup cannot declare a tail continuation")
    json.dumps(config, sort_keys=True, allow_nan=False)
    return config


class _ConfirmationLearner(CapacityValueLearner):
    """Shared bounded float64 loss, scoring and transactional checkpoint code."""

    def __init__(self, config, *, seed, feature_dim=31, before_forward, before_optimizer,
                 ancestor_sha256=None):
        tail = self._is_tail
        config = _configuration(config, tail=tail)
        if (type(feature_dim) is not int or feature_dim != 31
                or type(seed) is not int or seed < 0
                or not callable(before_forward) or not callable(before_optimizer)):
            raise ValueError("31 features, nonnegative integer seed and callbacks required")
        if tail:
            _sha(ancestor_sha256)
        elif ancestor_sha256 is not None:
            raise ValueError("warmup must start fresh")
        self.config, self.feature_dim, self.seed = config, feature_dim, seed
        self.ancestor_sha256 = ancestor_sha256
        self.continuation_sha256 = config.get("continuation_sha256")
        self.before_forward, self.before_optimizer = before_forward, before_optimizer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = CapacityConfirmationValue(feature_dim, architecture=config["architecture"],
                                                   width=config["width"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], foreach=False)
        self.rng = np.random.default_rng(seed + 1)
        self.updates = 0
        self.counts = dict(forwards=0, optimizer_steps=0)
        self.pending = None
        self.admissions = []

    def _forward(self, x):
        if type(x) is not np.ndarray or x.ndim != 3 or not len(x):
            raise ValueError("nonempty public feature batch required")
        x = _array(x, "float32", (len(x), 4, 31), "features")
        self.before_forward("value", len(x))
        self.counts["forwards"] += 1
        values = self.model(torch.from_numpy(x))
        if (not isinstance(values, torch.Tensor) or values.shape != (len(x),)
                or values.device.type != "cpu" or values.dtype != torch.float32
                or not torch.isfinite(values).all()):
            raise ValueError("one finite float32 value per state required")
        return values

    def residuals(self, x):
        """Return raw-cost residuals, not normalized network values."""
        with torch.no_grad():
            values = self._forward(x).numpy().astype(np.float64)
        if values.shape != (len(x),) or not np.isfinite(values).all():
            raise ValueError("one finite residual per state required")
        values *= self.config["cost_scale"]
        if not np.isfinite(values).all():
            raise ValueError("residual cost overflow")
        return values

    def score_endpoints(self, features, heuristics, *, terminal=None):
        """Return H + residual in raw costs; terminal endpoints contribute zero.

        The caller adds prefix costs and applies the unchanged MPC aggregation.
        Terminal rows are validated but never sent through the model.
        """
        if type(features) is not np.ndarray or features.ndim != 3 or not len(features):
            raise ValueError("nonempty endpoint features required")
        n = len(features)
        x = _array(features, "float32", (n, 4, 31), "features")
        h = _array(heuristics, "float64", (n,), "heuristics")
        done = (np.zeros(n, dtype=bool) if terminal is None
                else _array(terminal, "bool", (n,), "terminal"))
        result = np.zeros(n, dtype=np.float64)
        if (~done).any():
            result[~done] = h[~done] + self.residuals(x[~done])
        if not np.isfinite(result).all():
            raise ValueError("nonfinite endpoint scores")
        return result

    def _available(self):
        if (self.pending is not None or self.updates % 32
                or self.updates + 32 > self.config["max_new_updates"]):
            raise ValueError("pending targets or exhausted world allowance")

    def _pending_hash(self, pending, binding):
        header = dict(format=self.format, config=self.config, seed=self.seed,
                      ancestor_sha256=self.ancestor_sha256,
                      continuation_sha256=self.continuation_sha256,
                      admission={k: v for k, v in binding.items() if k != "pending_sha256"})
        return _digest(header, {k: pending[k] for k in ("states", "targets")})

    def _admit(self, states, targets, hashes, *, world_index, records):
        self._available()
        x = _array(states, "float32", (len(states), 4, 31), "states")
        y = _array(targets, "float64", (len(states),), "targets")
        pending = dict(states=x, targets=y, completed=0, source_hashes=list(hashes))
        binding = dict(world_index=world_index, source_hashes=list(hashes), records=copy.deepcopy(records))
        binding["pending_sha256"] = self._pending_hash(pending, binding)
        self.pending = pending
        self.admissions.append(binding)

    def update(self):
        """One bounded Adam update; failed attempts restore state but keep charges."""
        if (self.pending is None or self.updates >= self.config["max_new_updates"]
                or type(self.pending.get("completed")) is not int
                or not 0 <= self.pending["completed"] < 32
                or self.pending["completed"] != self.updates % 32):
            raise RuntimeError("no admitted bounded update")
        prior = self.state_dict()
        try:
            indices = self.rng.integers(0, len(self.pending["states"]), size=64)
            prediction = self._forward(self.pending["states"][indices]).to(torch.float64)
            target = torch.from_numpy(self.pending["targets"][indices])
            loss = torch.nn.functional.mse_loss(prediction, target)
            if not torch.isfinite(loss):
                raise ValueError("nonfinite confirmation value loss")
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5., error_if_nonfinite=True)
            self.before_optimizer("value", 64)
            self.counts["optimizer_steps"] += 1
            self.optimizer.step()
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("nonfinite updated value parameters")
            if any(not torch.isfinite(v).all() for entry in self.optimizer.state.values()
                   for v in entry.values() if isinstance(v, torch.Tensor)):
                raise ValueError("nonfinite updated Adam moments")
            self.updates += 1
            self.pending["completed"] += 1
            receipt = dict(loss=float(loss.detach()), gradient_norm=float(norm), update=self.updates,
                           cohort_update=self.pending["completed"], indices=indices.tolist(),
                           method=self.config["method"], architecture=self.config["architecture"],
                           ancestor_sha256=self.ancestor_sha256,
                           continuation_sha256=self.continuation_sha256)
            if self.pending["completed"] == 32:
                self.pending = None
            return receipt
        except BaseException:
            charged = copy.deepcopy(self.counts)
            self.load_state_dict(prior)
            self.counts = charged
            raise

    def state_dict(self):
        return copy.deepcopy(dict(format=self.format, config=self.config, seed=self.seed,
            feature_dim=self.feature_dim, ancestor_sha256=self.ancestor_sha256,
            continuation_sha256=self.continuation_sha256, model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict(), rng=self.rng.bit_generator.state,
            updates=self.updates, counts=self.counts, pending=self.pending, admissions=self.admissions))

    def _validate_admissions(self, state):
        updates, pending, admissions = state["updates"], state["pending"], state["admissions"]
        if (type(admissions) is not list or len(admissions) != updates // 32 + (pending is not None)
                or len(admissions) > self.config["max_new_updates"] // 32):
            raise ValueError("admission count disagrees with update boundary")
        seen = set()
        for index, binding in enumerate(admissions):
            if (type(binding) is not dict
                    or set(binding) != {"world_index", "source_hashes", "records", "pending_sha256"}):
                raise ValueError("invalid admission binding")
            world = binding["world_index"]
            if type(world) is not int or world in seen:
                raise ValueError("duplicate/invalid world index")
            seen.add(world)
            hashes = binding["source_hashes"]
            if type(hashes) is not list or len(hashes) != (2 if self._is_tail else 1):
                raise ValueError("invalid source hashes")
            for value in (*hashes, binding["pending_sha256"]):
                _sha(value)
            if self._is_tail:
                records = validate_metadata(binding["records"], world_index=world,
                                            continuation_sha256=self.continuation_sha256)
                length = sum(64 - r["start_epoch"] for r in records)
            else:
                if world != index or binding["records"] != []:
                    raise ValueError("invalid observed warmup admission")
                length = 64
            if pending is not None and index == len(admissions) - 1:
                if (type(pending) is not dict
                        or set(pending) != {"states", "targets", "completed", "source_hashes"}
                        or type(pending["completed"]) is not int
                        or not 0 <= pending["completed"] < 32
                        or pending["completed"] != updates % 32
                        or updates >= self.config["max_new_updates"]
                        or pending["source_hashes"] != hashes):
                    raise ValueError("invalid pending update boundary")
                _array(pending["states"], "float32", (length, 4, 31), "pending states")
                _array(pending["targets"], "float64", (length,), "pending targets")
                if self._pending_hash(pending, binding) != binding["pending_sha256"]:
                    raise ValueError("pending content/configuration hash mismatch")
        if pending is None and updates % 32:
            raise ValueError("incomplete world lacks pending targets")

    def _restored_optimizer(self, model, saved, updates):
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config["lr"], foreach=False)
        expected = optimizer.state_dict()
        if (type(saved) is not dict or set(saved) != {"state", "param_groups"}
                or saved["param_groups"] != expected["param_groups"]
                or type(saved["state"]) is not dict):
            raise ValueError("optimizer configuration mismatch")
        ids = expected["param_groups"][0]["params"]
        if set(saved["state"]) != (set(ids) if updates else set()):
            raise ValueError("missing/unexpected Adam moments")
        for key, parameter in zip(ids, model.parameters()):
            if not updates:
                break
            entry = saved["state"][key]
            if type(entry) is not dict or set(entry) != {"step", "exp_avg", "exp_avg_sq"}:
                raise ValueError("invalid Adam state fields")
            for name, value in entry.items():
                shape = torch.Size([]) if name == "step" else parameter.shape
                if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                        or value.dtype != torch.float32 or value.shape != shape
                        or not torch.isfinite(value).all()):
                    raise ValueError("invalid Adam tensor shape/dtype/value")
            if entry["step"].item() != updates or (entry["exp_avg_sq"] < 0).any():
                raise ValueError("invalid Adam step/moments")
        optimizer.load_state_dict(copy.deepcopy(saved))
        return optimizer

    def load_state_dict(self, state):
        """Fully validate candidate state before replacing live model/Adam/RNG."""
        keys = {"format", "config", "seed", "feature_dim", "ancestor_sha256", "continuation_sha256",
                "model", "optimizer", "rng", "updates", "counts", "pending", "admissions"}
        if (type(state) is not dict or set(state) != keys or state["format"] != self.format
                or state["config"] != self.config or type(state["feature_dim"]) is not int
                or state["feature_dim"] != 31 or type(state["seed"]) is not int
                or state["seed"] != self.seed or state["ancestor_sha256"] != self.ancestor_sha256
                or state["continuation_sha256"] != self.continuation_sha256):
            raise ValueError("confirmation schema/configuration/lineage mismatch")
        state = copy.deepcopy(state)
        _configuration(state["config"], tail=self._is_tail)
        updates, counts = state["updates"], state["counts"]
        if (type(updates) is not int or not 0 <= updates <= self.config["max_new_updates"]
                or type(counts) is not dict or set(counts) != {"forwards", "optimizer_steps"}
                or any(type(v) is not int or v < 0 for v in counts.values())
                or counts["optimizer_steps"] < updates):
            raise ValueError("invalid confirmation counters")
        self._validate_admissions(state)
        candidate = copy.deepcopy(self.model)
        candidate.load_state_dict(state["model"], strict=True)
        optimizer = self._restored_optimizer(candidate, state["optimizer"], updates)
        rng = np.random.default_rng()
        try:
            rng.bit_generator.state = state["rng"]
        except (TypeError, ValueError, KeyError) as error:
            raise ValueError("invalid sampling RNG state") from error
        self.model, self.optimizer, self.rng = candidate, optimizer, rng
        self.updates, self.counts = updates, counts
        self.pending, self.admissions = state["pending"], state["admissions"]

    @classmethod
    def from_bytes(cls, raw, *, before_forward, before_optimizer, expected_sha256=None):
        """Resume trusted local bytes, optionally checking their external hash."""
        if type(raw) is not bytes or (expected_sha256 is not None
                and hashlib.sha256(raw).hexdigest() != _sha(expected_sha256)):
            raise ValueError("snapshot bytes/hash mismatch")
        state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=False)
        if (type(state) is not dict or state.get("format") != cls.format
                or not {"config", "seed", "feature_dim", "ancestor_sha256"}.issubset(state)):
            raise ValueError("confirmation snapshot format mismatch")
        out = cls(state["config"], seed=state["seed"], feature_dim=state["feature_dim"],
                  ancestor_sha256=state["ancestor_sha256"],
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.load_state_dict(state)
        return out


class CapacityConfirmationWarmLearner(_ConfirmationLearner):
    """Fresh paired initialization; 48 observed plainH8 worlds x 32 TD8 updates."""

    format = "capacity-confirmation-warm-v1"
    _is_tail = False

    def admit_episode(self, data):
        """Own a complete episode, including the historical runner's lists.

        Float32 feature matrices and float64/Python-float costs retain their
        precision when stacked; no dtype coercion repairs malformed inputs.
        """
        self._available()
        if type(data) is not dict or set(data) != {"features", "heuristics", "costs"}:
            raise ValueError("complete observed features/heuristics/costs required")
        owned = {key: _array(np.asarray(data[key]), dtype, shape, key)
                 for key, dtype, shape in (("features", "float32", (65, 4, 31)),
                                           ("heuristics", "float64", (65,)),
                                           ("costs", "float64", (64,)))}
        rows = observed_td_rows(**owned, horizon=8, scale=1000000.)
        if any(not np.isfinite(rows[k]).all() for k in ("observed_cost", "base", "bootstrap_base")):
            raise ValueError("nonfinite observed TD rows")
        source_hash = _digest(dict(schema="observed-plainH8-64"), owned)
        with torch.no_grad():
            future = self._forward(rows["next_states"].copy()).numpy().astype(np.float64)
        if future.shape != (64,) or not np.isfinite(future).all():
            raise ValueError("one finite frozen bootstrap per observed row required")
        targets = rows["observed_cost"] + np.where(
            rows["done"], 0., rows["bootstrap_base"] + future) - rows["base"]
        self._admit(rows["states"], targets, [source_hash], world_index=self.updates // 32, records=[])


class CapacityConfirmationTailLearner(_ConfirmationLearner):
    """Own warm ancestor; common graph continuation; fresh Adam/sampling RNG.

    Use fork_weights with corresponding warm bytes and their expected SHA256.
    continuation_sha256 in config identifies the separate frozen graph policy.
    Use the same seed and shared records for all tail arms to pair minibatches.
    """

    format = "capacity-confirmation-tail-v1"
    _is_tail = True

    @classmethod
    def fork_weights(cls, raw, config, *, seed, expected_sha256, before_forward, before_optimizer):
        _sha(expected_sha256)
        _configuration(config, tail=True)
        warm = CapacityConfirmationWarmLearner.from_bytes(raw, expected_sha256=expected_sha256,
            before_forward=before_forward, before_optimizer=before_optimizer)
        if (warm.updates != 1536 or warm.pending is not None
                or warm.config["architecture"] != config["architecture"]):
            raise ValueError("completed corresponding warm ancestor required; no architecture conversion")
        out = cls(config, seed=seed, feature_dim=31, ancestor_sha256=expected_sha256,
                  before_forward=before_forward, before_optimizer=before_optimizer)
        out.model.load_state_dict(warm.model.state_dict(), strict=True)
        return out

    def admit_episode(self, data):
        raise ValueError("tail learner requires two complete native branches via admit_tails")

    def admit_tails(self, records, *, world_index, tape_sha256):
        """Freeze one complete native world before any optimizer callback.

        Branches are canonicalized by root so caller ordering cannot unpair TD/MC
        minibatches. Each world index 0..23 is admitted at most once.
        """
        self._available()
        _sha(tape_sha256)
        snapshots = validate_records(records, world_index=world_index,
                                     continuation_sha256=self.continuation_sha256,
                                     tape_sha256=tape_sha256)
        if world_index in {b["world_index"] for b in self.admissions}:
            raise ValueError("native world already admitted")
        snapshots = sorted(snapshots, key=lambda r: r["root_epoch"])
        validated = [mc_rows(r) for r in snapshots]
        states = np.concatenate([r["states"] for r in validated])
        hashes = [r["metadata"]["source_data_sha256"] for r in validated]
        metadata = [r["metadata"]["record"] for r in validated]
        if self.config["method"] == "planner_tail_mc":
            targets = np.concatenate([r["targets"] for r in validated])
        else:
            values = np.concatenate([self.residuals(states[i:i + 64].copy())
                                     for i in range(0, len(states), 64)])
            outputs, offset = [], 0
            for record in snapshots:
                length = len(record["costs"])
                rows = td_rows(record, frozen_residuals=np.r_[values[offset:offset + length], 0.])
                outputs.append(rows["targets"])
                offset += length
            targets = np.concatenate(outputs)
        self._admit(states, targets, hashes, world_index=world_index, records=metadata)
