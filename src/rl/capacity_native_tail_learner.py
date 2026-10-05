"""Two-branch native-tail admission with inherited weight forks and fitting.

Native public features are not predicted endpoint features; this is not an exact
model-bias correction. The collector owns native provenance, frozen continuation
execution and tape identity. Checkpoint hashes bind declarations and data, not
independently verified collection history. No scientific job is started here.
"""

import copy
import hashlib
import json

import numpy as np

from src.rl.capacity_native_tail_targets import (
    TAIL_SCHEMA, TAILS_PER_WORLD, mc_rows, td_rows, validate_metadata,
    validate_record, validate_records,
)
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner


class CapacityNativeTailLearner(CapacityPlannerTailLearner):
    format = "capacity-native-tail-learner-v1"

    def __init__(self, config, *, ancestor_sha256, **kwargs):
        config = copy.deepcopy(config)
        if (not isinstance(config, dict)
                or config.get("method") not in ("planner_tail_td", "planner_tail_mc")
                or config.get("tail_policy") != "frozen_mpc"
                or config.get("tail_schema") != TAIL_SCHEMA
                or type(config.get("tails_per_world")) is not int
                or config["tails_per_world"] != TAILS_PER_WORLD
                or type(config.get("continuation_sha256")) is not str
                or config["continuation_sha256"] != ancestor_sha256):
            raise ValueError("explicit native two-branch schema and frozen ancestor policy required")
        # Bind only JSON-safe configuration, before constructing any model.
        json.dumps(config, sort_keys=True, separators=(",", ":"), allow_nan=False)
        super().__init__(config, ancestor_sha256=ancestor_sha256, **kwargs)
        self._tail_admission = None

    def _pending_sha256(self, pending, metadata, world_index):
        header = dict(format=self.format, config=self.config, world_index=world_index,
                      ancestor_sha256=self.ancestor_sha256, records=metadata,
                      source_hashes=pending["source_hashes"])
        digest = hashlib.sha256(b"capacity-native-tail-pending-v1\0")
        digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"),
                                 allow_nan=False).encode("ascii") + b"\0")
        # The inherited update increments completed, but does not change targets.
        for name, dtype in (("states", "<f4"), ("targets", "<f8")):
            values = np.asarray(pending[name], dtype=dtype)
            digest.update(name.encode("ascii") + b"\0" + values.dtype.str.encode("ascii") + b"\0")
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def admit_tails(self, records, *, world_index=None, tape_sha256=None):
        """Admit one complete native world, before any bounded inherited update.

        World index may be supplied explicitly or inferred as the smaller root.
        The optional trusted tape hash is checked on both branches. TD freezes
        residuals in batches of at most 64, using owned public native features;
        MC makes no model calls. Both methods keep targets in float64.
        """
        if self.pending is not None or self.updates + 32 > 768:
            raise ValueError("pending targets or exhausted update allowance")
        if not isinstance(records, (list, tuple)) or len(records) != TAILS_PER_WORLD:
            raise ValueError("exactly two native branches per world required")
        snapshots = [validate_record(r, continuation_sha256=self.ancestor_sha256,
                                     tape_sha256=tape_sha256) for r in records]
        if world_index is None:
            world_index = min(r["root_epoch"] for r in snapshots)
        snapshots = validate_records(snapshots, world_index=world_index,
                                     continuation_sha256=self.ancestor_sha256,
                                     tape_sha256=tape_sha256)
        # Validate every return and own the entire cohort before any callback.
        validated = [mc_rows(r) for r in snapshots]
        metadata = [r["metadata"]["record"] for r in validated]
        states = np.concatenate([r["states"] for r in validated])
        hashes = [r["metadata"]["source_data_sha256"] for r in validated]
        if self.config["method"] == "planner_tail_mc":
            targets = np.concatenate([r["targets"] for r in validated])
        else:
            values = np.concatenate([self.residuals(states[i:i + 64].copy())
                                     for i in range(0, len(states), 64)])
            if values.shape != (len(states),):
                raise ValueError("one frozen residual per native nonterminal state required")
            outputs, offset = [], 0
            for record in snapshots:
                length = len(record["costs"])
                rows = td_rows(record, frozen_residuals=np.r_[values[offset:offset + length], 0.])
                outputs.append(rows["targets"])
                offset += length
            targets = np.concatenate(outputs)
        pending = dict(states=states, targets=targets, source_hashes=hashes)
        binding = dict(world_index=world_index, records=metadata,
                       pending_sha256=self._pending_sha256(pending, metadata, world_index))
        self._admit(states, targets, hashes)
        self._tail_admission = binding

    def admit_episode(self, rows):
        """Generic episode rows cannot bypass native branch admission."""
        raise ValueError("native learner requires admit_tails with two complete native records")

    def state_dict(self):
        state = super().state_dict()
        state["tail_admission"] = (copy.deepcopy(self._tail_admission)
                                   if self.pending is not None else None)
        return state

    def load_state_dict(self, state):
        """Validate the owned checkpoint fully before replacing any live state."""
        keys = {"format", "config", "feature_dim", "model", "optimizer", "rng",
                "updates", "counts", "pending", "ancestor_sha256", "tail_admission"}
        if (not isinstance(state, dict) or set(state) != keys
                or state["format"] != self.format or state["config"] != self.config
                or state["ancestor_sha256"] != self.ancestor_sha256):
            raise ValueError("native-tail configuration/schema/ancestor mismatch")
        state = copy.deepcopy(state)
        pending, binding = state["pending"], state["tail_admission"]
        if pending is None:
            if binding is not None:
                raise ValueError("native admission without pending targets")
        else:
            if (not isinstance(binding, dict)
                    or set(binding) != {"records", "world_index", "pending_sha256"}
                    or not isinstance(pending, dict)
                    or set(pending) != {"states", "targets", "completed", "source_hashes"}):
                raise ValueError("missing native-tail pending binding")
            metadata = validate_metadata(binding["records"], world_index=binding["world_index"],
                                         continuation_sha256=self.ancestor_sha256)
            length = sum(64 - record["start_epoch"] for record in metadata)
            x, y, hashes = pending["states"], pending["targets"], pending["source_hashes"]
            if (type(x) is not np.ndarray or x.shape != (length, 4, 31)
                    or x.dtype != np.dtype("float32") or type(y) is not np.ndarray
                    or y.shape != (length,) or y.dtype != np.dtype("float64")
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not isinstance(hashes, list) or len(hashes) != TAILS_PER_WORLD
                    or any(type(h) is not str or len(h) != 64
                           or any(c not in "0123456789abcdef" for c in h) for h in hashes)
                    or type(binding["pending_sha256"]) is not str
                    or binding["pending_sha256"] != self._pending_sha256(
                        pending, metadata, binding["world_index"])):
                raise ValueError("invalid native rows or metadata/config/source/pending hash")
        # Parent builds candidate model, optimizer and RNG before its atomic swap.
        super().load_state_dict(state)
        self._tail_admission = binding
