"""Six-tail policy-bound admission; historical weight forks and fitting stay intact.

TD and MC consume the same public predicted-root data within a label policy.
Adaptive and frozen-MPC continuations are separate trajectories, not relabelings.
The collector owns generation and must keep frozen-MPC continuation weights at
the declared ancestor throughout generation. This adapter checks that declared
provenance; it never generates trajectories or changes continuation weights.
"""

import copy
import hashlib
import json

import numpy as np

from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from src.rl.capacity_planner_tail_targets import mc_rows, td_rows


TAIL_SCHEMA = "capacity-policy-tail-record-v1"
TAILS_PER_WORLD = 6
QUANTILES = (.1, .5, .9)


class CapacityPolicyTailLearner(CapacityPlannerTailLearner):
    format = "capacity-policy-tail-learner-v1"

    def __init__(self, config, *, ancestor_sha256, **kwargs):
        config = copy.deepcopy(config)
        policy, method = config.get("tail_policy"), config.get("method")
        if (policy not in ("adaptive", "frozen_mpc")
                or method not in ("planner_tail_td", "planner_tail_mc")
                or method == "planner_tail_mc" and policy != "frozen_mpc"):
            raise ValueError("explicit tail_policy and compatible TD/MC method required")
        continuation = config.get("continuation_sha256")
        if ((policy == "frozen_mpc" and continuation != ancestor_sha256)
                or (policy == "adaptive" and continuation is not None)):
            raise ValueError("continuation_sha256 must bind the frozen ancestor only")
        config.setdefault("continuation_sha256", None)
        config.setdefault("tails_per_world", TAILS_PER_WORLD)
        config.setdefault("tail_schema", TAIL_SCHEMA)
        if (type(config["tails_per_world"]) is not int
                or config["tails_per_world"] != TAILS_PER_WORLD
                or config["tail_schema"] != TAIL_SCHEMA):
            raise ValueError("exact six-tail policy schema required")
        super().__init__(config, ancestor_sha256=ancestor_sha256, **kwargs)
        self._tail_admission = None

    def _tail_metadata(self, records):
        if not isinstance(records, (list, tuple)) or len(records) != TAILS_PER_WORLD:
            raise ValueError("exactly six policy tails per reference world required")
        keys = ("decision_epoch", "candidate", "quantile", "start_epoch",
                "label_policy", "continuation_sha256")
        metadata, roots, identities = [], {}, set()
        for record in records:
            if not isinstance(record, dict) or any(k not in record for k in keys):
                raise ValueError("complete policy-tail metadata required")
            decision, candidate, quantile, start, policy, continuation = (
                record[k] for k in keys)
            if (policy != self.config["tail_policy"]
                    or continuation != self.config["continuation_sha256"]):
                raise ValueError("tail policy/continuation hash mismatch")
            if (type(decision) is not int or not 0 <= decision < 48
                    or type(candidate) is not int or not 0 <= candidate < 16
                    or type(start) is not int or start != decision + 8
                    or not isinstance(quantile, (float, int, np.floating, np.integer))
                    or quantile not in QUANTILES):
                raise ValueError("invalid predicted root or tail start")
            identity = (decision, candidate, float(quantile))
            if identity in identities:
                raise ValueError("duplicate decision/candidate/quantile tail")
            identities.add(identity)
            roots.setdefault(decision, []).append((candidate, float(quantile)))
            metadata.append(dict(zip(keys, (decision, candidate, float(quantile),
                                            start, policy, continuation))))
        if (len(roots) != 2 or any(len(rows) != 3
                or len({candidate for candidate, _ in rows}) != 1
                or sorted(q for _, q in rows) != list(QUANTILES)
                for rows in roots.values())):
            raise ValueError("two decision roots, one candidate and three quantiles each required")
        return metadata

    def _pending_sha256(self, pending, metadata):
        header = dict(format=self.format, config=self.config,
                      ancestor_sha256=self.ancestor_sha256, records=metadata,
                      source_hashes=pending["source_hashes"])
        digest = hashlib.sha256(json.dumps(header, sort_keys=True, separators=(",", ":"),
                                          allow_nan=False).encode("ascii"))
        # Completed updates are deliberately excluded: inherited fitting increments
        # that counter while the admitted states and frozen targets stay unchanged.
        for name, dtype in (("states", "<f4"), ("targets", "<f8")):
            values = np.asarray(pending[name], dtype=dtype)
            digest.update(name.encode("ascii") + b"\0")
            digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
            digest.update(values.tobytes(order="C"))
        return digest.hexdigest()

    def admit_tails(self, records):
        if self.pending is not None or self.updates + 32 > 768:
            raise ValueError("pending targets or exhausted update allowance")
        metadata = self._tail_metadata(records)
        # Copy the entire supplied cohort before model callbacks. TD never rereads
        # caller-owned arrays after MC validation or after a bootstrap callback.
        snapshots = []
        for record in records:
            if any(k not in record for k in ("epochs", "features", "heuristics", "costs")):
                raise ValueError("complete stored tail arrays required")
            snapshots.append({k: copy.deepcopy(record[k])
                              for k in ("epochs", "features", "heuristics", "costs")})
        validated = [mc_rows(**record) for record in snapshots]
        if any(row["metadata"]["start_epoch"] != meta["start_epoch"]
               for row, meta in zip(validated, metadata)):
            raise ValueError("stored epochs must start at decision_epoch + 8 and end at 64")
        states = np.concatenate([row["states"] for row in validated])
        hashes = [row["metadata"]["source_data_sha256"] for row in validated]
        if self.config["method"] == "planner_tail_mc":
            targets = np.concatenate([row["targets"] for row in validated])
        else:
            values = np.concatenate([self.residuals(states[i:i + 64])
                                     for i in range(0, len(states), 64)])
            outputs, offset = [], 0
            for record, row in zip(snapshots, validated):
                length = len(row["states"])
                output = td_rows(**record,
                                 frozen_residuals=np.r_[values[offset:offset + length], 0.])
                outputs.append(output["targets"])
                offset += length
            targets = np.concatenate(outputs)
        pending = dict(states=states, targets=targets, source_hashes=hashes)
        binding = dict(records=metadata, pending_sha256=self._pending_sha256(pending, metadata))
        self._admit(states, targets, hashes)
        self._tail_admission = binding

    def state_dict(self):
        state = super().state_dict()
        state["tail_admission"] = (copy.deepcopy(self._tail_admission)
                                   if self.pending is not None else None)
        return state

    def load_state_dict(self, state):
        if (state.get("format") != self.format or state.get("config") != self.config
                or state.get("ancestor_sha256") != self.ancestor_sha256
                or "tail_admission" not in state):
            raise ValueError("policy-tail configuration/schema/ancestor mismatch")
        pending, binding = state.get("pending"), state["tail_admission"]
        if pending is None:
            if binding is not None:
                raise ValueError("tail admission without pending targets")
        else:
            if (not isinstance(binding, dict)
                    or set(binding) != {"records", "pending_sha256"}
                    or not isinstance(pending, dict)
                    or set(pending) != {"states", "targets", "completed", "source_hashes"}):
                raise ValueError("missing policy-tail pending binding")
            metadata = self._tail_metadata(binding["records"])
            length = sum(64 - record["start_epoch"] for record in metadata)
            x, y = np.asarray(pending["states"]), np.asarray(pending["targets"])
            hashes = pending["source_hashes"]
            if (binding["records"] != metadata or x.shape != (length, 4, 31)
                    or x.dtype != np.dtype("float32") or y.shape != (length,)
                    or y.dtype != np.dtype("float64")
                    or not np.isfinite(x).all() or not np.isfinite(y).all()
                    or not isinstance(hashes, list) or len(hashes) != TAILS_PER_WORLD
                    or any(not isinstance(h, str) or len(h) != 64
                           or any(c not in "0123456789abcdef" for c in h) for h in hashes)
                    or binding["pending_sha256"] != self._pending_sha256(pending, metadata)):
                raise ValueError("invalid policy-tail rows or pending target hash")
        super().load_state_dict(state)
        self._tail_admission = copy.deepcopy(binding)
