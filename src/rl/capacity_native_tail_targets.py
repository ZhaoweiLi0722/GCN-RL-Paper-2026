"""Pure-data admission and residual targets for two native patient branches.

Public native features differ from predicted endpoint features. Native returns
on these features are not an exact model-bias correction at forecast endpoints.
Provenance and hashes bind the supplied declaration, not independently verified
collection history; the collector owns native forks and frozen-MPC execution.
No model, forward call, environment, or optimizer is imported or invoked here.
"""

import hashlib
import json

import numpy as np

from src.rl.capacity_planner_tail_targets import (
    COST_SCALE, FEATURE_SHAPE, TD_HORIZON, TERMINAL_EPOCH,
    mc_rows as _mc_rows, td_rows as _td_rows,
)


TAIL_SCHEMA = "capacity-native-tail-record-v1"
TAILS_PER_WORLD = 2
_METADATA_KEYS = (
    "format", "root_epoch", "candidate", "start_epoch", "label_policy",
    "continuation_sha256", "source_state_sha256", "tape_sha256", "provenance",
)
_ARRAY_KEYS = ("epochs", "features", "heuristics", "costs")
_RECORD_KEYS = frozenset((*_METADATA_KEYS, *_ARRAY_KEYS))
_DIAGNOSTIC_KEYS = frozenset(("prefix_costs", "total_branch_cost", "settlement",
                              "policy_is_fixed_parent_not_updated_student", "feature_distribution"))
_ROW_KEYS = (
    "states", "next_states", "base", "bootstrap_base", "observed_cost", "done",
    "bootstrap_residual", "targets",
)


def _sha256(value, name):
    if (type(value) is not str or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError(f"{name} must be a lowercase SHA256 hex string")
    return value


def _float_array(value, name, dtype, shape):
    # Do not silently narrow double features or promote already-rounded costs.
    expected = np.dtype(dtype)
    if (type(value) is not np.ndarray or value.dtype.kind != "f"
            or value.dtype.itemsize != expected.itemsize or value.shape != shape):
        raise ValueError(f"{name} must be a {expected.name} ndarray of shape {shape}")
    result = np.array(value, dtype=expected, order="C", copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def _metadata(record, *, continuation_sha256=None, tape_sha256=None):
    if not isinstance(record, dict) or set(record) != set(_METADATA_KEYS):
        raise ValueError("exact native-tail metadata required; no forecast/quantile fields")
    for key, expected in (("format", TAIL_SCHEMA), ("label_policy", "frozen_mpc"),
                          ("provenance", "native_patient_branch")):
        if type(record[key]) is not str or record[key] != expected:
            raise ValueError(f"{key} must be {expected}")
    root, candidate, start = (record[k] for k in ("root_epoch", "candidate", "start_epoch"))
    if (type(root) is not int or not 0 <= root < 48
            or type(candidate) is not int or not 0 <= candidate < 16
            or type(start) is not int or start != root + TD_HORIZON):
        raise ValueError("invalid root/candidate or start_epoch != root_epoch + 8")
    for key in ("continuation_sha256", "source_state_sha256", "tape_sha256"):
        _sha256(record[key], key)
    for key, expected in (("continuation_sha256", continuation_sha256),
                          ("tape_sha256", tape_sha256)):
        if expected is not None and record[key] != _sha256(expected, key):
            raise ValueError(f"{key} mismatch")
    return {key: record[key] for key in _METADATA_KEYS}


def _json_snapshot(value):
    """Canonical JSON diagnostics without coercing keys or nonfinite scalars."""
    if type(value) is dict:
        if any(type(k) is not str for k in value):
            raise ValueError("settlement keys must be strings")
        return {k: _json_snapshot(v) for k, v in value.items()}
    if type(value) is list:
        return [_json_snapshot(v) for v in value]
    if (value is None or type(value) in (str, bool, int)
            or type(value) is float and np.isfinite(value)):
        return value
    raise ValueError("settlement must contain only finite JSON data")


def validate_record(record, *, continuation_sha256=None, tape_sha256=None):
    """Return a canonical owned snapshot of one exact v1 record.

    Metadata integers are Python ints (not bools). Epochs may be an integer
    ndarray or a list/tuple of integers; features must already be float32 and
    heuristics/costs float64 ndarrays. Arrays are copied into little-endian,
    C-contiguous storage. The collector's complete optional audit bundle
    (prefix_costs, total_branch_cost, settlement, fixed-parent flag and native
    feature_distribution) is copied and hash-bound, but not used as target data.
    All other extra fields, including forecast/quantile labels, are rejected.

    Optional expected hashes are checked against caller-owned trusted context.
    Single-record validation does not establish the two-branch world schedule;
    use ``validate_records`` or ``admit_tails`` at that boundary.
    """
    if (not isinstance(record, dict)
            or set(record) not in (_RECORD_KEYS, _RECORD_KEYS | _DIAGNOSTIC_KEYS)):
        raise ValueError("exact native-tail v1 fields required; no forecast/quantile fields")
    out = _metadata({key: record[key] for key in _METADATA_KEYS},
                    continuation_sha256=continuation_sha256, tape_sha256=tape_sha256)

    epochs = record["epochs"]
    if isinstance(epochs, (list, tuple)):
        if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
               for v in epochs):
            raise ValueError("epochs must contain only integers, not bools")
    elif type(epochs) is not np.ndarray:
        raise ValueError("epochs must be an integer ndarray or list/tuple")
    epochs = np.asarray(epochs)
    expected = np.arange(out["start_epoch"], TERMINAL_EPOCH + 1, dtype="<i8")
    if (epochs.dtype.kind not in "iu" or epochs.shape != expected.shape
            or not np.array_equal(epochs, expected)):
        raise ValueError("epochs must be start_epoch..64 inclusive, contiguous and integer")
    count = len(expected)
    out.update(
        epochs=expected.copy(),
        features=_float_array(record["features"], "features", "<f4", (count, *FEATURE_SHAPE)),
        heuristics=_float_array(record["heuristics"], "heuristics", "<f8", (count,)),
        costs=_float_array(record["costs"], "costs", "<f8", (count - 1,)),
    )
    if (out["costs"] < 0).any():
        raise ValueError("costs must be nonnegative")
    if _DIAGNOSTIC_KEYS.issubset(record):
        prefix = _float_array(record["prefix_costs"], "prefix_costs", "<f8", (TD_HORIZON,))
        total, settlement = record["total_branch_cost"], record["settlement"]
        distribution = record["feature_distribution"]
        if ((prefix < 0).any() or type(total) is not float or not np.isfinite(total) or total < 0
                or type(settlement) is not dict or settlement.get("settled") is not True
                or record["policy_is_fixed_parent_not_updated_student"] is not True
                or type(distribution) is not str
                or distribution != "native_observed_not_forecast_endpoint"):
            raise ValueError("invalid native collector diagnostics")
        out.update(prefix_costs=prefix, total_branch_cost=total,
                   settlement=_json_snapshot(settlement),
                   policy_is_fixed_parent_not_updated_student=True,
                   feature_distribution=distribution)
    return out


def validate_metadata(records, *, world_index, continuation_sha256, tape_sha256=None):
    """Copy/check two metadata-only records for admission or pending restore.

    Roots are i and i+24; candidate=(i+8*root_slot)%16. The two records must
    share the declared frozen continuation and tape. No response quantiles are
    admitted. This does not validate arrays or authenticate collection history.
    """
    if type(world_index) is not int or not 0 <= world_index < 24:
        raise ValueError("world_index must be an int in 0..23")
    _sha256(continuation_sha256, "continuation_sha256")
    if not isinstance(records, (list, tuple)) or len(records) != TAILS_PER_WORLD:
        raise ValueError("exactly two native branches per world required")
    snapshots = tuple(_metadata(r, continuation_sha256=continuation_sha256,
                               tape_sha256=tape_sha256) for r in records)
    identities = [(r["root_epoch"], r["candidate"]) for r in snapshots]
    if len(set(identities)) != TAILS_PER_WORLD:
        raise ValueError("duplicate native branch identities")
    expected = {(world_index + 24 * slot, (world_index + 8 * slot) % 16)
                for slot in range(TAILS_PER_WORLD)}
    if set(identities) != expected:
        raise ValueError("native branches must use roots i/i+24 and prescribed candidates")
    if len({r["tape_sha256"] for r in snapshots}) != 1:
        raise ValueError("native branches must share the same world tape_sha256")
    return snapshots


def validate_records(records, *, world_index, continuation_sha256, tape_sha256=None):
    """Copy exactly two complete native branches, preserving supplied order.

    Validates the same schedule as ``validate_metadata`` plus all stored arrays.
    A trusted expected tape can additionally be supplied by the runner.
    """
    if not isinstance(records, (list, tuple)) or len(records) != TAILS_PER_WORLD:
        raise ValueError("exactly two native branches per world required")
    snapshots = tuple(validate_record(r, continuation_sha256=continuation_sha256,
                                      tape_sha256=tape_sha256) for r in records)
    validate_metadata([{key: r[key] for key in _METADATA_KEYS} for r in snapshots],
                      world_index=world_index, continuation_sha256=continuation_sha256,
                      tape_sha256=tape_sha256)
    return snapshots


def _source_hash(record):
    header = {key: record[key] for key in _METADATA_KEYS}
    if "prefix_costs" in record:
        header.update({key: record[key] for key in _DIAGNOSTIC_KEYS if key != "prefix_costs"})
    digest = hashlib.sha256(b"capacity-native-tail-source-v1\0")
    digest.update(json.dumps(header, sort_keys=True, separators=(",", ":"),
                             allow_nan=False).encode("ascii") + b"\0")
    for name in (*_ARRAY_KEYS, *(("prefix_costs",) if "prefix_costs" in record else ())):
        values = record[name]
        digest.update(name.encode("ascii") + b"\0" + values.dtype.str.encode("ascii") + b"\0")
        digest.update(np.asarray(values.shape, dtype="<i8").tobytes())
        digest.update(values.tobytes(order="C"))
    return digest.hexdigest()


def source_data_sha256(record):
    """Bind all v1 metadata and canonical arrays, independent of memory layout.

    This is a content hash, not proof of native provenance or a target hash:
    TD frozen residuals are deliberately not part of the source record.
    """
    return _source_hash(validate_record(record))


def _rows(record, *, kind, frozen_residuals=None):
    arrays = {key: record[key] for key in _ARRAY_KEYS}
    if kind == "td8":
        residuals = _float_array(frozen_residuals, "frozen_residuals", "<f8",
                                 (len(record["epochs"]),))
        rows = _td_rows(**arrays, frozen_residuals=residuals)
    else:
        rows = _mc_rows(**arrays)
    rows["metadata"].update(
        format="capacity-native-tail-targets-v1", source_data_sha256=_source_hash(record),
        record={key: record[key] for key in _METADATA_KEYS},
        feature_source="public_native_state",
    )
    return rows


def mc_rows(record, *, continuation_sha256=None, tape_sha256=None):
    """Return MC rows: (sum(costs[t:]) - H[t])/1e6, in float64.

    Reuses the planner-tail formula exactly, but only after native admission.
    All rows are terminal, with zero bootstrap terms and copied float32 states.
    """
    return _rows(validate_record(record, continuation_sha256=continuation_sha256,
                                 tape_sha256=tape_sha256), kind="mc")


def td_rows(record, *, frozen_residuals, continuation_sha256=None, tape_sha256=None):
    """Return TD8 rows with an explicit float64 snapshot in RAW cost units.

    At end=min(t+8,L): (sum(costs[t:end])+H[end]+residual[end]-H[t])/1e6.
    Both bootstrap terms are zero at epoch 64; all terminal inputs must still
    be finite. No model callback or scaled network output is accepted.
    """
    return _rows(validate_record(record, continuation_sha256=continuation_sha256,
                                 tape_sha256=tape_sha256), kind="td8",
                 frozen_residuals=frozen_residuals)


def admit_tails(records, *, world_index, continuation_sha256, tape_sha256=None,
                target_kind="mc", frozen_residuals=None):
    """Build concatenated owned rows and source hashes for one complete world.

    ``target_kind`` is ``mc`` or ``td8``. For TD8, ``frozen_residuals`` must be
    a list/tuple of two float64 arrays aligned with records in supplied order,
    including each terminal state. MC rejects unused residuals. The learner
    can consume states, targets and source_hashes without any scientific call
    here. Source hashes bind records, not learner targets or optimizer state.
    """
    if type(target_kind) is not str or target_kind not in ("mc", "td8"):
        raise ValueError("target_kind must be mc or td8")
    snapshots = validate_records(records, world_index=world_index,
                                 continuation_sha256=continuation_sha256,
                                 tape_sha256=tape_sha256)
    if target_kind == "td8":
        if not isinstance(frozen_residuals, (list, tuple)) or len(frozen_residuals) != TAILS_PER_WORLD:
            raise ValueError("two aligned frozen_residuals snapshots required for td8")
        residuals = [_float_array(v, "frozen_residuals", "<f8", (len(r["epochs"]),))
                     for v, r in zip(frozen_residuals, snapshots)]
    else:
        if frozen_residuals is not None:
            raise ValueError("mc does not accept frozen_residuals")
        residuals = [None] * TAILS_PER_WORLD
    rows = [_rows(r, kind=target_kind, frozen_residuals=v) for r, v in zip(snapshots, residuals)]
    result = {key: np.concatenate([r[key] for r in rows]) for key in _ROW_KEYS}
    result.update(
        source_hashes=[r["metadata"]["source_data_sha256"] for r in rows],
        metadata=dict(format="capacity-native-tail-admission-v1", world_index=world_index,
                      target_kind=target_kind, tails_per_world=TAILS_PER_WORLD,
                      records=[r["metadata"] for r in rows], cost_scale=COST_SCALE,
                      terminal_epoch=TERMINAL_EPOCH),
    )
    return result
