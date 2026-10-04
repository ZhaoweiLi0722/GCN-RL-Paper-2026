"""Pure-data residual targets for complete planner forecast tails ending at 64.

Each builder returns L rows, one for every nonterminal continuation state.
The six conventional ``capacity_value_learner.td_rows`` array keys are retained;
``bootstrap_residual`` and ``targets`` are also scaled by 1e6. Features are
float32; all cost components and target arithmetic are float64. Inputs are
copied, never modified or retained by reference. No model or environment is
imported, and this module has no scientific execution entry point.
"""

import hashlib

import numpy as np


COST_SCALE = 1000000.
TERMINAL_EPOCH = 64
TD_HORIZON = 8
MAX_TAIL_LENGTH = 56
FEATURE_SHAPE = (4, 31)


def _float_array(values, name, dtype="<f8"):
    raw = np.asarray(values)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be real numeric data")
    with np.errstate(over="ignore", invalid="ignore"):
        out = np.array(raw, dtype=dtype, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite and representable as {out.dtype}")
    return out


def _tail(features, heuristics, costs, epochs):
    c = _float_array(costs, "costs")
    if c.ndim != 1 or not 1 <= len(c) <= MAX_TAIL_LENGTH:
        raise ValueError("costs must have shape (L,) with tail length 1..56")
    if (c < 0).any():
        raise ValueError("costs must be nonnegative")
    length = len(c)
    e = np.asarray(epochs)
    expected = np.arange(TERMINAL_EPOCH - length, TERMINAL_EPOCH + 1)
    if (e.shape != (length + 1,) or e.dtype.kind not in "iu"
            or not np.array_equal(e, expected)):
        raise ValueError("epochs must be a complete contiguous integer path ending at 64")
    e = np.array(e, dtype="<i8", copy=True)
    x = _float_array(features, "features", dtype="<f4")
    h = _float_array(heuristics, "heuristics")
    if x.shape != (length + 1, *FEATURE_SHAPE):
        raise ValueError("features must have shape (L+1, 4, 31)")
    if h.shape != (length + 1,):
        raise ValueError("heuristics must have shape (L+1,)")
    return x, h, c, e


def _rows(features, heuristics, costs, *, epochs, frozen_residuals, kind):
    x, h, c, e = _tail(features, heuristics, costs, epochs)
    length = len(c)
    residuals = (_float_array(frozen_residuals, "frozen_residuals")
                 if kind == "td8" else np.zeros(length + 1, dtype=np.float64))
    if residuals.shape != (length + 1,):
        raise ValueError("frozen_residuals must have shape (L+1,) in raw cost units")
    # Own an immutable snapshot; never defer evaluation to a changing model.
    residuals.flags.writeable = False
    ends = (np.minimum(np.arange(length) + TD_HORIZON, length) if kind == "td8"
            else np.full(length, length, dtype=np.int64))
    done = e[ends] == TERMINAL_EPOCH
    bootstrap_h = np.where(done, 0., h[ends])
    bootstrap_r = np.where(done, 0., residuals[ends])
    with np.errstate(over="ignore", invalid="ignore"):
        observed = (np.asarray([np.sum(c[t:end], dtype=np.float64)
                                for t, end in enumerate(ends)], dtype=np.float64)
                    if kind == "td8" else np.cumsum(c[::-1], dtype=np.float64)[::-1])
        targets = (observed + (bootstrap_h + bootstrap_r) - h[:-1]) / COST_SCALE
    if not np.isfinite(observed).all() or not np.isfinite(targets).all():
        raise ValueError("cost sum or residual target overflows float64")

    # Bind the canonical supplied data, not its unverified collection provenance.
    digest = hashlib.sha256(b"capacity-planner-tail-data-v1\0")
    for name, values in (("epochs", e), ("features", x), ("heuristics", h), ("costs", c)):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(values.tobytes(order="C"))

    return dict(
        states=x[:-1].copy(), next_states=x[ends].copy(),
        base=h[:-1] / COST_SCALE, bootstrap_base=bootstrap_h / COST_SCALE,
        observed_cost=observed / COST_SCALE, done=done,
        bootstrap_residual=bootstrap_r / COST_SCALE, targets=targets,
        metadata=dict(format="capacity-planner-tail-targets-v1", target_kind=kind,
                      source_data_sha256=digest.hexdigest(), tail_length=length,
                      start_epoch=int(e[0]), terminal_epoch=TERMINAL_EPOCH,
                      td_horizon=TD_HORIZON if kind == "td8" else None,
                      cost_scale=COST_SCALE),
    )


def td_rows(features, heuristics, costs, *, epochs, frozen_residuals):
    """Build all L eight-step TD rows from a complete 1..56-step forecast tail.

    ``epochs`` is the integer state path [64-L, ..., 64]; costs[i] belongs
    to epochs[i] -> epochs[i+1]. ``frozen_residuals`` is an explicit finite
    (L+1,) snapshot in RAW cost units aligned with features, not a model or
    already-scaled network outputs. For each t, end=min(t+8, L), and target is
    (sum(costs[t:end]) + H[end] + residuals[end] - H[t]) / 1e6, with both
    bootstrap terms replaced by zero at epoch 64. Terminal heuristic/residual inputs
    must still be finite, but their values do not affect the target.

    All array fields have a leading row dimension of L. Downstream adapters
    may concatenate rows; the historical learner's 64-row admission/update
    logic is not called or changed here.
    """
    return _rows(features, heuristics, costs, epochs=epochs,
                 frozen_residuals=frozen_residuals, kind="td8")


def mc_rows(features, heuristics, costs, *, epochs):
    """Build all L direct-return rows from the SAME complete forecast-tail data.

    Target[t]: (sum(costs[t:]) - H[t]) / 1e6. ``next_states`` repeats
    the terminal feature row, ``done`` is all True, and both bootstrap components
    are zero. This describes regression on the supplied continuation costs,
    not a claim that they are observed native-world returns.
    """
    return _rows(features, heuristics, costs, epochs=epochs,
                 frozen_residuals=None, kind="mc")
