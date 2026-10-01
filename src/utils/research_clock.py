"""A same-host, cross-process clock for live research budget deadlines."""

import math
import time


CLOCK_ID = "posix-clock-monotonic-v1"
INJECTED_CLOCK_ID = "injected-test-clock-not-cross-process"


def shared_monotonic():
    # Python 3.9 on this Mac gives time.monotonic() a process-local origin.
    # Do not fall back to it when comparing a child's ledger with its parent.
    try:
        value = time.clock_gettime(time.CLOCK_MONOTONIC)
    except (AttributeError, OSError) as exc:
        raise RuntimeError("cross-process CLOCK_MONOTONIC is required") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError("invalid cross-process clock value")
    return value


def clock_record():
    shared_monotonic()
    resolution = time.clock_getres(time.CLOCK_MONOTONIC)
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError("invalid cross-process clock resolution")
    return {"id": CLOCK_ID, "api": "time.clock_gettime(time.CLOCK_MONOTONIC)",
            "resolution_seconds": resolution, "scope": "same_host_same_boot_live_processes"}
