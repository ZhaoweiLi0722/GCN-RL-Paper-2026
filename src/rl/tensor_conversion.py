"""Safe tensor-to-NumPy conversion helpers for replay-state heuristics."""

from __future__ import annotations

from typing import Any

import numpy as np


def independent_contiguous_numpy(tensor: Any) -> np.ndarray:
    """Return an owned C-contiguous NumPy copy of a detached CPU tensor."""

    shared = tensor.detach().to(device="cpu").contiguous().numpy()
    return np.array(shared, copy=True, order="C")
