"""Edge helpers for distributed manufacturing graphs."""

from __future__ import annotations

Edge = tuple[int, int]


def complete_undirected_edges(num_nodes: int) -> tuple[Edge, ...]:
    """Return all undirected edges for a complete graph."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    return tuple((i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes))
