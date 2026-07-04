"""Edge helpers for distributed manufacturing graphs."""

from __future__ import annotations

Edge = tuple[int, int]


def complete_undirected_edges(num_nodes: int) -> tuple[Edge, ...]:
    """Return all undirected edges for a complete graph."""

    if num_nodes < 1:
        raise ValueError("num_nodes must be positive")
    return tuple((i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes))


def ring_edges(num_nodes: int) -> tuple[Edge, ...]:
    """Return an undirected ring over facility nodes."""

    if num_nodes < 2:
        return ()
    return tuple((i, (i + 1) % num_nodes) for i in range(num_nodes))


def k_nearest_ring_edges(num_nodes: int, k: int = 2) -> tuple[Edge, ...]:
    """Return a symmetric k-nearest ring graph."""

    if num_nodes < 2 or k < 1:
        return ()
    edges: set[Edge] = set()
    for i in range(num_nodes):
        for offset in range(1, k + 1):
            j = (i + offset) % num_nodes
            if i == j:
                continue
            edges.add((min(i, j), max(i, j)))
    return tuple(sorted(edges))
