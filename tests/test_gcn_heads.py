"""Component tests for the graph network heads (Phase 6, group 3).

These certify the *new* surface — the GCN encoder ∘ backbone-head composition —
directly: shapes, action bounds, log-prob shape, stochasticity, adjacency
structure, and gradient flow through the graph. The RL update math itself is
already LQR-certified on the flat backbones (Phase 5).
"""

from __future__ import annotations

import unittest

try:
    from src.rl.networks import torch
except Exception:  # pragma: no cover
    torch = None

NODE_DIM = 9
N = 3
NUM_NODES = 3
ACTION_DIM = 12  # facility_net: 4 * N
EDGES = ((0, 1), (1, 2))
GCN_HIDDEN = (16, 16)
HEAD_HIDDEN = (32,)


@unittest.skipIf(torch is None, "torch not available")
class GraphHeadTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)
        self.nodes = torch.randn(4, NUM_NODES, NODE_DIM)

    def _squashed_actor(self):
        from src.models.gcn import GCNSquashedGaussianActor

        return GCNSquashedGaussianActor(
            NODE_DIM, N, NUM_NODES, ACTION_DIM, EDGES, GCN_HIDDEN, HEAD_HIDDEN
        )

    def _gaussian_actor(self):
        from src.models.gcn import GCNGaussianActor

        return GCNGaussianActor(
            NODE_DIM, N, NUM_NODES, ACTION_DIM, EDGES, GCN_HIDDEN, HEAD_HIDDEN
        )

    def test_extractor_output_dim(self) -> None:
        from src.models.gcn import GraphFeatureExtractor

        ext = GraphFeatureExtractor(NODE_DIM, N, NUM_NODES, EDGES, GCN_HIDDEN)
        # global_flat readout: N facility rows + one mean-context row, each width 16.
        self.assertEqual(ext.output_dim, (N + 1) * GCN_HIDDEN[-1])
        self.assertEqual(ext(self.nodes).shape, (4, ext.output_dim))

    def test_squashed_actor_shapes_and_bounds(self) -> None:
        actor = self._squashed_actor()
        action, log_prob, mean_action = actor.sample(self.nodes)
        self.assertEqual(action.shape, (4, ACTION_DIM))
        self.assertEqual(log_prob.shape, (4, 1))
        self.assertTrue(torch.all(action > -1.0) and torch.all(action < 1.0))
        self.assertTrue(torch.all(mean_action.abs() < 1.0))
        det = actor.deterministic(self.nodes)
        self.assertEqual(det.shape, (4, ACTION_DIM))

    def test_gaussian_actor_log_prob_and_entropy(self) -> None:
        actor = self._gaussian_actor()
        action, log_prob, entropy = actor.sample(self.nodes)
        self.assertEqual(action.shape, (4, ACTION_DIM))
        self.assertEqual(log_prob.shape, (4, 1))
        self.assertEqual(entropy.shape, (4, 1))
        # Recomputed log-prob on the same action matches the sampled one.
        recomputed = actor.log_prob(self.nodes, action)
        self.assertEqual(recomputed.shape, (4, 1))
        self.assertTrue(torch.isfinite(recomputed).all())

    def test_value_head_scalar(self) -> None:
        from src.models.gcn import GCNValue

        value = GCNValue(NODE_DIM, N, NUM_NODES, EDGES, GCN_HIDDEN, HEAD_HIDDEN)
        self.assertEqual(value(self.nodes).shape, (4, 1))

    def test_stochastic_differs_from_deterministic(self) -> None:
        actor = self._squashed_actor()
        torch.manual_seed(1)
        sampled, _, _ = actor.sample(self.nodes)
        det = actor.deterministic(self.nodes)
        self.assertFalse(torch.allclose(sampled, det))

    def test_adjacency_symmetric_with_self_loops(self) -> None:
        from src.models.gcn import build_normalized_adjacency

        adj = build_normalized_adjacency(NUM_NODES, EDGES)
        self.assertTrue(torch.allclose(adj, adj.t()))
        self.assertTrue(torch.all(torch.diagonal(adj) > 0.0))  # self-loops present

    def test_gradient_flows_through_encoder(self) -> None:
        actor = self._squashed_actor()
        _, log_prob, _ = actor.sample(self.nodes)
        log_prob.sum().backward()
        enc_grads = [
            p.grad for p in actor.extractor.encoder.parameters() if p.requires_grad
        ]
        self.assertTrue(enc_grads and all(g is not None for g in enc_grads))
        self.assertTrue(any(g.abs().sum() > 0 for g in enc_grads))


if __name__ == "__main__":
    unittest.main()
