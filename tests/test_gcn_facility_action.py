"""Test-gate for the facility_action readout and curriculum warm-start (Lever 2).

The lean pilot undertrained the default ``global_flat`` graph actor at 20-clinic
scale (head input ~ num_facilities x encoder_dim). The ``facility_action`` readout
applies a shared per-facility head: fewer parameters, permutation-equivariant, and
size-invariant so a small-network policy transfers to a larger one (2->20 curriculum
warm-start). Before adopting it in the campaign we certify two things here:

  1. Layout: facility_action emits a type-major action vector [comp0 x N, comp1 x N,
     ...], which matches the environment's decoding of the normalized action as
     [w(0:N), e(N:2N), q(2N:3N), p(3N:4N)] blocks. (No hidden facility<->type permutation.)
  2. Transfer: every learnable weight of a facility_action actor is size-invariant, so
     transfer_matching_parameters copies the full policy across graph sizes, skipping
     only the non-learnable adjacency buffer. global_flat, by contrast, can transfer
     only the encoder.
"""

from __future__ import annotations

import unittest

try:
    from src.rl.networks import torch
except Exception:  # pragma: no cover
    torch = None

NODE_DIM = 9
GCN_HIDDEN = (16, 16)
HEAD_HIDDEN = (32,)


def _line_edges(n):
    return tuple((i, i + 1) for i in range(n - 1))


@unittest.skipIf(torch is None, "torch not available")
class FacilityActionLayoutTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(0)

    def _actor(self, n, readout_mode, include_global_context=True):
        from src.models.gcn import GCNActor

        return GCNActor(
            NODE_DIM,
            n,               # num_facilities
            n,               # num_nodes (no hub in this unit test)
            4 * n,           # action_dim = facility_net (w, e, q, p) per facility
            _line_edges(n),
            GCN_HIDDEN,
            HEAD_HIDDEN,
            include_global_context=include_global_context,
            readout_mode=readout_mode,
        )

    def test_facility_action_shape_and_bounds(self) -> None:
        n = 4
        actor = self._actor(n, "facility_action")
        out = actor(torch.randn(5, n, NODE_DIM))
        self.assertEqual(out.shape, (5, 4 * n))
        self.assertTrue(torch.all(out.abs() <= 1.0))  # tanh-squashed

    def test_layout_is_type_major_matching_env(self) -> None:
        """Output block t (indices t*N : (t+1)*N) == head component t across facilities,
        i.e. the same [w x N, e x N, q x N, p x N] type-major layout the env decodes."""
        n = 3
        actor = self._actor(n, "facility_action")
        nodes = torch.randn(6, n, NODE_DIM)
        out = actor(nodes)

        # Manually reconstruct the per-facility head outputs.
        encoded = actor.encoder(nodes)
        facility = encoded[:, :n, :]
        ctx = encoded.mean(dim=1, keepdim=True).expand(-1, n, -1)
        per_facility = actor.head(torch.cat((facility, ctx), dim=-1))  # (6, n, 4)

        # Reshaping the flat action to (batch, 4 types, n facilities) must recover,
        # for each type t, that type's head component across all facilities.
        reshaped = out.view(6, 4, n)
        for t in range(4):
            self.assertTrue(
                torch.allclose(reshaped[:, t, :], per_facility[:, :, t], atol=1e-6),
                f"type block {t} does not match head component {t}",
            )


@unittest.skipIf(torch is None, "torch not available")
class CurriculumTransferTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(1)

    def _actor(self, n, readout_mode):
        from src.models.gcn import GCNActor

        return GCNActor(
            NODE_DIM, n, n, 4 * n, _line_edges(n), GCN_HIDDEN, HEAD_HIDDEN,
            readout_mode=readout_mode,
        )

    def test_facility_action_params_are_size_invariant(self) -> None:
        small = self._actor(2, "facility_action")
        large = self._actor(20, "facility_action")
        s, l = small.state_dict(), large.state_dict()
        learnable = [k for k in s if k != "encoder.adjacency"]
        for k in learnable:
            self.assertIn(k, l)
            self.assertEqual(s[k].shape, l[k].shape, f"{k} shape differs across graph sizes")
        # Only the adjacency buffer is size-dependent.
        self.assertNotEqual(s["encoder.adjacency"].shape, l["encoder.adjacency"].shape)

    def test_warm_start_transfers_full_policy_2_to_20(self) -> None:
        from src.models.gcn import transfer_matching_parameters

        small = self._actor(2, "facility_action")
        large = self._actor(20, "facility_action")
        # Perturb the small policy so it is distinct from large's fresh init.
        with torch.no_grad():
            for p in small.parameters():
                p.add_(torch.randn_like(p))

        summary = transfer_matching_parameters(small.state_dict(), large)

        self.assertIn("encoder.adjacency", summary["skipped"])
        self.assertNotIn("encoder.adjacency", summary["transferred"])
        self.assertTrue(any("head" in k for k in summary["transferred"]))
        self.assertTrue(any("layers" in k for k in summary["transferred"]))

        # Every learnable weight of the 20-clinic actor now equals the 2-clinic one.
        s, l = small.state_dict(), large.state_dict()
        for k, v in l.items():
            if k == "encoder.adjacency":
                continue
            self.assertTrue(torch.allclose(v, s[k], atol=1e-6), f"{k} not transferred")

        # The warm-started 20-clinic actor still emits valid, in-range actions.
        out = large(torch.randn(3, 20, NODE_DIM))
        self.assertEqual(out.shape, (3, 80))
        self.assertTrue(torch.all(out.abs() <= 1.0))

    def test_global_flat_transfers_encoder_only(self) -> None:
        from src.models.gcn import transfer_matching_parameters

        small = self._actor(2, "global_flat")
        large = self._actor(20, "global_flat")
        summary = transfer_matching_parameters(small.state_dict(), large)
        # Encoder weights are node-independent and transfer; the flat head width
        # scales with facility count, so head params cannot transfer.
        self.assertTrue(any("layers" in k for k in summary["transferred"]))
        self.assertTrue(any("head" in k for k in summary["skipped"]))


if __name__ == "__main__":
    unittest.main()
