"""Cross-platform paths in transferred training manifests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.evaluate_multiscenario_network_residual import (
    aggregate_holdout_results,
    resolve_checkpoint_variants,
    resolve_manifest_artifact_path,
)


class MultiscenarioCheckpointPathTests(unittest.TestCase):
    def test_resolves_windows_separators_on_posix(self) -> None:
        workspace = Path.cwd().resolve()
        with tempfile.TemporaryDirectory(
            dir=workspace,
        ) as temporary_directory:
            checkpoint = (
                Path(temporary_directory) / "nested" / "checkpoint.pt"
            )
            checkpoint.parent.mkdir(parents=True)
            checkpoint.write_bytes(b"checkpoint")
            relative = checkpoint.relative_to(workspace)
            windows_path = str(relative).replace("/", "\\")

            resolved = resolve_manifest_artifact_path(windows_path)
            variants = resolve_checkpoint_variants(
                {
                    "checkpoint": windows_path,
                    "pretrain_checkpoint": windows_path,
                },
                ("pretrain",),
            )

            self.assertEqual(resolved.resolve(), checkpoint.resolve())
            self.assertEqual(
                variants["pretrain"].resolve(),
                checkpoint.resolve(),
            )

    def test_matches_new_conservative_graph_and_flat_algorithms(
        self,
    ) -> None:
        graph = "gcn_residual_mdl2_network_td3_bc"
        flat = "flat_residual_mdl2_network_td3_bc"
        holdout = {
            (graph, 0): [{"tag": "graph"}],
            (flat, 0): [{"tag": "flat"}],
        }
        anchors = {
            (graph, 0): [{"tag": "anchor"}],
            (flat, 0): [{"tag": "anchor"}],
        }

        def fake_bundle(candidate, baseline, *, seed):
            return {
                "candidate": candidate[0]["tag"],
                "baseline": baseline[0]["tag"],
                "seed": seed,
            }

        with patch(
            "evaluation.evaluate_multiscenario_network_residual."
            "metric_bootstrap_bundle",
            side_effect=fake_bundle,
        ):
            result = aggregate_holdout_results(
                holdout,
                anchors,
                bootstrap_seed=10,
            )

        comparison = result[f"{graph}_vs_{flat}"]
        self.assertEqual(comparison["candidate"], "graph")
        self.assertEqual(comparison["baseline"], "flat")
        self.assertEqual(result["graph_vs_flat"], comparison)


if __name__ == "__main__":
    unittest.main()
