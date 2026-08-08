from __future__ import annotations

import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from evaluation.headroom_teacher_bundle import (
    ARTIFACT_NAMES,
    create_teacher_bundle,
    extract_teacher_bundle,
    sha256_file,
    verify_teacher_bundle,
)
from evaluation.run_gcn_residual_sweep import (
    save_local_search_demonstrations,
)
from src.rl.experiment import write_rows


class HeadroomTeacherBundleTests(unittest.TestCase):
    def test_bundle_round_trip_preserves_canonical_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = self._write_teacher_root(source)
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            bundle = root / "teacher.zip"
            provenance = {
                "git_commit": "test-commit",
                "config_sha256": sha256_file(config_path),
                "source_sha256": {},
            }

            created = create_teacher_bundle(
                source,
                bundle,
                expected_config=config,
                provenance=provenance,
            )
            verified = verify_teacher_bundle(bundle)

            self.assertEqual(
                created["bundle_sha256"],
                verified["bundle_sha256"],
            )
            destination = root / "imported"
            with patch(
                "evaluation.headroom_teacher_bundle.git_output",
                return_value="test-commit",
            ):
                extract_teacher_bundle(
                    bundle,
                    destination,
                    repo_root=root,
                    config_path=config_path,
                )
            for name in ARTIFACT_NAMES:
                self.assertEqual(
                    (source / name).read_bytes(),
                    (destination / name).read_bytes(),
                )

    def test_verify_rejects_bundle_missing_canonical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source"
            source.mkdir()
            config = self._write_teacher_root(source)
            bundle = root / "teacher.zip"
            create_teacher_bundle(
                source,
                bundle,
                expected_config=config,
                provenance={"git_commit": "test-commit"},
            )

            malformed = root / "missing-summary.zip"
            with zipfile.ZipFile(bundle) as archive:
                payloads = {
                    name: archive.read(name) for name in archive.namelist()
                }
            manifest = json.loads(payloads["manifest.json"])
            manifest["artifacts"] = [
                artifact
                for artifact in manifest["artifacts"]
                if artifact["archive_path"] != "teacher/summary.json"
            ]
            payloads["manifest.json"] = (
                json.dumps(manifest, indent=2, sort_keys=True) + "\n"
            ).encode("utf-8")
            payloads.pop("teacher/summary.json")
            with zipfile.ZipFile(malformed, "x") as archive:
                for name, payload in payloads.items():
                    archive.writestr(name, payload)
            Path(f"{malformed}.sha256").write_text(
                f"{sha256_file(malformed)}  {malformed.name}\n"
            )

            with self.assertRaisesRegex(ValueError, "canonical artifact set"):
                verify_teacher_bundle(malformed)

    @staticmethod
    def _write_teacher_root(source: Path) -> dict:
        config = {
            "state_probe_rollouts": 1,
            "teacher_replications": 1,
            "store_option_advantages": True,
        }
        write_rows(
            [
                {
                    "rollout": 0,
                    "step": 0,
                    "selected_group": "anchor",
                    "score_improvement": 0.0,
                    "lookahead_cost_improvement": 0.0,
                }
            ],
            source / "state_probe.csv",
        )
        row = {
            "replication": 0,
            "total_cost": 1.0,
            "completion_service_level": 1.0,
            "patients_lost": 0.0,
            "patient_ineligibility_during_manufacturing_rate": 0.0,
        }
        write_rows([row], source / "anchor.csv")
        write_rows([row], source / "teacher.csv")
        save_local_search_demonstrations(
            source / "teacher_cache.npz",
            {
                "states": np.zeros((1, 2), dtype=np.float32),
                "actions": np.zeros((1, 1), dtype=np.float32),
                "weights": np.ones(1, dtype=np.float32),
                "improved_mask": np.zeros(1, dtype=bool),
                "transition_states": np.zeros((1, 2), dtype=np.float32),
                "transition_actions": np.zeros((1, 1), dtype=np.float32),
                "transition_rewards": np.zeros(1, dtype=np.float32),
                "transition_next_states": np.zeros((1, 2), dtype=np.float32),
                "transition_dones": np.ones(1, dtype=bool),
                "option_advantages": np.zeros((1, 1), dtype=np.float32),
                "option_feasible": np.ones((1, 1), dtype=bool),
                "option_groups": np.asarray(["anchor"]),
                "option_epsilons": np.zeros(1, dtype=np.float32),
                "option_signs": np.zeros(1, dtype=np.float32),
                "improved_steps": 0,
                "anchor_keep_steps": 1,
                "service_rejected_steps": 0,
                "mean_step_improvement": 0.0,
                "improved_weight_fraction": 0.0,
            },
        )
        summary = {
            "config": config,
            "state_probe": {"states": 1},
            "state_probe_shards": {"states": 1, "count": 1},
            "online_teacher": {
                "teacher_demonstration_samples": 1,
                "teacher_total_decisions": 1,
            },
            "teacher_shards": {"replications": 1, "count": 1},
            "decision": {"advance_to_network_residual_training": True},
        }
        (source / "summary.json").write_text(json.dumps(summary))
        return config


if __name__ == "__main__":
    unittest.main()
