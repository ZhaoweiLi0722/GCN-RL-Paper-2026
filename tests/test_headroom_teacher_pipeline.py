from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from evaluation.run_headroom_teacher_pipeline import (
    build_shard_command,
    claim_phase,
)


class HeadroomTeacherPipelineTests(unittest.TestCase):
    def test_state_probe_command_is_cpu_shard_only(self) -> None:
        command = build_shard_command(
            "/python",
            "experiments/configs/teacher.json",
            "state-probe",
            2,
            10,
        )

        self.assertIn("--skip-teacher", command)
        self.assertIn("--state-probe-shard-index", command)
        self.assertNotIn("--teacher-shard-index", command)

    def test_teacher_command_is_teacher_shard_only(self) -> None:
        command = build_shard_command(
            "/python",
            "experiments/configs/teacher.json",
            "teacher",
            1,
            3,
        )

        self.assertIn("--skip-state-probe", command)
        self.assertIn("--teacher-shard-index", command)
        self.assertNotIn("--state-probe-shard-index", command)

    def test_phase_claim_is_single_use(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            claim_phase(root, "state-probe", {"state": "claimed"})

            with self.assertRaises(FileExistsError):
                claim_phase(root, "state-probe", {"state": "claimed"})


if __name__ == "__main__":
    unittest.main()
