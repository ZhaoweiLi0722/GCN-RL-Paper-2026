"""Clock portability/failure tests, with no simulator or optimizer."""

import unittest
from unittest.mock import patch

from src.utils import research_clock


class ResearchClockTests(unittest.TestCase):
    def test_posix_clock_is_explicit_and_finite(self):
        self.assertGreaterEqual(research_clock.shared_monotonic(), 0.)
        record = research_clock.clock_record()
        self.assertEqual(record["id"], research_clock.CLOCK_ID)
        self.assertGreater(record["resolution_seconds"], 0.)

    def test_unsupported_runtime_cannot_fall_back_to_process_local_clock(self):
        for error in (AttributeError("missing"), OSError("unavailable")):
            with patch.object(research_clock.time, "clock_gettime", side_effect=error):
                with self.assertRaisesRegex(RuntimeError, "CLOCK_MONOTONIC"):
                    research_clock.shared_monotonic()

    def test_invalid_clock_values_and_resolution_fail_closed(self):
        for value in (float("nan"), float("inf"), -1.):
            with patch.object(research_clock.time, "clock_gettime", return_value=value):
                with self.assertRaises(ValueError):
                    research_clock.shared_monotonic()
        with patch.object(research_clock.time, "clock_getres", return_value=0.):
            with self.assertRaises(ValueError):
                research_clock.clock_record()
