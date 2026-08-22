import csv
import tempfile
import unittest
from pathlib import Path

from evaluation.aggregate_results import read_rows


class AggregateResultsTests(unittest.TestCase):
    def test_read_rows_accepts_large_routing_event_fields(self):
        large_event_log = "x" * 200_000
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "routing_rows.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("scenario", "specimen_route_events_json"),
                )
                writer.writeheader()
                writer.writerow(
                    {
                        "scenario": "routing_nominal_history",
                        "specimen_route_events_json": large_event_log,
                    }
                )

            rows = read_rows([path])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["scenario"], "routing_nominal_history")
        self.assertEqual(
            rows[0]["specimen_route_events_json"],
            large_event_log,
        )


if __name__ == "__main__":
    unittest.main()
