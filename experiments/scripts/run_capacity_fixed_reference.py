"""Prepare or run the separately authorized fixed-staffing supplement once."""

import argparse
from pathlib import Path

from src.rl.capacity_fixed_reference import prepare, run
from src.rl.capacity_value_comparison_execution import configure_runtime


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--prepare", action="store_true")
    mode.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    prepare(root) if args.prepare else run(root)
