"""Run the single prospective pilot only after its exact approval is committed."""

import argparse
from pathlib import Path

from src.rl.dynamic_candidate_execution import child, configure_runtime, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    return child(root) if args.child else launch(root)


if __name__ == "__main__":
    raise SystemExit(main())
