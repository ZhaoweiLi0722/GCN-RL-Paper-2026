"""Prepare or run only the separately approved saved-model qualification."""

import argparse
from pathlib import Path

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_saved_execution import FROZEN, child, configure_runtime, freeze, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--prepare", action="store_true", help="hash bindings only; no checkpoint loads")
    mode.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    if args.prepare:
        write_json_once(root / FROZEN, freeze(root))
        return 0
    return child(root) if args.child else launch(root)


if __name__ == "__main__":
    raise SystemExit(main())
