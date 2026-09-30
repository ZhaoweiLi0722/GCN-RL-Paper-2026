"""Single locked P1 attempt. No resume, retry, override or tuning flags."""

import argparse
from pathlib import Path

from src.rl.candidate_pilot_execution import EFFECTIVE, child, configure_runtime, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    return child(root) if args.child else launch(root, root / EFFECTIVE)


if __name__ == "__main__":
    raise SystemExit(main())
