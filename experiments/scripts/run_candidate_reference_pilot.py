"""One approved P2 attempt. No retry, resume, override or tuning options."""

import argparse
from pathlib import Path

from src.rl.candidate_pilot_execution import child, configure_runtime, launch
from src.rl.candidate_reference_spec import EFFECTIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    return child(root, reference_prior=True) if args.child else launch(root, root / EFFECTIVE, reference_prior=True)


if __name__ == "__main__":
    raise SystemExit(main())
