"""Single locked P1 attempt. No resume, retry, override or tuning flags."""

import argparse
from pathlib import Path

from src.rl.candidate_pilot_execution import EFFECTIVE, child, configure_runtime, launch
from src.rl.candidate_pilot_recovery import EFFECTIVE as RECOVERY_EFFECTIVE


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--recovery1", action="store_true", help="Only the explicitly approved specimen-graph recovery")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    return (child(root, recovery=args.recovery1) if args.child else
            launch(root, root / (RECOVERY_EFFECTIVE if args.recovery1 else EFFECTIVE), recovery=args.recovery1))


if __name__ == "__main__":
    raise SystemExit(main())
