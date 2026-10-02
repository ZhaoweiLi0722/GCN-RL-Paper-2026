"""Freeze preparation or run the separately approved continuation exactly once."""

import time

STARTED = time.clock_gettime(time.CLOCK_MONOTONIC)

import argparse
from pathlib import Path

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_recovery_execution import FROZEN, child, configure_runtime, freeze, launch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--freeze", action="store_true")
    mode.add_argument("--child", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    configure_runtime()
    if args.freeze:
        packet = freeze(root)
        write_json_once(root / FROZEN, packet)
        print(packet["packet_sha256"])
        return 0
    return child(root) if args.child else launch(root, started=STARTED)


if __name__ == "__main__":
    raise SystemExit(main())
