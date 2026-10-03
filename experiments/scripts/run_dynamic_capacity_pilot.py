"""Scope-frozen dynamic capacity pilot; no unapproved smoke/retry mode."""

import argparse
from pathlib import Path

from src.rl.capacity_pilot_execution import AUTH, FROZEN, authorization, child, freeze, launch
from src.rl.paired_cohort_execution import write_json_once


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze", action="store_true")
    group.add_argument("--launch", action="store_true")
    group.add_argument("--child", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.freeze:
        packet = freeze(root)
        write_json_once(root/FROZEN,packet)
        write_json_once(root/AUTH,authorization(packet))
        print(packet["packet_sha256"])
        return 0
    return child(root) if args.child else launch(root)


if __name__ == "__main__":
    raise SystemExit(main())
