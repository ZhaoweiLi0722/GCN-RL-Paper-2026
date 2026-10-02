"""Prepare metadata or explicitly run one separately approved paired-cohort attempt."""

import time

STARTED = time.clock_gettime(time.CLOCK_MONOTONIC)

import argparse
import json
from pathlib import Path

from src.rl import paired_cohort_execution as execution


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    for name in ("prepare", "freeze", "authorize", "launch", "child"):
        group.add_argument("--" + name, action="store_true")
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    if args.prepare or not any((args.freeze, args.authorize, args.launch, args.child)):
        print(json.dumps(execution.prepare(root), sort_keys=True, allow_nan=False))
        return 0
    from src.rl.candidate_pilot_execution import configure_runtime
    configure_runtime()
    if args.freeze:
        packet = execution.freeze(root)
        execution.write_json_once(root / execution.FROZEN, packet)
        print(packet["packet_sha256"])
        return 0
    if args.authorize:
        execution._clean(root)
        packet = execution.committed_json(root, execution.FROZEN)
        execution.verify_bindings(root, packet)
        execution.write_json_once(root / execution.AUTHORIZATION, execution.authorization(packet))
        return 0
    return execution.child(root) if args.child else execution.launch(root, STARTED)


if __name__ == "__main__":
    raise SystemExit(main())
