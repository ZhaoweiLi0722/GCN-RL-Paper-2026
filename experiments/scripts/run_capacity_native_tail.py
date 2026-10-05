"""One approved native-tail comparison; no resume/search/retry mode."""

import argparse
from pathlib import Path

from src.rl.capacity_native_tail_execution import AUTH, FROZEN, authorization, child, configure_runtime, freeze, launch
from src.rl.paired_cohort_execution import write_json_once


def main():
    parser=argparse.ArgumentParser()
    mode=parser.add_mutually_exclusive_group(required=True)
    for name in ("freeze","launch","child"):
        mode.add_argument("--"+name,action="store_true")
    args=parser.parse_args()
    configure_runtime()
    root=Path(__file__).resolve().parents[2]
    if args.freeze:
        packet=freeze(root)
        write_json_once(root/FROZEN,packet)
        write_json_once(root/AUTH,authorization(packet))
        print(packet["packet_sha256"])
        return 0
    return child(root) if args.child else launch(root)


if __name__=="__main__":
    raise SystemExit(main())
