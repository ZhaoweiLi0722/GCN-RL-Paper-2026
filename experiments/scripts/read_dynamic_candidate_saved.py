"""Recompute already saved S1 receipts without research execution."""

import argparse
import json
from pathlib import Path
import subprocess

from src.rl.dynamic_candidate_saved_readout import build_saved_readout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--proposal", required=True, type=Path)
    parser.add_argument("--preservation-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.resolve().is_relative_to(args.source.resolve()):
        raise ValueError("use a new report outside the immutable source")
    result = build_saved_readout(args.source, args.proposal, args.preservation_manifest)
    result["reader_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("episodes", "raw_rows", "source_unchanged")} |
                     {"decision": result["qualification"]["decision"], "output": str(args.output)}))


if __name__ == "__main__":
    main()
