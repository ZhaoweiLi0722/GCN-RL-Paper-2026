"""Inspect or freeze the unapproved dynamic pilot; scientific launch is disabled."""

import argparse
import json
from pathlib import Path

from src.rl.candidate_pilot_recording import write_json_once
from src.rl.dynamic_candidate_preparation import (
    configure_runtime, prepare_proposal, reject_scientific_execution,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run", action="store_true", help="Always rejected: no scientific authorization exists")
    args = parser.parse_args()
    if args.run:
        reject_scientific_execution()
    configure_runtime()
    result = prepare_proposal(args.root, freeze=args.freeze)
    if args.output:
        destination = args.output if args.output.is_absolute() else args.root / args.output
        root = args.root.resolve()
        if destination.absolute() != destination.resolve() or not destination.resolve().is_relative_to(root):
            raise ValueError("proposal evidence must stay in the worktree without redirects")
        write_json_once(destination, result)
    print(json.dumps({"source_frozen": result["source_frozen"], "ready_to_launch": False,
                      "scientific_execution_authorized": False, "parameter_counts": result["parameter_counts"],
                      "static_inputs": len(result["static_inputs"]["inputs"]),
                      "seed_inventory_files": len(result.get("seed_collision_audit", {}).get("files", [])),
                      "patient_environment_calls": 0, "optimizer_steps": 0}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
