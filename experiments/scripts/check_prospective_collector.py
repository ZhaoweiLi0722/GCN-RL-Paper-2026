"""Run only the bounded N4 engineering fixture and write a new, immutable report."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from src.rl.patient_replay_collector import json_value
from src.rl.prospective_engineering_check import run_engineering_check
from src.rl.networks import torch


def local_source_hashes(modules, root):
    sources = {}
    for module in modules:
        name = getattr(module, "__file__", None)
        # Torch dynamic namespaces expose relative pseudo filenames (e.g. _ops.py).
        # Only absolute loader filenames establish local file provenance.
        if name and Path(name).is_absolute():
            path = Path(name).resolve()
            if path.suffix == ".py" and path.is_relative_to(root):
                sources[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return dict(sorted(sources.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output already exists; never overwrite prior evidence")
    expected_config = ROOT / "experiments/configs/prospective_collector_engineering_20260929.json"
    if args.config.resolve() != expected_config:
        parser.error("use the committed N4 fixture config, not a scientific experiment config")
    dirty = subprocess.check_output([
        "git", "status", "--porcelain", "--untracked-files=all", "--", "src",
        str(Path(__file__).relative_to(ROOT)), str(expected_config.relative_to(ROOT)),
        "specs/2026-09-29-prospective-collector-engineering/protocol.md",
    ], cwd=ROOT, text=True)
    if dirty.strip():
        parser.error("commit source/config/protocol before the recorded engineering run")
    config = json.loads(args.config.read_text())
    result = run_engineering_check(config)
    result["execution_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    result["config_file_sha256"] = hashlib.sha256(args.config.read_bytes()).hexdigest()
    # Capture loaded local code, including transitive environment/heuristic helpers.
    result["source_sha256"] = local_source_hashes(tuple(sys.modules.values()), ROOT)
    result["runtime"] = {"python": sys.version.split()[0], "numpy": np.__version__, "torch": torch.__version__}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, default=json_value, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output), "cases": result["case_count"],
                      "decisions": result["decision_count"], "optimizer_updates": result["optimizer_updates"]}))


if __name__ == "__main__":
    main()
