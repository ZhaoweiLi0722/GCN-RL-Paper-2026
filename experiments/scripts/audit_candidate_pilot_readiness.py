"""Read-only P1 input/stream audit. Does not construct an environment or learner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_resources import audit_stream_collisions, stream_manifest


PROPOSAL = "experiments/configs/candidate_return_pilot_20260930.json"
PROPOSAL_SHA = "fe7d1cc4227e86c0d16360e6b285748e5b4be268c19270b9df29afc442eb5052"
PROTOCOL_SHA = "75c9c47f484dafd345d9e1c670a81330469ed3cf986f8a80864e5cd6ee3f027b"
KNOWN_PARTIAL_NON_SEED_REPORTS = {
    "reports/2026-09-30-critic-saved-data-diagnosis/diagnosis.json":
        "5713c90bbf396d3448447cfab2de9a5ca7626f0bb43795d1df4f0170905473ff",
    "results/critic_saved_data_diagnosis_archive_20260930/payload/reports/2026-09-30-critic-saved-data-diagnosis/diagnosis.json":
        "5713c90bbf396d3448447cfab2de9a5ca7626f0bb43795d1df4f0170905473ff",
}


def sha(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def historical_files(root):
    tracked = subprocess.check_output(["git", "ls-files", "-z", "experiments/configs"], cwd=root).decode().split("\0")
    files = []
    for name in tracked:
        if not name or name == PROPOSAL or name == "experiments/configs/candidate_return_pilot_20260930_execution.json":
            continue
        path = root / name
        if path.suffix == ".json":
            files.append(path)
        elif path.suffix in (".yaml", ".yml", ".toml", ".jsonl"):
            raise ValueError(f"unhandled tracked config format: {path}")
    for directory in (root / "results", root / "reports"):
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            rel = path.relative_to(root)
            if rel.parts[1] in ("candidate_return_pilot_20260930", "2026-09-30-candidate-pilot-integration"):
                continue
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                files.append(path)
            elif path.is_file() and any(term in path.name.lower() for term in ("seed", "stream", "manifest")):
                if path.suffix in (".csv", ".yaml", ".yml", ".toml"):
                    raise ValueError(f"unhandled historical seed/manifest format: {path}")
    return sorted(set(files))


def audit(root):
    root = Path(root).resolve()
    if sha(root / PROPOSAL) != PROPOSAL_SHA:
        raise ValueError("proposal changed")
    config = json.loads((root / PROPOSAL).read_text())
    if sha(root / config["protocol"]) != PROTOCOL_SHA:
        raise ValueError("protocol changed")
    reference = config["reference"]
    inputs = {reference["archive_manifest"]: reference["archive_manifest_sha256"]}
    environments = []
    for block in config["blocks"]:
        for kind, template in (("config", "config_template"), ("policy", "policy_template")):
            name = str(Path(reference["directory"]) / reference[template].format(seed=block))
            inputs[name] = reference["locks"][str(block)][kind]
            if kind == "config":
                environments.append(json.loads((root / name).read_text())["env"])
    for name, expected in inputs.items():
        if sha(root / name) != expected:
            raise ValueError(f"R4 inherited input hash changed: {name}")
    if any(env != environments[0] for env in environments[1:]):
        raise ValueError("R4 effective environments differ")
    manifest = stream_manifest(config)
    files, exclusions = [], []
    for path in historical_files(root):
        name = str(path.relative_to(root))
        if name in KNOWN_PARTIAL_NON_SEED_REPORTS:
            if sha(path) != KNOWN_PARTIAL_NON_SEED_REPORTS[name]:
                raise ValueError(f"preserved partial report changed: {name}")
            # The inspected 183-byte prefix contains only aggregate score fields.
            # Its complete diagnosis.v2.json is included; do not repair the old file.
            exclusions.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size,
                               "reason": "known preserved truncated non-seed score report; complete v2 scanned"})
        else:
            files.append(path)
    collision = audit_stream_collisions(manifest, files)
    for row in collision["files"] + collision["collisions"]:
        row["path"] = str(Path(row["path"]).relative_to(root))
    return {"kind": "read_only_p1_readiness_subset_not_execution_acceptance",
            "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
            "proposal_sha256": PROPOSAL_SHA, "protocol_sha256": PROTOCOL_SHA, "verified_r4_inputs": inputs,
            "effective_environments_equal": True, "streams": manifest, "collision_audit": collision,
            "explicit_non_seed_parse_exclusions": exclusions,
            "new_environment_constructions": 0, "new_environment_steps": 0, "new_scientific_updates": 0,
            "scope": "tracked JSON configs and all locally present prior results/reports JSON/JSONL; not unavailable external evidence",
            "ready_to_execute": False,
            "remaining": ["complete committed orchestration and effective execution config",
                          "independent outcome verification and source/runtime manifest", "budgeted real preflight"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = audit(args.root)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
    print(json.dumps({"verified_r4_inputs": len(result["verified_r4_inputs"]),
                      "historical_files": len(result["collision_audit"]["files"]),
                      "collisions": result["collision_audit"]["collisions"],
                      "ready_to_execute": False}, sort_keys=True))
    return 0 if result["collision_audit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
