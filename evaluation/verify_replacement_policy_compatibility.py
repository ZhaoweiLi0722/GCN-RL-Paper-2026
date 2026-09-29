"""Independent stdlib-only arithmetic/hash verification of recorded R5 outputs."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def vector(value):
    return value if isinstance(value, list) else [value]


def check_rows(rows, expected):
    if len(rows) != 16 or [r["replay_index"] for r in rows] != [i * 1353 // 15 for i in range(16)]:
        raise ValueError("Recorded sample selection differs")
    maxima = {"actor": 0., "gate_scores": 0., "request": 0.}
    for row in rows:
        if len(row["observation"]) != 561 or row["time_coordinate"] != row["observation"][560]:
            raise ValueError("Observation/time contract differs")
        if any(not math.isfinite(x) for x in row["observation"]):
            raise ValueError("Nonfinite observation")
        a, b = row["cpu_policy"], row["mps_policy"]
        if a != row["cpu_full_state"]:
            raise ValueError("CPU full-state route differs")
        for key in maxima:
            av, bv = vector(a[key]), vector(b[key])
            width = 1 if key == "gate_scores" else 80
            if len(av) != width or len(bv) != width or any(not math.isfinite(x) for x in av + bv):
                raise ValueError("Output dimensions or finiteness differs")
            error = max(abs(x - y) for x, y in zip(av, bv))
            if error > (1e-6 if key == "request" else 1e-5):
                raise ValueError("Recorded tolerance failure")
            maxima[key] = max(maxima[key], error)
        for output in (a, b):
            if any(abs(x) > 1 for x in output["request"]):
                raise ValueError("Out-of-domain request")
            # NumPy's saved float32 multiplication, then half-away-from-zero.
            scaled = [struct.unpack("f", struct.pack("f", x * 120))[0]
                      for x in output["request"][:20]]
            lots = [int(math.copysign(math.floor(abs(x) + .5), x)) for x in scaled]
            if lots != output["requested_lots"]:
                raise ValueError("Requested-lot arithmetic differs")
            scores = vector(output["gate_scores"])
            decisions = [score >= 0 for score in scores]  # sigmoid(logit) >= 0.5
            if decisions != vector(output["hard_gate"]):
                raise ValueError("Hard gate does not match locked threshold")
        if a["hard_gate"] != b["hard_gate"] or a["requested_lots"] != b["requested_lots"]:
            raise ValueError("Device discrete outputs differ")
    margin = min(x for row in rows for x in vector(row["cpu_policy"]["gate_margin"]))
    if maxima != expected["max_cpu_mps_absolute_difference"] or margin != expected["minimum_gate_margin"]:
        raise ValueError("Summary reducer differs")
    return {"observations": len(rows), "max_cpu_mps_absolute_difference": maxima,
            "minimum_gate_margin": margin, "raw_output_arithmetic_verified": True}


def verify(root):
    root = Path(root)
    summary = json.loads((root / "summary.json").read_text())
    execution = json.loads((root / "execution.json").read_text())
    if summary["status"] != "completed" or summary["exit_code"] != 0:
        raise ValueError("Audit not complete")
    if summary["environment_steps"] or summary["optimizer_updates"]:
        raise ValueError("Unexpected scientific execution")
    for relative, sha in execution["source_locks"].items():
        if digest(relative) != sha:
            raise ValueError(f"Inference source changed: {relative}")
    config = json.loads(Path("experiments/configs/replacement_policy_compatibility_20260929.json").read_text())
    if digest(config["archive_manifest"]) != config["archive_manifest_sha256"]:
        raise ValueError("R4 manifest changed")
    manifest = json.loads(Path(config["archive_manifest"]).read_text())
    for path, sha in manifest["files"].items():
        if digest(Path(config["payload_root"]) / path) != sha:
            raise ValueError(f"R4 payload changed: {path}")
    result = {"status": "verified", "raw_observations": 0, "seeds": [],
              "new_inference_or_environment_steps": 0, "original_failure_preserved": True,
              "r4_payload_files_verified": len(manifest["files"]),
              "existing_source_files_verified": len(execution["source_locks"])}
    if [x["seed"] for x in summary["seeds"]] != [60, 61, 62]:
        raise ValueError("Wrong seed set")
    for item in summary["seeds"]:
        raw = root / f"seed{item['seed']}_raw.json"
        if digest(raw) != item["raw_sha256"]:
            raise ValueError("Raw output hash mismatch")
        info = check_rows(json.loads(raw.read_text()), item)
        result["raw_observations"] += info["observations"]
        result["seeds"].append(dict(info, seed=item["seed"], raw_sha256=digest(raw)))
    previous = execution["continuation"]
    for name, sha in previous["prior_files"].items():
        if digest(Path(previous["prior_root"]) / name) != sha:
            raise ValueError("Prior failure evidence changed")
    result["summary_sha256"] = digest(root / "summary.json")
    result["execution_sha256"] = digest(root / "execution.json")
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="results/replacement_policy_compatibility_remaining_20260929")
    parser.add_argument("--output")
    args = parser.parse_args()
    output = json.dumps(verify(args.root), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if args.output:
        with Path(args.output).open("x", encoding="utf-8") as handle:
            handle.write(output)
    print(output, end="")


if __name__ == "__main__":
    main()
