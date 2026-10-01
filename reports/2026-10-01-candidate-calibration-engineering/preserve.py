"""Non-overwriting preservation of one closed artificial acceptance packet."""

import json
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import copy_verified, create_archive, inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_calibration_engineering_20261001"
REPORT = ROOT / "reports/2026-10-01-candidate-calibration-engineering"
STORAGE = ROOT / "results/candidate_calibration_engineering_20261001_archive"
DROPBOX = Path("/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts/candidate_calibration_engineering_20261001")


def main():
    if STORAGE.exists() or DROPBOX.exists() or (REPORT / "preservation.json").exists():
        raise FileExistsError("preserve any existing snapshot; no automatic overwrite")
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    terminal = json.loads((RUN / "terminal.json").read_text())
    verified = json.loads((REPORT / "verification.json").read_text())
    assert terminal["status"] == "completed" and not terminal["engineering_passed"]
    assert verified["optimizer_calls_verified"] == 1152 and verified["passed_fixtures"] == 0
    raw = inventory(RUN)
    assert raw == verified["artifact_inventory"]
    claim = json.loads((RUN / "claim.json").read_text())
    payload = STORAGE / "payload"
    for relative, expected in raw.items():
        assert sha256_file(RUN / relative) == expected
        copy_verified(RUN / relative, payload / "acceptance" / relative)
    sources = dict(claim["source_hashes"])
    for path in list(REPORT.glob("*.py")) + list(REPORT.glob("*.json")) + [
            ROOT / "AGENTS.md", ROOT / "specs/2026-10-01-candidate-calibration-engineering/readout.md"]:
        sources[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for relative, expected in sources.items():
        assert sha256_file(ROOT / relative) == expected
        copy_verified(ROOT / relative, payload / "repository" / relative)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json_once(payload / "packet-provenance.json", {"execution_commit": claim["commit"],
        "closure_commit": commit, "raw_acceptance_hashes": raw, "repository_hashes": sources,
        "patient_calls": 0, "artificial_optimizer_calls": 1152, "engineering_passed": False})
    archive = STORAGE / "completed-artificial-acceptance.tar.gz"
    manifest = create_archive(payload, archive)
    copies = [copy_verified(path, DROPBOX / path.name) for path in (
        archive, archive.with_name(archive.name + ".manifest.json"),
        ROOT / "specs/2026-10-01-candidate-calibration-engineering/readout.md")]
    assert inventory(RUN) == raw
    receipt = {"format": "calibration-engineering-preservation-v1", "closure_commit": commit,
        "archive": str(archive), "archive_manifest": manifest, "dropbox_local_copies": copies,
        "original_acceptance_unchanged": True, "cloud_sync_verified": False, "howard_access_verified": False}
    write_json_once(REPORT / "preservation.json", receipt)
    copy_verified(REPORT / "preservation.json", DROPBOX / "preservation.json")
    print(json.dumps({"files":manifest["file_count"],"bytes":manifest["size_bytes"],
        "archive_sha256":manifest["archive_sha256"],"dropbox_local_verified":True,"cloud_sync_verified":False}))


if __name__ == "__main__":
    main()
