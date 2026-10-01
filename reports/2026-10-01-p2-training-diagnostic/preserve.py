"""Archive a new diagnostic packet without touching original P2 evidence."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import copy_verified, create_archive, inventory


ROOT = Path.cwd()
REPORT = ROOT / "reports/2026-10-01-p2-training-diagnostic"
STORAGE = ROOT / "results/p2_saved_training_diagnostic_20261001_archive"
DROPBOX = Path("/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts/candidate_reference_prior_pilot_20261001/training-diagnostic-20261001")


def main():
    if STORAGE.exists() or DROPBOX.exists() or (REPORT / "preservation.json").exists():
        raise FileExistsError("preserve existing packet; no implicit retry or overwrite")
    assert not subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
    files = [f"reports/2026-10-01-p2-training-diagnostic/{name}" for name in (
        "diagnose.py", "crosscheck.py", "preserve.py", "result.json", "crosscheck.json", "acceptance.json")]
    files += [f"specs/2026-10-01-p2-training-diagnostic/{name}" for name in ("protocol.md", "readout.md")]
    files += ["tests/test_p2_training_diagnostic.py", "src/models/candidate_policy.py",
              "src/models/reference_prior_candidate.py", "src/rl/candidate_ppo_kernel.py",
              "src/rl/candidate_ppo_objective.py", "src/rl/candidate_rollout.py"]
    payload = STORAGE / "payload"
    for relative in files:
        copy_verified(ROOT / relative, payload / relative)
    source_files = inventory(payload)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json_once(payload / "provenance.json", {
        "commit": commit, "source_files": source_files,
        "original_scientific_root": "results/candidate_reference_prior_pilot_20261001",
        "new_scientific_execution": False, "created_utc": datetime.now(timezone.utc).isoformat()})
    archive = STORAGE / "p2-training-diagnostic.tar.gz"
    manifest = create_archive(payload, archive)
    copies = [copy_verified(p, DROPBOX / p.name) for p in (
        archive, archive.with_name(archive.name + ".manifest.json"),
        ROOT / "specs/2026-10-01-p2-training-diagnostic/readout.md")]
    receipt = {"format": "p2-training-diagnostic-preservation-v1", "commit": commit,
               "archive": str(archive), "archive_manifest": manifest, "dropbox_local_copies": copies,
               "original_source_files_unchanged": all((payload / p).read_bytes() == (ROOT / p).read_bytes() for p in files),
               "cloud_sync_verified": False, "howard_access_verified": False}
    assert receipt["original_source_files_unchanged"]
    write_json_once(REPORT / "preservation.json", receipt)
    copy_verified(REPORT / "preservation.json", DROPBOX / "preservation.json")
    print(json.dumps({"file_count":manifest["file_count"], "archive_sha256":manifest["archive_sha256"],
                      "dropbox_local_copies_verified":True,"cloud_sync_verified":False,"howard_access_verified":False}))


if __name__ == "__main__":
    main()
