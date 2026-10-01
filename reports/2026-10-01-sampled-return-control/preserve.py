"""Archive this completed artificial packet locally without changing run bytes."""

import json
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import copy_verified, create_archive, inventory, sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_sampled_return_control_20261001"
LAUNCHER = ROOT / "results/candidate_sampled_return_control_20261001-launcher"
REPORT = ROOT / "reports/2026-10-01-sampled-return-control"
SPEC = ROOT / "specs/2026-10-01-sampled-return-control"
STORAGE = ROOT / "results/candidate_sampled_return_control_20261001_archive"


def main():
    if STORAGE.exists() or (REPORT / "preservation.json").exists():
        raise FileExistsError("preserve existing snapshots; no overwrite")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise RuntimeError("commit closure/readout before preservation")
    terminal = json.loads((RUN / "terminal.json").read_text())
    verified = json.loads((REPORT / "verification.json").read_text())
    assert terminal["status"] == "completed" and terminal["exit_code"] == 0
    assert verified["optimizer_charges_verified"] == 2304
    assert verified["training_observations_verified"] == 13824
    assert verified["passed_fixtures"] == 1 and verified["total_fixtures"] == 9
    assert terminal["engineering_passed"] is False
    raw, logs = inventory(RUN), inventory(LAUNCHER)
    assert raw == verified["artifact_inventory"]
    assert (LAUNCHER / "stderr.log").stat().st_size == 0
    assert (LAUNCHER / "verification.stderr.log").stat().st_size == 0
    claim = json.loads((RUN / "claim.json").read_text())
    payload = STORAGE / "payload"
    for source, hashes, destination in ((RUN, raw, "acceptance"), (LAUNCHER, logs, "launcher")):
        for relative, expected in hashes.items():
            assert sha256_file(source / relative) == expected
            copy_verified(source / relative, payload / destination / relative)
    sources = dict(claim["source_hashes"])
    for path in list(REPORT.glob("*.py")) + list(REPORT.glob("*.json")) + [
            SPEC / "readout.md", SPEC / "execution_authorization.json"]:
        relative = path.relative_to(ROOT).as_posix()
        digest = sha256_file(path)
        if relative in sources and sources[relative] != digest:
            raise ValueError("locked repository file changed")
        sources[relative] = digest
    for relative, expected in sources.items():
        assert sha256_file(ROOT / relative) == expected
        copy_verified(ROOT / relative, payload / "repository" / relative)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    write_json_once(payload / "packet-provenance.json", {
        "execution_commit": claim["execution_commit"], "closure_commit": commit,
        "raw_acceptance_hashes": raw, "launcher_hashes": logs, "repository_hashes": sources,
        "patient_calls": 0, "artificial_optimizer_calls": 2304,
        "training_observations": 13824, "passed_fixtures": 1, "engineering_passed": False})
    archive = STORAGE / "completed-sampled-return-control.tar.gz"
    manifest = create_archive(payload, archive)
    assert inventory(RUN) == raw and inventory(LAUNCHER) == logs
    receipt = {"format": "sampled-return-preservation-v1", "closure_commit": commit,
        "archive": str(archive), "archive_manifest": manifest, "dropbox_local_copies": [],
        "dropbox_copy_status": "not_authorized_in_this_packet",
        "original_acceptance_unchanged": True, "original_launcher_unchanged": True,
        "cloud_sync_verified": False, "howard_access_verified": False}
    write_json_once(REPORT / "preservation.json", receipt)
    print(json.dumps({"files": manifest["file_count"], "bytes": manifest["size_bytes"],
        "archive_sha256": manifest["archive_sha256"], "dropbox_local_verified": False,
        "cloud_sync_verified": False}))


if __name__ == "__main__":
    main()
