"""Copy the completed supplement only; never launch or rearchive science."""

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

BASE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(BASE))
from src.utils.research_archive import copy_verified, sha256_file

RUN = BASE / "results/capacity_fixed_reference_20261008"
SPEC = BASE / "specs/2026-10-08-mdl2-fixed-reference"
OUT = Path(__file__).resolve().parent
DEST = Path(
    "/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/"
    "GCN-DRL Paper 2026/Research Artifacts/capacity_family_selection_20261006/"
    "supplement-mdl2-fixed2-20261008"
)


def deliver(mode):
    if mode not in ("artifacts", "closure"):
        raise ValueError("Expected artifacts or closure")
    terminal = json.loads((RUN / "terminal.json").read_text())
    if terminal["status"] != "completed" or not terminal["scientific_completion_verified"]:
        raise ValueError("Only this completed supplement may be delivered")
    sources = {}
    if mode == "artifacts":
        for name in ("payload.tar.gz", "payload.tar.gz.manifest.json"):
            sources["archive/" + name] = RUN / "archives" / name
        sources["archive/archive-receipt.json"] = RUN / "archive-receipt.json"
        for name in ("protocol.md", "approval-intent.json", "frozen.json",
                     "integration-readout.md"):
            sources["authority/" + name] = SPEC / name
        for name in ("claim.json", "terminal.json"):
            sources["execution/" + name] = RUN / name
        for name in ("comparison.json", "completion.json", "progress.jsonl"):
            sources["readout/" + name] = RUN / "payload" / name
        sources["readout/terminal-readout.md"] = SPEC / "terminal-readout.md"
        sources["readout/export_saved.py"] = Path(__file__).resolve()
    else:
        for path in (SPEC / "terminal-readout.md", SPEC / "dropbox-handoff.md",
                     BASE / "docs/team_updates/autonomous-local-work-queue.md",
                     BASE / "docs/patient_indexed_specimen_routing_locked_execution_plan.md",
                     BASE / "specs/2026-10-01-adaptive-paper-delivery/workflow.json",
                     OUT / "delivery-artifacts.json"):
            sources["closure/" + path.name] = path

    receipt_path = OUT / ("delivery-" + mode + ".json")
    if receipt_path.exists():
        raise FileExistsError("Delivery receipt already exists; inspect it instead of rerunning")
    entries = []
    for relative, source in sources.items():
        expected_bytes = source.stat().st_size
        target = DEST / relative
        record = copy_verified(source, target)
        if target.stat().st_size != expected_bytes or source.stat().st_size != expected_bytes:
            raise ValueError("Source or destination byte count changed")
        if sha256_file(target) != record["sha256"] or sha256_file(source) != record["sha256"]:
            raise ValueError("Source or destination digest changed")
        entries.append(dict(source=str(source), destination=str(target),
                            bytes=expected_bytes, sha256=record["sha256"]))
    receipt = dict(
        at_utc=datetime.now(timezone.utc).isoformat(), mode=mode, destination=str(DEST),
        local_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=BASE, text=True).strip(),
        working_tree_clean=not bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=BASE, text=True).strip()),
        scientific_completion_verified=True, local_copy_verified=True,
        cloud_sync_verified=False, collaborator_access_verified=False,
        overwrote_different_files=False, permission_changes=False,
        external_messages_sent=False, existing_single_archive_reused=True,
        files=len(entries), bytes=sum(e["bytes"] for e in entries), entries=entries,
    )
    with receipt_path.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    copied = copy_verified(receipt_path, DEST / receipt_path.name)
    if Path(copied["path"]).stat().st_size != receipt_path.stat().st_size:
        raise ValueError("Delivery receipt byte count mismatch")
    print(json.dumps({k: v for k, v in receipt.items() if k != "entries"}, sort_keys=True))


if __name__ == "__main__":
    deliver(sys.argv[1])
