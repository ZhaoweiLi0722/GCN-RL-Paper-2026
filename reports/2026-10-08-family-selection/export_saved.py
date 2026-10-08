"""Additive delivery of this completed run to its sole authorized destination."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys

BASE = Path(__file__).resolve().parents[2]
RUN = BASE / "results/capacity_family_selection_20261006"
SPEC = BASE / "specs/2026-10-06-family-selection"
DEST = Path("/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts/capacity_family_selection_20261006")
OUT = Path(__file__).resolve().parent
MODE = sys.argv[1]
assert MODE in ("artifacts", "closure")
sources = {}


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(2**20), b""):
            h.update(chunk)
    return h.hexdigest()


def add(src, relative):
    src = Path(src)
    assert src.is_file() and not src.is_symlink(), str(src)
    assert relative not in sources
    sources[relative] = src


if MODE == "artifacts":
    for name in ("payload.tar.gz", "payload.tar.gz.manifest.json"):
        add(RUN / "archives" / name, "archive/" + name)
    add(RUN / "archive-receipt.json", "archive/archive-receipt.json")
    add(RUN / "payload/artifact-inventory.json", "archive/artifact-inventory.json")
    for path in sorted((RUN / "launcher").iterdir()):
        add(path, "launcher/" + path.name)
    add(BASE / "reports/capacity_family_selection_20261006_detached_launcher.log", "launcher/detached-launcher.log")
    for name in ("protocol.md", "approval-intent.json", "frozen.json", "authorization.json", "integration-readout.md", "launch-readout.md"):
        add(SPEC / name, "authority/" + name)
    frozen = json.loads((SPEC / "frozen.json").read_text())
    for group, directory in (("source_files", "frozen-source"), ("inputs", "bound-inputs")):
        for item in frozen[group].values():
            src = BASE / item["path"]
            assert sha(src) == item["sha256"] and src.stat().st_size == item["bytes"]
            add(src, "authority/" + directory + "/" + item["path"])
    for path in sorted((RUN / "terminal-readout").glob("*.json")):
        add(path, "readout/" + path.name)
    for name in ("read_saved.py", "export_saved.py", "terminal-process-and-validation.json"):
        add(OUT / name, "readout/" + name)
    add(SPEC / "terminal-readout.md", "readout/terminal-readout.md")
    paper = BASE / "paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks/main.tex"
    add(paper, "manuscript/main.tex")
else:
    for path in (SPEC / "terminal-readout.md", SPEC / "dropbox-handoff.md",
                 BASE / "docs/team_updates/autonomous-local-work-queue.md",
                 BASE / "docs/patient_indexed_specimen_routing_locked_execution_plan.md",
                 BASE / "specs/2026-10-01-adaptive-paper-delivery/workflow.json",
                 OUT / "delivery-artifacts.json", OUT / "automation-paused.json",
                 OUT / "local-commit.json"):
        add(path, "closure/" + path.name)

entries = []
for index, (relative, source) in enumerate(sources.items(), 1):
    target = DEST / relative
    expected_size, expected_hash = source.stat().st_size, sha(source)
    if target.exists():
        assert not target.is_symlink()
        assert target.stat().st_size == expected_size and sha(target) == expected_hash, "Refusing different existing bytes: " + str(target)
        action = "reused_identical"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as incoming, target.open("xb") as outgoing:
            shutil.copyfileobj(incoming, outgoing, length=2**20)
        action = "copied_additively"
    assert target.stat().st_size == expected_size and sha(target) == expected_hash
    entries.append(dict(source=str(source), destination=str(target), bytes=expected_size, sha256=expected_hash, action=action))
    if index <= 3 or index % 100 == 0:
        print("verified destination", index, relative, flush=True)

receipt = dict(at_utc=datetime.now(timezone.utc).isoformat(), destination=str(DEST), mode=MODE,
    local_copy_verified=True, cloud_sync_verified=False, collaborator_access_verified=False,
    overwrote_different_files=False, permission_changes=False, external_messages_sent=False,
    existing_single_archive_reused=True, files=len(entries), bytes=sum(e["bytes"] for e in entries), entries=entries)
encoded = (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode()
local = OUT / ("delivery-" + MODE + ".json")
with local.open("xb") as stream:
    stream.write(encoded)
target = DEST / ("delivery-" + MODE + ".json")
with target.open("xb") as stream:
    stream.write(encoded)
assert sha(target) == sha(local) and target.stat().st_size == len(encoded)
print(json.dumps({k:v for k,v in receipt.items() if k != "entries"}), flush=True)
