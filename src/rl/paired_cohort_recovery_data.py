"""Frozen remaining-work inventory and write-once import of completed labels."""

import copy
import json
from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.paired_cohort_collection import validate_context
from src.utils.research_archive import copy_verified


OLD_RUN = "results/paired_cohort_improvement_20261002"
OLD_PACKET = "specs/2026-10-02-paired-cohort-improvement/frozen.json"


def branch_schedule(contexts, cfg, streams):
    result = []
    for block in cfg["blocks"]:
        for cohort in range(cfg["context_cohorts_per_block"]):
            for t in cfg["context_after_prefix_steps"]:
                context, bank = validate_context(contexts[block, cohort, t])
                if (context["block"], context["cohort"], context["after_prefix_steps"]) != (block, cohort, t):
                    raise ValueError("foreign recovery context identity")
                if context["source_id"] != streams["namespace"]:
                    raise ValueError("context belongs to a different prospective stream")
                future = streams["conditional_future"][f"block{block}/cohort{cohort}/after{t}"]
                if len(future) != cfg["future_replications"]:
                    raise ValueError("future replication inventory changed")
                for rep, seed in enumerate(future):
                    for candidate in range(len(bank.class_keys)):
                        result.append(dict(block=block, cohort=cohort, after_prefix_steps=t,
                            replication=rep, candidate_index=candidate, future_seed=str(seed),
                            environment_calls=cfg["economic_endpoint"] - t,
                            branch_id=f"branches/block{block}/cohort{cohort}/after{t}/rep{rep}/class{candidate}"))
    return result


def recovery_manifest(workspace):
    """Read saved training data, never neural models or any test outcomes."""
    workspace = Path(workspace).resolve()
    old = workspace / OLD_RUN
    packet = json.loads((workspace / OLD_PACKET).read_text())
    cfg, streams = packet["config"], packet["streams"]
    contexts, records = {}, []
    for b in cfg["blocks"]:
        for c in range(cfg["context_cohorts_per_block"]):
            for t in cfg["context_after_prefix_steps"]:
                path = old / f"payload/contexts/block{b}/cohort{c}/after{t}.pt"
                contexts[b, c, t] = load_envelope(path)
                records.append(dict(key=[b, c, t], file=file_record(workspace, path)))
    schedule = branch_schedule(contexts, cfg, streams)
    completed, unique = [], set()
    for path in sorted((old / "launcher/branch-boundaries/block60").glob("*.json")):
        boundary = json.loads(path.read_text())
        name, files = boundary["branch_id"], boundary["index"]["files"]
        if name in unique or set(files) != {"header.json", "initial.pt", "events.jsonl", "states.json", "final.pt", "receipt.json", "verified.json"}:
            raise ValueError("duplicate or incomplete imported branch")
        unique.add(name)
        if any(file_record(old, record["path"]) != record for record in files.values()):
            raise ValueError("saved branch bytes changed")
        header = json.loads((old / files["header.json"]["path"]).read_text())
        manifest = header["manifest"]
        row = schedule[len(completed)]
        if any(str(manifest[k]) != str(row[k]) for k in ("branch_id", "block", "cohort", "after_prefix_steps", "replication", "candidate_index", "future_seed")):
            raise ValueError("saved completed branches are not the original exact prefix")
        completed.append(dict(branch_id=name, boundary=file_record(workspace, path), files=files))
    if len(records) != 36 or len(completed) != 117 or len(schedule) != 402:
        raise ValueError("frozen saved-data inventory differs from approved recovery")
    remaining = schedule[len(completed):]
    counts = {str(b): dict(branches=sum(r["block"] == b for r in remaining),
        environment_calls=sum(r["environment_calls"] for r in remaining if r["block"] == b)) for b in cfg["blocks"]}
    if counts != {"60": {"branches": 11, "environment_calls": 297}, "61": {"branches": 140, "environment_calls": 6020}, "62": {"branches": 134, "environment_calls": 5730}}:
        raise ValueError("remaining work differs from the explicitly approved package")
    interrupted = old / "payload" / remaining[0]["branch_id"]
    if sorted(p.name for p in interrupted.iterdir()) != ["events.jsonl", "header.json", "initial.pt"]:
        raise ValueError("interrupted branch inventory changed")
    result = dict(format="paired-cohort-recovery-data-v1", old_run=OLD_RUN,
        original_packet=file_record(workspace, OLD_PACKET), contexts=records, completed=completed,
        interrupted=[file_record(workspace, p) for p in sorted(interrupted.iterdir())],
        remaining=remaining, remaining_by_block=counts, old_environment_charge=6246,
        preserved_interrupted_charge=1, new_environment_charge=25655,
        cumulative_environment_charge=31901, original_attempt_remains_terminal=True)
    result["manifest_sha256"] = digest(result)
    return result


def verify_data_manifest(workspace, manifest):
    body = copy.deepcopy(manifest)
    sha = body.pop("manifest_sha256")
    if digest(body) != sha or body["format"] != "paired-cohort-recovery-data-v1":
        raise ValueError("recovery data manifest digest differs")
    workspace = Path(workspace).resolve()
    records = [body["original_packet"]] + [r["file"] for r in body["contexts"]] + body["interrupted"]
    for item in body["completed"]:
        records.append(item["boundary"])
        records.extend(dict(record, path=body["old_run"] + "/" + record["path"]) for record in item["files"].values())
    for record in records:
        if file_record(workspace, record["path"]) != record:
            raise ValueError("immutable recovery source changed: " + record["path"])
    return True


def import_saved_data(workspace, root, manifest, *, admit, budget):
    """Copy completed bytes only; partial evidence remains in the old root."""
    workspace, root = Path(workspace).resolve(), Path(root).resolve()
    contexts, indexes = {}, []
    old = workspace / manifest["old_run"]
    for item in manifest["contexts"]:
        budget.check()
        admit("context_load")
        record = item["file"]
        source = workspace / record["path"]
        if file_record(workspace, source) != record:
            raise ValueError("context changed after freeze")
        relative = source.relative_to(old)
        copy_verified(source, root / relative)
        saved = load_envelope(root / relative)
        context, _ = validate_context(saved)
        key = tuple(item["key"])
        if key != (context["block"], context["cohort"], context["after_prefix_steps"]) or key in contexts:
            raise ValueError("context import duplicated or changed identity")
        contexts[key] = saved
    for item in manifest["completed"]:
        budget.check()
        admit("branch_import")
        boundary = item["boundary"]
        if file_record(workspace, boundary["path"]) != boundary:
            raise ValueError("completed boundary changed after freeze")
        index = json.loads((workspace / boundary["path"]).read_text())["index"]
        if index["files"] != item["files"]:
            raise ValueError("import file inventory differs")
        for record in item["files"].values():
            if file_record(old, record["path"]) != record:
                raise ValueError("completed branch changed after freeze")
            target = root / record["path"]
            if target.exists():
                raise FileExistsError(target)
            copy_verified(old / record["path"], target)
        indexes.append(index)
    write_json_once(root / "payload/reused-data.json", dict(manifest_sha256=manifest["manifest_sha256"],
        contexts=len(contexts), complete_branches=len(indexes), reused_environment_calls=5111,
        preserved_old_environment_charge=manifest["old_environment_charge"],
        preserved_interrupted_charge=1, partial_branch_imported=False, new_simulator_calls=0))
    return contexts, indexes
