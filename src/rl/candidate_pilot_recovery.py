"""One explicit P1-R1 amendment, not a general retry or parameter override API."""

import copy
import json
from pathlib import Path

from src.rl.candidate_pilot_resources import read_ledger, stream_manifest
from src.utils.research_archive import inventory, sha256_file


AMENDMENT = "experiments/configs/candidate_return_pilot_20260930_recovery1.json"
AUTHORIZATION = "specs/2026-09-30-candidate-return-pilot/recovery1_authorization.md"
EFFECTIVE = "experiments/configs/candidate_return_pilot_20260930_recovery1_execution.json"


def recovery_configuration(root, original):
    amendment = json.loads((Path(root) / AMENDMENT).read_text())
    if (amendment["kind"] != "p1_recovery1_specimen_graph_amendment"
            or amendment["candidate_message_graph"] != "specimen_routes"
            or amendment["output_suffix"] != "_recovery1" or amendment["authorization"] != AUTHORIZATION
            or amendment["automatic_retry"] is not False
            or amendment["stream_handling"] != "retain_original_allocation_disclose_preflight_ordinal0_initialization_only"):
        raise ValueError("unapproved recovery amendment")
    cfg = copy.deepcopy(original)
    # Nothing else in the scientific proposal is replaceable by this amendment.
    cfg["candidate_message_graph"] = amendment["candidate_message_graph"]
    cfg["output_root"] += amendment["output_suffix"]
    cfg["dropbox_directory_proposed"] += amendment["output_suffix"]
    return cfg


def prior_attempt_receipt(root, original):
    root = Path(root)
    amendment = json.loads((root / AMENDMENT).read_text())
    manifest_path = root / amendment["prior_manifest"]
    if sha256_file(manifest_path) != amendment["prior_manifest_sha256"]:
        raise ValueError("prior failure manifest changed")
    manifest = json.loads(manifest_path.read_text())
    archive = manifest_path.parent / manifest["archive"]
    if sha256_file(archive) != amendment["prior_archive_sha256"] or manifest["archive_sha256"] != amendment["prior_archive_sha256"]:
        raise ValueError("prior failure archive changed")
    prior_root = root / original["output_root"]
    if inventory(prior_root) != manifest["files"]:
        raise ValueError("prior failed evidence changed")
    ledger = read_ledger(prior_root / "launcher/budget.jsonl")
    terminal = json.loads((prior_root / "launcher/terminal.json").read_text())
    if (ledger["counts"] or ledger["events"] != 2 or ledger["last_sha256"] != amendment["prior_ledger_sha256"]
            or terminal["status"] != "failed"):
        raise ValueError("recovery assumes only the verified zero-step terminal failure")
    return {"format": "p1_recovery1_prior_attempt_v1", "prior_root": original["output_root"],
        "manifest_sha256": amendment["prior_manifest_sha256"], "archive_sha256": amendment["prior_archive_sha256"],
        "files": manifest["files"], "ledger": ledger,
        "previously_initialized_preflight_seed": stream_manifest(original)["environment"]["preflight"][0],
        "previous_environment_constructions": 1, "previous_environment_steps": 0,
        "previous_optimizer_steps": 0, "previous_completed_episodes": 0,
        "demonstration_qualification_training_test_streams_previously_unconsumed": True,
        "preflight_ordinal0_is_not_claimed_fresh": True,
        "no_reward_based_selection_or_new_namespace": True}
