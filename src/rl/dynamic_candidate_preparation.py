"""Read-only preparation and source-bound *unapproved* packet construction.

This is not a numerical launcher. No historical model is loaded and no patient
environment is built. The entrypoint refuses scientific execution unconditionally.
"""

import json
from pathlib import Path
import subprocess

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_resources import audit_stream_collisions, digest
from src.rl.dynamic_candidate_specification import (
    DRAFT, build_dynamic_proposal, inspect_dynamic_inputs, static_model_counts,
)
from src.rl.dynamic_candidate_resources import dynamic_budget_plan, dynamic_stream_manifest


BRANCH = "codex/september-research-integration"
PACKET_DIRECTORY = "specs/2026-10-01-adaptive-paper-delivery/frozen-proposal"
RUN_DIRECTORY = "results/dynamic_candidate_pilot_20261001"
PROTOCOL = "specs/2026-10-01-adaptive-paper-delivery/pilot-protocol.md"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def source_locks(root):
    root = Path(root).resolve()
    names = git(root, "ls-files", "src", "experiments/scripts", "tests", "AGENTS.md").splitlines()
    if not names:
        raise ValueError("no tracked source inventory")
    return {name: file_record(root, name) for name in names if (root / name).is_file()}


def seed_evidence_inventory(root):
    """Tracked JSON configs plus locally present named seed/config/stream manifests.

    Do not rescan every historical episode or checkpoint. This scope is explicit
    and cannot establish freshness against unavailable external experiment files.
    """
    root = Path(root).resolve()
    files = set()
    for name in git(root, "ls-files", "experiments/configs").splitlines():
        path = root / name
        if path.name == ".gitkeep" and not path.is_symlink() and not path.read_bytes().strip():
            continue
        if path.suffix != ".json":
            raise ValueError(f"unhandled tracked config format in seed inventory: {name}")
        files.add(path)
    for directory in ("results", "reports", "specs"):
        base = root / directory
        if not base.exists():
            continue
        for path in base.rglob("*"):
            relative = path.relative_to(root).as_posix()
            if (relative.startswith(PACKET_DIRECTORY + "/")
                    or relative.startswith(RUN_DIRECTORY + "/") or relative == DRAFT):
                continue
            if any(word in path.name.lower() for word in ("seed", "stream", "manifest", "config")) and path.is_file():
                if path.suffix in (".json", ".jsonl"):
                    files.add(path)
                elif path.suffix in (".csv", ".yaml", ".yml", ".toml"):
                    raise ValueError("unhandled local seed declaration format")
    return sorted(files)


def prepare_proposal(root, *, freeze=False):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH:
        raise ValueError("wrong integration branch")
    if freeze and git(root, "status", "--porcelain"):
        raise ValueError("freeze requires clean committed source")
    config = build_dynamic_proposal(root)
    inputs = inspect_dynamic_inputs(root, config)
    reference = config["reference"]
    runtime_config = json.loads((root / reference["directory"] /
                                reference["config_template"].format(block=config["blocks"][0])).read_text())
    n = config["objective"]["num_facilities"]
    global_width = int(runtime_config["env"].get("include_time_state", False))
    facilities_width = config["objective"]["raw_state_width"] - global_width
    if facilities_width % n:
        raise ValueError("static public layout does not divide into facility features")
    counts = static_model_counts(n, facilities_width // n, global_width,
                                 config["model_proposal"]["encoder_width"], config["model_proposal"]["head_width"])
    config["model_proposal"].update(actor_parameter_count=counts["actor"], critic_parameter_count=counts["critic"])
    config["public_input_dimensions"] = {"facility_nodes": n, "node_features": facilities_width // n,
                                         "global_features": global_width, "message_adjacency": [n, n]}
    streams = dynamic_stream_manifest(config)
    result = {"format": "unapproved-dynamic-pilot-preparation-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": BRANCH,
        "implementation_commit": git(root, "rev-parse", "HEAD"), "source_frozen": bool(freeze),
        "protocol": file_record(root, PROTOCOL),
        "scientific_config": config, "static_inputs": inputs, "parameter_counts": counts,
        "streams": streams, "budget_plan": dynamic_budget_plan(config),
        "new_patient_builds": 0, "new_patient_steps": 0, "new_optimizer_steps": 0,
        "new_model_constructions": 0, "new_checkpoint_loads": 0}
    if freeze:
        locks = source_locks(root)
        # Bind every source byte to the named commit, not merely its working copy.
        for name, record in locks.items():
            import hashlib
            saved = subprocess.check_output(["git", "show", result["implementation_commit"] + ":" + name], cwd=root)
            if hashlib.sha256(saved).hexdigest() != record["sha256"]:
                raise ValueError("source bytes differ from implementation commit")
        evidence = seed_evidence_inventory(root)
        seed_values = {"environment": streams["environment"], "neural": streams["neural"]}
        collision = audit_stream_collisions(seed_values, evidence)
        for record in collision["files"] + collision["collisions"]:
            record["path"] = Path(record["path"]).relative_to(root).as_posix()
        if collision["passed"] is not True:
            raise ValueError("prospective stream collides with local declarations")
        result.update(source_files=locks, seed_collision_audit=collision, runtime=runtime_record(),
            freshness_scope="tracked configs plus named local seed/stream/manifest/config JSON and JSONL; unavailable external files not covered")
    result["packet_content_sha256"] = digest(result)
    return result


def reject_scientific_execution():
    raise PermissionError("New scientific execution has not been approved. This preparation entrypoint cannot launch it.")
