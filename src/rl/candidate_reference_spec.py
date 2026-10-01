"""Fixed approved P2 delta and read-only readiness, never parameter search."""

import hashlib
import json
from pathlib import Path
import subprocess

from experiments.scripts.audit_candidate_pilot_readiness import (
    PROPOSAL, PROPOSAL_SHA, PROTOCOL_SHA, KNOWN_PARTIAL_NON_SEED_REPORTS, sha)
from src.rl.candidate_pilot_compatibility import audit_reference_layouts
from src.rl.candidate_pilot_resources import audit_stream_collisions, stream_manifest, read_ledger
from src.utils.research_archive import inventory

DESIGN = "experiments/configs/candidate_reference_prior_design_20261001.json"
DESIGN_SHA = "0c7f99e045a77862a78a6a5c39e763683dcf7a1ed35296f95157e690952a5298"
PROTOCOL = "specs/2026-10-01-reference-prior-residual/protocol.md"
AUTHORIZATION = "specs/2026-10-01-reference-prior-residual/execution_authorization.md"
EFFECTIVE = "experiments/configs/candidate_reference_prior_pilot_20261001_execution.json"
OUTPUT = "results/candidate_reference_prior_pilot_20261001"


def configuration(root):
    root = Path(root)
    if sha(root / DESIGN) != DESIGN_SHA or sha(root / PROPOSAL) != PROPOSAL_SHA:
        raise ValueError("immutable P1/P2 design changed")
    cfg = json.loads((root / PROPOSAL).read_text())
    if sha(root / cfg["protocol"]) != PROTOCOL_SHA:
        raise ValueError("immutable P1 protocol changed")
    cfg.update(kind="p2_reference_prior_scientific_configuration", pilot_profile="p2_reference_prior",
        status="conditionally_authorized_by_separate_effective_packet", protocol=PROTOCOL,
        question="simulation_return_training_beyond_same_start_reference_prior", candidate_message_graph="specimen_routes")
    cfg["initialization"] = {"kind": "analytical_reference_prior_no_fit", "nonreference_mass": .1,
        "qualification_episodes_per_block": 2, "optimizer_steps_per_model": 0,
        "qualification": "exact_greedy_reference_zero_residual_and_value_probability_tolerance_8_float32_eps"}
    namespace = "P2-reference-prior-candidate-20261001-v1"
    cfg["rng"].update(namespace=namespace, ordinal_ranges={"preflight": [0, 11], "qualification": [12, 17],
        "training": [18, 113], "test": [114, 149]}, neural_role_paths=[
            "block{b}/{representation}/continuation/sample", "block{b}/{representation}/ppo/shuffle",
            "block{b}/{representation}/bc_continue/shuffle", "analysis/bootstrap"])
    cfg["policy_init_seeds"] = [int.from_bytes(hashlib.sha256(
        f"{namespace}/block{b}/model_initialization".encode()).digest()[:8], "big") % 2**63 for b in cfg["blocks"]]
    cfg["rng"]["model_initialization_derivation"] = "sha256(namespace+'/block'+str(block)+'/model_initialization')[:8] big modulo2**63"
    cfg["rng"]["declared_aliases"] = [
        "one initialization stream/block; graph/self_only identical tensors; flat same seed but different shape",
        "one continuation sample stream/block/representation; FROZEN/PPO/BC fork independent generators at same state",
        "training world starts paired across representations and continued roles within block",
        "test world starts paired across all11 policies within block; frozen policies duplicate R4",
        "preflight uses disposable sampler copies and never advances continuation template RNG"]
    cfg["caps"]["environment_steps"].pop("demonstrations")
    cfg["caps"]["optimizer_steps"].pop("initialization")
    for name in ("demonstrations_total", "initialization_per_model"):
        cfg["caps"]["seconds"].pop(name)
    cfg["caps"].update(maximum_environment_steps=51480, maximum_optimizer_steps=2304)
    cfg["readiness"] = {"authority": AUTHORIZATION, "effective_packet_required": True}
    cfg["output_root"] = OUTPUT
    cfg["dropbox_directory_proposed"] = str(Path(cfg["dropbox_directory_proposed"]).parent / Path(OUTPUT).name)
    return cfg


def historical_files(root):
    """Include P1 and P1-R1; exclude only this prospective P2 namespace packet."""
    root = Path(root).resolve()
    files = set()
    tracked = subprocess.check_output(["git", "ls-files", "-z", "experiments/configs"], cwd=root).decode().split("\0")
    for name in tracked:
        if not name or name in (DESIGN, EFFECTIVE):
            continue
        path = root / name
        if path.suffix in (".json", ".jsonl"):
            files.add(path)
        elif path.suffix in (".yaml", ".yml", ".toml", ".csv"):
            raise ValueError("unhandled historical config format: " + name)
    for directory in ("results", "reports"):
        for path in (root / directory).rglob("*"):
            relative = path.relative_to(root)
            if relative.parts[1] in (Path(OUTPUT).name, "2026-10-01-reference-prior-integration"):
                continue
            if path.is_file() and path.suffix in (".json", ".jsonl"):
                files.add(path)
            elif path.is_file() and any(s in path.name.lower() for s in ("seed", "stream", "manifest")):
                if path.suffix in (".csv", ".yaml", ".yml", ".toml"):
                    raise ValueError("unhandled historical seed format: " + str(relative))
    return sorted(files)


def prior_evidence(root):
    root = Path(root)
    cases = [
        ("candidate_return_pilot_20260930", "failed-attempt-3de710b.tar.gz",
         "3d29ccdff06ff52cb7b2fe1280f0ec5015d7fef252101fca3522da06ee3c877c",
         "96f0c53f60e20e9d35b104fb2942615968da90a78318b8bb0d61d2281d449392", {}),
        ("candidate_return_pilot_20260930_recovery1", "failed-attempt-010ca27.tar.gz",
         "801567b3ff7e5609a964a03761cdf594a8aa196c1ca6fa7f3c8f2259c875a21b",
         "2563b39fec1c04d46a13420ed9fd00572b2146615d632f9b060841138339e645", {"environment": 2064, "optimizer": 2304})]
    result = []
    for name, archive, archive_sha, manifest_sha, counts in cases:
        source = root / "results" / name
        path = root / "results" / (name + "_failure_archive") / archive
        manifest = path.with_name(path.name + ".manifest.json")
        if sha(path) != archive_sha or sha(manifest) != manifest_sha:
            raise ValueError("prior failed archive/manifest changed")
        data = json.loads(manifest.read_text())
        if inventory(source) != data["files"] or data["archive_sha256"] != archive_sha:
            raise ValueError("prior failed tree changed")
        ledger = read_ledger(source / "launcher/budget.jsonl")
        if ledger["counts"] != counts or json.loads((source / "launcher/terminal.json").read_text())["status"] != "failed":
            raise ValueError("prior failure status/budget changed")
        result.append({"root": str(source.relative_to(root)), "files": data["files"],
            "archive_sha256": archive_sha, "manifest_sha256": manifest_sha, "ledger": ledger})
    return result


def audit(root):
    root = Path(root).resolve()
    cfg = configuration(root)
    ref = cfg["reference"]
    inputs = {ref["archive_manifest"]: ref["archive_manifest_sha256"]}
    environments = []
    for block in cfg["blocks"]:
        for kind in ("policy", "config"):
            name = str(Path(ref["directory"]) / ref[kind + "_template"].format(seed=block))
            inputs[name] = ref["locks"][str(block)][kind]
            if kind == "config":
                environments.append(json.loads((root / name).read_text())["env"])
    if any(sha(root / name) != value for name, value in inputs.items()) or any(e != environments[0] for e in environments):
        raise ValueError("locked R4 inputs/effective environment mismatch")
    files, exclusions = [], []
    for path in historical_files(root):
        name = str(path.relative_to(root))
        if name in KNOWN_PARTIAL_NON_SEED_REPORTS:
            if sha(path) != KNOWN_PARTIAL_NON_SEED_REPORTS[name]:
                raise ValueError("preserved partial historical report changed")
            exclusions.append({"path": name, "sha256": sha(path), "bytes": path.stat().st_size,
                "reason": "preserved truncated non-seed score report; complete v2 included"})
        else:
            files.append(path)
    streams = stream_manifest(cfg)
    collision = audit_stream_collisions(streams, files)
    for row in collision["files"] + collision["collisions"]:
        row["path"] = str(Path(row["path"]).relative_to(root))
    return {"kind": "p2_read_only_readiness", "streams": streams, "collision_audit": collision,
        "verified_r4_inputs": inputs, "static_compatibility": audit_reference_layouts(root, cfg),
        "prior_failures": prior_evidence(root), "explicit_non_seed_parse_exclusions": exclusions,
        "new_environment_steps": 0, "new_optimizer_steps": 0,
        "scope": "all tracked configs and locally present historical results/reports JSON and JSONL; unavailable external files not verified"}
