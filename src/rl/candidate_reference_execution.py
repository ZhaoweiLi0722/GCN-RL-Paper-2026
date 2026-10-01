"""P2 implementation/input/runtime freeze; no implicit reuse of P1 authority."""

import hashlib
from pathlib import Path
import subprocess

from src.rl import candidate_reference_spec as spec
from src.rl.candidate_pilot_execution import git, runtime_record, source_files, BRANCH
from src.rl.candidate_pilot_compatibility import audit_reference_layouts, require_supported_layouts
from src.rl.candidate_pilot_resources import stream_manifest


def freeze_packet(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("P2 freeze requires committed clean declared worktree")
    cfg, audit = spec.configuration(root), spec.audit(root)
    require_supported_layouts(audit["static_compatibility"])
    if not audit["collision_audit"]["passed"]:
        raise ValueError("P2 historical stream collision")
    return {"kind": "p2_reference_prior_single_attempt_effective_execution", "scientific_execution_authorized": True,
        "workspace": str(root), "branch": BRANCH, "implementation_commit": git(root, "rev-parse", "HEAD"),
        "document_locks": {name: spec.sha(root / name) for name in (
            spec.DESIGN, spec.PROTOCOL, spec.AUTHORIZATION, spec.PROPOSAL,
            "specs/2026-09-30-candidate-return-pilot/protocol.md")},
        "scientific_config": cfg, "source_files": source_files(root), "runtime": runtime_record(),
        "readiness_audit": audit, "automatic_retry": False}


def verify_packet(root, packet, *, require_clean=True):
    root = Path(root).resolve()
    if (packet["kind"] != "p2_reference_prior_single_attempt_effective_execution"
            or packet["scientific_execution_authorized"] is not True or packet["automatic_retry"] is not False
            or packet["workspace"] != str(root) or packet["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH):
        raise ValueError("P2 workspace/branch/authority mismatch")
    if require_clean and git(root, "status", "--porcelain"):
        raise ValueError("P2 launch requires clean committed worktree")
    cfg = spec.configuration(root)
    expected_docs = {spec.DESIGN, spec.PROTOCOL, spec.AUTHORIZATION, spec.PROPOSAL,
                     "specs/2026-09-30-candidate-return-pilot/protocol.md"}
    if (set(packet["document_locks"]) != expected_docs or any(spec.sha(root / name) != value
            for name, value in packet["document_locks"].items()) or packet["scientific_config"] != cfg):
        raise ValueError("P2 protocol/authorization/fixed scientific scope drift")
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    if source_files(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("P2 source/runtime differs")
    for name, expected in (packet["source_files"] | packet["document_locks"]).items():
        raw = subprocess.check_output(["git", "show", packet["implementation_commit"] + ":" + name], cwd=root)
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("P2 lock differs from actual implementation commit")
    audit = packet["readiness_audit"]
    compatibility = audit_reference_layouts(root, cfg)
    require_supported_layouts(compatibility)
    if (audit["static_compatibility"] != compatibility or audit["streams"] != stream_manifest(cfg)
            or audit["collision_audit"]["passed"] is not True or audit["collision_audit"]["collisions"]
            or audit["prior_failures"] != spec.prior_evidence(root)):
        raise ValueError("P2 input/seed/prior-failure audit mismatch")
    for name, expected in audit["verified_r4_inputs"].items():
        if spec.sha(root / name) != expected:
            raise ValueError("P2 inherited R4 input changed")
    for record in audit["collision_audit"]["files"] + audit["explicit_non_seed_parse_exclusions"]:
        if spec.sha(root / record["path"]) != record["sha256"]:
            raise ValueError("P2 historical evidence changed")
    return cfg
