"""Source/runtime freezes and an exclusive, one-attempt P1 entrypoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import numpy as np
import torch

from experiments.scripts.audit_candidate_pilot_readiness import audit, sha, PROPOSAL, PROPOSAL_SHA, PROTOCOL_SHA
from src.rl.candidate_pilot_campaign import PatientBackend, PilotCampaign
from src.rl.candidate_pilot_compatibility import audit_reference_layouts, require_supported_layouts
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import PilotBudget, digest, stream_manifest
from src.rl.candidate_pilot_watchdog import supervise
from src.utils.research_archive import copy_verified
from src.utils.research_clock import CLOCK_ID, clock_record, shared_monotonic


EFFECTIVE = "experiments/configs/candidate_return_pilot_20260930_execution.json"
AUTHORIZATION = "specs/2026-09-30-candidate-return-pilot/authorization.md"
BRANCH = "codex/september-research-integration"


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def configure_runtime():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.set_default_dtype(torch.float32)
    torch.use_deterministic_algorithms(True)


def runtime_record():
    return {"python_executable": sys.executable, "python": sys.version, "platform": platform.platform(),
        "numpy": np.__version__, "torch": str(torch.__version__), "torch_build": torch.__config__.show(),
        "torch_threads": torch.get_num_threads(), "torch_interop_threads": torch.get_num_interop_threads(),
        "default_dtype": str(torch.get_default_dtype()), "deterministic": torch.are_deterministic_algorithms_enabled(),
        "device": "cpu", "budget_clock": clock_record(),
        "package_entry_locks": {str(Path(module.__file__).resolve()): sha(module.__file__)
                                               for module in (np, torch)}}


def source_files(root):
    names = git(root, "ls-files", "src", "experiments/scripts", "tests", "AGENTS.md").splitlines()
    return {name: sha(root / name) for name in names if (root / name).is_file()}


def freeze_packet(root):
    """Read-only packet builder; caller must commit its output before launch."""
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("freeze requires the declared clean, committed worktree")
    subset = audit(root)
    require_supported_layouts(subset["static_compatibility"])
    if not subset["collision_audit"]["passed"]:
        raise ValueError("historical seed collision")
    cfg = json.loads((root / PROPOSAL).read_text())
    return {"kind": "p1_single_attempt_effective_execution", "scientific_execution_authorized": True,
        "workspace": str(root), "branch": BRANCH, "implementation_commit": git(root, "rev-parse", "HEAD"),
        "proposal": PROPOSAL, "proposal_sha256": PROPOSAL_SHA,
        "protocol": cfg["protocol"], "protocol_sha256": PROTOCOL_SHA,
        "authorization": AUTHORIZATION, "authorization_sha256": sha(root / AUTHORIZATION),
        "scientific_config": cfg, "source_files": source_files(root), "runtime": runtime_record(),
        "readiness_audit": subset, "automatic_retry": False, "new_scope_authorized": False}


def verify_packet(root, effective, *, require_clean=True):
    root = Path(root).resolve()
    if (effective["kind"] != "p1_single_attempt_effective_execution" or effective["scientific_execution_authorized"] is not True
            or str(root) != effective["workspace"] or effective["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH):
        raise ValueError("wrong workspace/branch/authorization")
    if require_clean and git(root, "status", "--porcelain"):
        raise ValueError("scientific launch requires clean committed worktree")
    subprocess.run(["git", "merge-base", "--is-ancestor", effective["implementation_commit"], "HEAD"], cwd=root, check=True)
    if (effective["proposal"] != PROPOSAL or effective["proposal_sha256"] != PROPOSAL_SHA
            or effective["protocol_sha256"] != PROTOCOL_SHA or effective["authorization"] != AUTHORIZATION):
        raise ValueError("locked proposal/protocol/authorization identity differs")
    for name, expected in ((PROPOSAL, PROPOSAL_SHA), (effective["protocol"], PROTOCOL_SHA),
                           (AUTHORIZATION, effective["authorization_sha256"])):
        if sha(root / name) != expected:
            raise ValueError("locked protocol/config/authorization changed")
    cfg = json.loads((root / PROPOSAL).read_text())
    if effective["scientific_config"] != cfg or cfg["scientific_execution_authorized"] is not False:
        raise ValueError("original draft parameters/authorization were modified")
    if source_files(root) != effective["source_files"] or runtime_record() != effective["runtime"]:
        raise ValueError("source/runtime lock differs")
    for name, expected in effective["source_files"].items():
        content = subprocess.check_output(["git", "show", effective["implementation_commit"] + ":" + name], cwd=root)
        import hashlib
        if hashlib.sha256(content).hexdigest() != expected:
            raise ValueError("source locks not bound to actual implementation commit")
    inherited = effective["readiness_audit"]
    compatibility = audit_reference_layouts(root, cfg)
    require_supported_layouts(compatibility)
    if inherited.get("static_compatibility") != compatibility:
        raise ValueError("static compatibility receipt differs")
    if inherited["streams"] != stream_manifest(cfg) or inherited["collision_audit"]["passed"] is not True:
        raise ValueError("stream audit mismatch")
    for name, expected in inherited["verified_r4_inputs"].items():
        if sha(root / name) != expected:
            raise ValueError("inherited R4 input changed")
    for record in inherited["collision_audit"]["files"] + inherited["explicit_non_seed_parse_exclusions"]:
        if sha(root / record["path"]) != record["sha256"]:
            raise ValueError("prior audited evidence changed")
    return cfg


def launch(root, effective_path):
    root, effective_path = Path(root).resolve(), Path(effective_path).resolve()
    if effective_path != root / EFFECTIVE:
        raise ValueError("only the committed effective execution config can launch")
    effective = json.loads(effective_path.read_text())
    cfg = verify_packet(root, effective)
    if git(root, "show", "HEAD:" + EFFECTIVE) != effective_path.read_text().strip():
        raise ValueError("effective execution config is not committed at HEAD")
    refreshed = audit(root)
    require_supported_layouts(refreshed["static_compatibility"])
    if (refreshed["collision_audit"] != effective["readiness_audit"]["collision_audit"]
            or refreshed["explicit_non_seed_parse_exclusions"] != effective["readiness_audit"]["explicit_non_seed_parse_exclusions"]):
        raise ValueError("historical seed inventory changed after freeze")
    destination = Path(cfg["dropbox_directory_proposed"])
    ancestor = destination
    while not ancestor.exists():
        ancestor = ancestor.parent
    if not ancestor.is_dir() or not os.access(ancestor, os.W_OK):
        raise PermissionError("authorized local archive destination unavailable before attempt")
    output = root / cfg["output_root"]
    started = shared_monotonic()
    output.mkdir(parents=True, exist_ok=False)  # Exclusive single-attempt claim, even after failure.
    launcher = output / "launcher"
    launcher.mkdir()
    claim = {"format": "p1-exclusive-claim-v1", "pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "implementation_commit": effective["implementation_commit"],
        "effective_sha256": sha(effective_path), "started_monotonic": started, "clock_id": CLOCK_ID,
        "maximum_seconds": cfg["caps"]["maximum_seconds"], "automatic_retry": False}
    write_json_once(launcher / "claim.json", claim)
    try:
        locks = output / "payload" / "locks"
        copy_verified(effective_path, locks / "effective-execution.json")
        copy_verified(root / PROPOSAL, locks / "original-proposal.json")
        copy_verified(root / cfg["protocol"], locks / "protocol.md")
        copy_verified(root / AUTHORIZATION, locks / "authorization.md")
        subprocess.run(["git", "bundle", "create", str(locks / "source.bundle"), "HEAD"], cwd=root, check=True,
                       timeout=max(1., cfg["caps"]["maximum_seconds"] - (shared_monotonic() - started)))
        remaining = cfg["caps"]["maximum_seconds"] - (shared_monotonic() - started)
        result = supervise([sys.executable, "-m", "experiments.scripts.run_candidate_return_pilot", "--child"],
            cwd=root, stdout_path=launcher / "stdout.log", stderr_path=launcher / "stderr.log",
            ledger_path=launcher / "budget.jsonl", report_path=launcher / "supervisor.json", maximum_seconds=remaining)
        completed = launcher / "completed.json"
        success = result["passed"] and completed.is_file() and json.loads(completed.read_text())["status"] == "completed"
        write_json_once(launcher / "terminal.json", {"status": "completed" if success else "failed",
            "supervisor": result, "scientific_completion_verified": success, "automatic_retry": False})
        return 0 if success else 1
    except BaseException as exc:
        write_json_once(launcher / "launch-failure.json", {"status": "failed", "error": repr(exc), "automatic_retry": False})
        return 1


def child(root):
    root = Path(root).resolve()
    effective_path = root / EFFECTIVE
    effective = json.loads(effective_path.read_text())
    cfg = verify_packet(root, effective)
    output = root / cfg["output_root"]
    claim = json.loads((output / "launcher/claim.json").read_text())
    if (claim["pid"] != os.getppid() or claim.get("clock_id") != CLOCK_ID
            or claim["effective_sha256"] != sha(effective_path)
            or claim["head"] != git(root, "rev-parse", "HEAD") or (output / "launcher/terminal.json").exists()):
        raise ValueError("child lacks matching live exclusive parent claim")
    write_json_once(output / "launcher/child.json", {"pid": os.getpid(), "ppid": os.getppid()})
    budget = PilotBudget(output / "launcher/budget.jsonl", cfg)
    streams = stream_manifest(cfg)
    def recheck():
        if sha(effective_path) != claim["effective_sha256"] or git(root, "rev-parse", "HEAD") != claim["head"]:
            raise ValueError("execution HEAD/effective config drift")
        verify_packet(root, effective, require_clean=False)
    return PilotCampaign(output, cfg, streams, budget, PatientBackend(root, cfg, streams),
                         final_lock_check=recheck).run()
