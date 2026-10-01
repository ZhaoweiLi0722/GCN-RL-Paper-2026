"""One-shot, separately approved saved-qualification execution and preparation.

Preparation hashes files without deserializing models. Only the approved owned
child loads four checkpoints. No environment, optimizer or campaign is created.
"""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_execution import configure_runtime, runtime_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.candidate_pilot_watchdog import supervise
from src.rl.dynamic_candidate_execution import committed_json, confined
from src.rl.dynamic_candidate_preparation import BRANCH, git, source_locks
from src.utils.research_clock import CLOCK_ID, shared_monotonic


DIRECTORY = "specs/2026-10-01-adaptive-paper-delivery"
PROPOSAL = DIRECTORY + "/saved-qualification-recovery.json"
FROZEN = DIRECTORY + "/saved-qualification-frozen.json"
AUTHORIZATION = DIRECTORY + "/saved-qualification-authorization.json"
ORIGINAL = "payload/locks/worktree/" + DIRECTORY + "/frozen-proposal/proposal.json"
BOUNDARY = "launcher/recovery/000027-collection-block-complete.pt"


def check_proposal(proposal):
    expected = {"attempts": 1, "wall_seconds_including_loading_scoring_and_readback": 900,
        "checkpoint_loads": 4, "saved_initialized_policies": 3, "saved_state_scorings": 624,
        "blocks": [60, 61, 62], "paths_each_block": ["r4", "initializer_greedy"], "rows_each_path": 104,
        "new_environment_calls": 0, "new_optimizer_updates": 0, "new_rollouts": 0, "final_test_episodes": 0}
    if (proposal["schema"] != "s1-saved-qualification-recovery-proposal-v1"
            or proposal["scientific_execution_authorized"] is not False or proposal["ready_to_launch"] is not False
            or proposal["limits"] != expected or proposal["automatic_training_after_pass"] is not False
            or proposal["reward_or_scenario_change"] is not False
            or proposal["remote_action_or_dropbox_export"] is not False
            or proposal["source_root"] != "results/dynamic_candidate_pilot_20261001"
            or proposal["new_output_root"] != "results/dynamic_candidate_saved_qualification_20261001"
            or set(proposal["inputs"]) != {BOUNDARY, *(f"payload/initialization/block{b}/graph/final.pt" for b in (60, 61, 62))}):
        raise ValueError("exact bounded saved-qualification proposal required")


def input_bindings(root, proposal):
    """Use the prior readout as evidence, not a repeat of its numerical audit."""
    check_proposal(proposal)
    source = proposal["source_root"]
    readout_path = proposal["raw_readout"]
    readout_record = file_record(root, readout_path)
    if readout_record["sha256"] != proposal["raw_readout_sha256"]:
        raise ValueError("saved readout changed")
    readout = json.loads(confined(root, readout_path).read_text())
    original_name = source + "/" + ORIGINAL
    original_record = file_record(root, original_name)
    if original_record["sha256"] != readout["proposal_file_sha256"]:
        raise ValueError("original archived scientific packet changed")
    original = json.loads(confined(root, original_name).read_text())
    if (digest(original["scientific_config"]) != proposal["unchanged_original_config_sha256"]
            or readout["config_sha256"] != proposal["unchanged_original_config_sha256"]
            or readout["source_unchanged"] is not True or readout["episodes"] != 39):
        raise ValueError("readout or unchanged scientific config differs")
    files = {readout_path: readout_record, original_name: original_record}
    for name, sha in proposal["inputs"].items():
        full = source + "/" + name
        files[full] = file_record(root, full)
        if files[full]["sha256"] != sha:
            raise ValueError("saved checkpoint bytes changed")
    outcomes = [row for row in readout["episode_outcomes"] if row["split"] == "qualification"]
    expected = {(b, r, w) for b in (60, 61, 62) for r in ("r4", "initializer_greedy") for w in (0, 1)}
    if len(outcomes) != 12 or {(r["block"], r["role"], r["world_index"]) for r in outcomes} != expected:
        raise ValueError("readout qualification matrix changed")
    for row in outcomes:
        for record in row["raw_files"].values():
            full = source + "/" + record["path"]
            observed = file_record(root, full)
            if observed != dict(record, path=full):
                raise ValueError("saved qualification raw file changed")
            files[full] = observed
    return files, original, outcomes


def freeze(root):
    root = Path(root).resolve()
    if git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain"):
        raise ValueError("freeze requires clean committed integration worktree")
    proposal = committed_json(root, PROPOSAL)
    inputs, original, _ = input_bindings(root, proposal)
    result = {"format": "saved-qualification-frozen-v1", "scientific_execution_authorized": False,
        "ready_to_launch": False, "workspace": str(root), "branch": BRANCH,
        "implementation_commit": git(root, "rev-parse", "HEAD"), "proposal": file_record(root, PROPOSAL),
        "limits": proposal["limits"], "source_files": source_locks(root), "input_files": inputs,
        "config_sha256": digest(original["scientific_config"]), "streams_sha256": digest(original["streams"]),
        "runtime": runtime_record(), "new_checkpoint_loads": 0, "new_model_forwards": 0,
        "new_environment_calls": 0, "new_optimizer_updates": 0}
    result["packet_sha256"] = digest(result)
    return result


def validate_authorization(authorization, packet):
    seal = dict(packet)
    sha = seal.pop("packet_sha256", None)
    if digest(seal) != sha:
        raise ValueError("frozen packet seal differs")
    if (packet["scientific_execution_authorized"] is not False or packet["ready_to_launch"] is not False
            or authorization.get("format") != "saved-qualification-explicit-authorization-v1"
            or authorization.get("approved") is not True or authorization.get("packet_sha256") != sha
            or authorization.get("implementation_commit") != packet["implementation_commit"]
            or authorization.get("proposal_sha256") != packet["proposal"]["sha256"]
            or authorization.get("limits") != packet["limits"]
            or authorization.get("automatic_retry") is not False
            or authorization.get("training_after_qualification") is not False):
        raise PermissionError("explicit approval for this exact saved-qualification packet required")
    user = authorization.get("user_approval", {})
    if (user.get("user") != "Zhaowei" or not isinstance(user.get("verbatim"), str)
            or not user["verbatim"].strip() or not user.get("recorded_at_utc")):
        raise PermissionError("actual approval text and time required")


def approved(root):
    # Fail before loading any checkpoint, constructing a model or claiming work.
    authorization = committed_json(root, AUTHORIZATION)
    packet = committed_json(root, FROZEN)
    validate_authorization(authorization, packet)
    proposal = committed_json(root, PROPOSAL)
    check_proposal(proposal)
    if (packet["workspace"] != str(Path(root).resolve()) or packet["branch"] != BRANCH
            or git(root, "branch", "--show-current") != BRANCH or git(root, "status", "--porcelain")
            or packet["proposal"] != file_record(root, PROPOSAL) or packet["limits"] != proposal["limits"]):
        raise ValueError("approved workspace/proposal changed")
    return packet, proposal, authorization


def verify_bindings(root, packet, proposal):
    subprocess.run(["git", "merge-base", "--is-ancestor", packet["implementation_commit"], "HEAD"], cwd=root, check=True)
    subprocess.run(["git", "diff", "--exit-code", packet["implementation_commit"], "--",
                    "src", "experiments/scripts", "tests", "AGENTS.md"], cwd=root, check=True, stdout=subprocess.DEVNULL)
    if source_locks(root) != packet["source_files"] or runtime_record() != packet["runtime"]:
        raise ValueError("frozen source/runtime changed")
    inputs, original, outcomes = input_bindings(root, proposal)
    if (inputs != packet["input_files"] or digest(original["scientific_config"]) != packet["config_sha256"]
            or digest(original["streams"]) != packet["streams_sha256"]):
        raise ValueError("frozen scientific inputs changed")
    return original, outcomes


class ScoringBudget:
    """Append before each irreversible load/forward; no refund or resume API."""
    def __init__(self, path, *, started, seconds, clock=shared_monotonic):
        self.started, self.seconds, self.clock = started, seconds, clock
        self.counts = {"checkpoint_load": 0, "scoring": 0}
        self.limits = {"checkpoint_load": 4, "scoring": 624}
        self.handle = Path(path).open("x")
        self.previous, self.sequence = "0" * 64, 0

    def check(self):
        now = self.clock()
        if now < self.started or now - self.started >= self.seconds:
            raise TimeoutError("saved qualification total wall deadline")

    def debit(self, kind):
        self.check()
        if kind not in self.counts or self.counts[kind] >= self.limits[kind]:
            raise ValueError("saved qualification budget exhausted/unknown operation")
        self.counts[kind] += 1
        row = {"sequence": self.sequence, "previous": self.previous, "operation": kind,
               "count": self.counts[kind], "clock": self.clock()}
        row["sha256"] = digest(row)
        self.handle.write(json.dumps(row, sort_keys=True) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.previous, self.sequence = row["sha256"], self.sequence + 1
        self.check()

    def close(self):
        self.handle.close()


def load_envelope(path, expected_sha, budget, *, loader=None):
    # Hash the same bytes the weights-only parser consumes, avoiding a path race.
    import io
    import torch
    from src.rl.prospective_ddpg_kernel import state_digest
    budget.check()
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha:
        raise ValueError("checkpoint hash differs immediately before load")
    budget.debit("checkpoint_load")
    data = (torch.load if loader is None else loader)(io.BytesIO(raw), map_location="cpu", weights_only=True)
    budget.check()
    if set(data) != {"state", "sha256"} or state_digest(data["state"]) != data["sha256"]:
        raise ValueError("saved checkpoint envelope differs")
    return data["state"]


def run_saved(root, packet, proposal, budget, *, loader=None):
    from src.rl.dynamic_candidate_saved_qualification import (
        restore_saved_policy, validate_saved_boundary, score_saved_qualification)
    original, outcomes = verify_bindings(root, packet, proposal)
    budget.check()
    config, streams = original["scientific_config"], original["streams"]
    source = confined(root, proposal["source_root"])
    output = confined(root, proposal["new_output_root"])
    boundary = load_envelope(source / BOUNDARY, proposal["inputs"][BOUNDARY], budget, loader=loader)
    initializers = {}
    for block in proposal["limits"]["blocks"]:
        name = f"payload/initialization/block{block}/graph/final.pt"
        state = load_envelope(source / name, proposal["inputs"][name], budget, loader=loader)
        initializers[block] = restore_saved_policy(state, config, streams, block)
        budget.check()
    episodes = []
    for row in outcomes:
        files = row["raw_files"]
        header = json.loads(confined(source, files["header"]["path"]).read_text())
        rows = [json.loads(line) for line in confined(source, files["events"]["path"]).read_text().splitlines()]
        episodes.append({"header": header, "rows": rows, "files": files})
        budget.check()
    examples = validate_saved_boundary(boundary, initializers, episodes, config, streams)
    budget.check()
    result = score_saved_qualification(initializers, examples, outcomes, config,
        before_forward=lambda: budget.debit("scoring"),
        on_path=lambda block, role, data: write_json_once(output / f"paths/block{block}-{role}.json", data))
    budget.check()
    if budget.counts != budget.limits:
        raise ValueError("qualification did not use exact load/scoring matrix")
    after = {name: file_record(root, name) for name in packet["input_files"]}
    if after != packet["input_files"]:
        raise ValueError("saved inputs changed during qualification")
    result.update(counts=budget.counts, last_debit_sha256=budget.previous, input_files_before=packet["input_files"],
        input_files_after=after, source_unchanged=True, environment_calls=0, optimizer_updates=0,
        new_rollouts=0, final_test_episodes=0, packet_sha256=packet["packet_sha256"])
    budget.check()
    write_json_once(output / "qualification.json", result)
    return result


def child(root):
    packet, proposal, authorization = approved(root)
    output = confined(root, proposal["new_output_root"])
    claim = json.loads((output / "claim.json").read_text())
    if (claim["pid"] != os.getppid() or claim["head"] != git(root, "rev-parse", "HEAD")
            or claim["packet_sha256"] != packet["packet_sha256"] or claim["authorization_sha256"] != digest(authorization)
            or claim["clock_id"] != CLOCK_ID or (output / "terminal.json").exists()):
        raise PermissionError("live exclusive parent claim required")
    write_json_once(output / "child-claim.json", {"pid": os.getpid(), "ppid": os.getppid()})
    budget = ScoringBudget(output / "debits.jsonl", started=claim["started"], seconds=900)
    try:
        run_saved(root, packet, proposal, budget)
        budget.check()
        return 0
    except BaseException as error:
        write_json_once(output / "failure.json", {"error": repr(error), "counts": budget.counts,
            "last_debit_sha256": budget.previous, "automatic_retry": False})
        return 1
    finally:
        budget.close()


def verify_saved_output(output, packet, started):
    """Read back the exact debit chain and six path receipts without scoring."""
    result = json.loads((output / "qualification.json").read_text())
    counts, previous, last_clock = {"checkpoint_load": 0, "scoring": 0}, "0" * 64, started
    for sequence, line in enumerate((output / "debits.jsonl").read_text().splitlines()):
        row = json.loads(line)
        sha = row.pop("sha256")
        kind = row["operation"]
        if (kind not in counts or row["sequence"] != sequence or row["previous"] != previous
                or digest(row) != sha or not last_clock <= row["clock"] < started + 900):
            raise ValueError("saved scoring debit readback differs")
        counts[kind] += 1
        if row["count"] != counts[kind]:
            raise ValueError("saved scoring debit counter differs")
        previous, last_clock = sha, row["clock"]
    if (counts != {"checkpoint_load": 4, "scoring": 624} or result["counts"] != counts
            or result["last_debit_sha256"] != previous or result["packet_sha256"] != packet["packet_sha256"]
            or result["input_files_before"] != packet["input_files"] or result["input_files_after"] != packet["input_files"]
            or result["source_unchanged"] is not True or result["automatic_training_authorized"] is not False
            or any(result[k] != 0 for k in ("environment_calls", "optimizer_updates", "new_rollouts", "final_test_episodes"))
            or set(result["blocks"]) != {"60", "61", "62"}):
        raise ValueError("saved qualification completion matrix differs")
    for block in ("60", "61", "62"):
        for role in ("r4", "initializer_greedy"):
            path = result["blocks"][block]["paths"][role]
            if path["rows"] != 104 or json.loads((output / f"paths/block{block}-{role}.json").read_text()) != path:
                raise ValueError("saved qualification path receipt differs")
    return result


def launch(root):
    started = shared_monotonic()
    packet, proposal, authorization = approved(root)
    output = confined(root, proposal["new_output_root"])
    output.mkdir(parents=True, exist_ok=False)
    write_json_once(output / "claim.json", {"pid": os.getpid(), "ppid": os.getppid(),
        "head": git(root, "rev-parse", "HEAD"), "packet_sha256": packet["packet_sha256"],
        "authorization_sha256": digest(authorization), "started": started, "clock_id": CLOCK_ID,
        "automatic_retry": False})
    try:
        write_json_once(output / "binding.json", {"packet": packet, "authorization": authorization})
        remaining = 900 - (shared_monotonic() - started)
        if remaining <= 0:
            raise TimeoutError("qualification admission deadline")
        supervisor = supervise([sys.executable, "-m", "experiments.scripts.run_dynamic_candidate_saved_qualification", "--child"],
            cwd=root, stdout_path=output / "stdout.log", stderr_path=output / "stderr.log",
            ledger_path=output / "unused-no-campaign-ledger", report_path=output / "supervisor.json",
            maximum_seconds=remaining)
        result = verify_saved_output(output, packet, started) if supervisor["passed"] else None
        clean = (output / "stderr.log").stat().st_size == 0
        success = (result is not None and clean and result["counts"] == {"checkpoint_load": 4, "scoring": 624}
                   and result["packet_sha256"] == packet["packet_sha256"] and result["source_unchanged"] is True
                   and not (output / "failure.json").exists() and shared_monotonic() - started < 900)
        write_json_once(output / "terminal.json", {"status": "completed" if success else "failed",
            "exit_code": 0 if success else 1, "qualification_passed": result["passed"] if success else None,
            "elapsed_seconds": shared_monotonic() - started, "automatic_retry": False,
            "automatic_training_authorized": False, "supervisor": supervisor})
        return 0 if success else 1
    except BaseException as error:
        write_json_once(output / "launch-failure.json", {"status": "failed", "error": repr(error), "automatic_retry": False})
        return 1
