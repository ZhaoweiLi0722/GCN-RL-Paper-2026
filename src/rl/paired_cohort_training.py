"""Serial branch and actor jobs for an externally admitted, bounded campaign.

These functions do not supply launch authority or construct patient simulators.
They bind actual collector/update interfaces to the existing nonrefund budget
and write-once evidence. Scientific callers must first verify their frozen packet.
"""

import copy
from dataclasses import asdict
from pathlib import Path

from src.rl.candidate_imitation import decode_example
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.paired_cohort_actor import PairedCohortActor, PairedCohortActorSettings, PairedCohortExample
from src.rl.paired_cohort_collection import make_branch, validate_context
from src.rl.paired_cohort_recording import PairedBranchRecorder
from src.rl.paired_cohort_verification import raw_json, verify_paired_cohort_branch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.networks import torch
from src.rl.routing_candidate_contract import RequestCandidates, RoutingRequestSchema


def branch_verifier(config, streams):
    """Bind independent expected seeds before inspecting a branch's own receipt."""
    config, streams = copy.deepcopy((config, streams))
    def verify(*, header, rows, states, receipt):
        context = header["context"]
        block, cohort, t = (context[k] for k in ("block", "cohort", "after_prefix_steps"))
        replication = header["manifest"]["replication"]
        return verify_paired_cohort_branch(context, header["manifest"], header["initial_state"],
            states["prefix_final"], states["final_state"], rows, receipt, config=config,
            expected_context_seed=streams["environment"][str(block)]["context"][cohort],
            expected_future_seed=streams["conditional_future"][f"block{block}/cohort{cohort}/after{t}"][replication])
    return verify


def block_dataset(assembled, block):
    """Bind each public-only block slice to the complete independent raw matrix."""
    complete = raw_json(assembled)
    source = complete.pop("dataset_sha256")
    if digest(complete) != source or complete["format"] != "paired-cohort-label-dataset-v1":
        raise ValueError("complete independent label dataset required")
    states = [row for row in complete["labels"] if row["block"] == block]
    if len(states) != 12 or len({(r["cohort"], r["after_prefix_steps"]) for r in states}) != 12:
        raise ValueError("exact twelve block contexts required")
    result = dict(format="paired-cohort-dataset-v1", block=block, source_dataset_sha256=source,
                  states=states)
    result["dataset_sha256"] = digest(result)
    return result


def collect_block(root, config, streams, block, contexts, template, producer_factory, reference,
                  budget, *, admit, verifier, recorder_type=PairedBranchRecorder, branch_factory=make_branch):
    """Every canonical candidate at every specified state, exactly two futures."""
    root = Path(root).resolve()
    if budget.active != f"paired_branches/block{block}" or block not in config["blocks"]:
        raise PermissionError("active prescribed branch owner required")
    expected = [(c, t) for c in range(config["context_cohorts_per_block"])
                for t in config["context_after_prefix_steps"]]
    if set(contexts) != set(expected) or not callable(admit) or not callable(verifier):
        raise ValueError("complete prespecified context inventory and admission required")
    index, run, recorder = [], None, None
    try:
        for cohort, t in expected:
            saved, bank = validate_context(contexts[cohort, t])
            if (saved["block"] != block or saved["cohort"] != cohort or saved["after_prefix_steps"] != t
                    or saved["source_id"] != streams["namespace"] or len(bank.requests) > config["max_original_requests"]):
                raise ValueError("context differs from fixed block/time/public support")
            future = streams["conditional_future"][f"block{block}/cohort{cohort}/after{t}"]
            if len(future) != config["future_replications"]:
                raise ValueError("exact future replication inventory required")
            for replication, seed in enumerate(future):
                for candidate in range(len(bank.class_keys)):
                    budget.check()
                    name = f"branches/block{block}/cohort{cohort}/after{t}/rep{replication}/class{candidate}"
                    run = branch_factory(template, producer_factory, reference, contexts[cohort, t],
                        future_seed=seed, candidate_index=candidate, replication=replication, branch_id=name,
                        before_clone=lambda: admit("conditional_branch_clone"), record=lambda *args: None, enabled=True)
                    recorder = recorder_type(root, "payload/" + name, run)
                    run.record = recorder.append
                    while not run.closed:
                        budget.check()
                        admit("branch_step")
                        run.step(before_step=lambda: budget.debit_environment("clone"))
                    if len(run.rows) != config["economic_endpoint"] - t:
                        raise ValueError("remaining-horizon branch length differs")
                    index.append(recorder.finish(run, verifier=verifier))
                    recorder = None
                    write_json_once(Path(root) / f"launcher/branch-boundaries/block{block}/{len(index):04d}.json",
                        dict(branch_id=name, index=index[-1], budget=budget.snapshot()))
        write_json_once(Path(root) / f"payload/branches/block{block}/index.json", index)
        return index
    except BaseException as error:
        if recorder is not None:
            recorder.close_partial()
        write_json_once(Path(root) / f"launcher/branch-failure-block{block}.json",
            dict(error=repr(error), completed=len(index), refunded=False, budget=budget.snapshot()))
        if run is not None:
            save_envelope(Path(root) / f"launcher/branch-failure-block{block}.pt", run.state_dict())
        raise


def actor_examples(dataset, contract):
    """Independent reader -> detached labels; no hidden state reaches actor inputs."""
    body = copy.deepcopy(dataset)
    sha = body.pop("dataset_sha256")
    if digest(body) != sha or body["format"] != "paired-cohort-dataset-v1":
        raise ValueError("independently verified finite dataset digest required")
    examples = []
    for row in body["states"]:
        public = copy.deepcopy(row["public_example"])
        raw = public["candidates"]
        canonical = RequestCandidates(RoutingRequestSchema(**raw["schema"]), raw["state_token"], raw["requests"])
        if digest(raw_json(asdict(canonical))) != digest(raw):
            raise ValueError("JSON support differs from canonical original requests")
        # JSON has no tuple type; restore canonical schema before the existing
        # tensor-envelope decoder's type-sensitive binding check.
        public["candidates"] = asdict(canonical)
        observation, bank = decode_example(public, contract)
        if row["public_example"]["split"] != "training":
            raise ValueError("test states cannot supply training labels")
        examples.append(PairedCohortExample(observation, bank,
            torch.tensor(row["raw_costs"], dtype=torch.float64, device="cpu"),
            tuple(tuple(v) for v in row["replication_seed_ids"]), bank.reference_class, bank.class_keys))
    return tuple(examples), sha


def fit_block(root, config, block, arm, prototype, contract, dataset, budget, *, admit,
              owner_type=PairedCohortActor):
    root = Path(root).resolve()
    phase = {"paired_cost": "paired_actor", "bc_continue": "bc_actor"}.get(arm)
    if phase is None or block not in config["blocks"] or budget.active != f"{phase}/block{block}" or not callable(admit):
        raise PermissionError("active finite actor owner and admission required")
    if dataset["block"] != block:
        raise ValueError("foreign block labels")
    examples, sha = actor_examples(dataset, contract)
    admit("actor_fork")
    owner = owner_type(prototype, contract, PairedCohortActorSettings.from_config(config), examples,
                       dataset_sha256=sha, arm=arm, enabled=True)
    directory = Path(root) / f"payload/models/block{block}/{arm}"
    save_envelope(directory / "initial.pt", owner.state_dict())
    before = prototype.snapshot_sha256()
    try:
        while owner.steps < config["actor_updates_per_arm_block"]:
            admit("actor_update")
            receipt = owner.update(before_optimizer_step=budget.debit_optimizer, before_compute=budget.check)
            if receipt["critic_optimizer_steps"] != 0 or prototype.snapshot_sha256() != before:
                raise ValueError("critic update or initializer mutation")
            path = directory / f"step{owner.steps:03d}.pt"
            save_envelope(path, owner.state_dict())
            write_json_once(directory / f"step{owner.steps:03d}.json",
                dict(receipt=receipt, model=file_record(Path(root), path), dataset_sha256=sha,
                     state_sha256=state_digest(owner.state_dict()), budget=budget.snapshot()))
        save_envelope(directory / "final.pt", owner.state_dict())
        return owner, file_record(Path(root), directory / "final.pt")
    except BaseException as error:
        save_envelope(directory / "failure.pt", owner.state_dict())
        write_json_once(directory / "failure.json", dict(error=repr(error), refunded=False, budget=budget.snapshot()))
        raise
