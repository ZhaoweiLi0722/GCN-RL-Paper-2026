"""Admitted per-round collection and actor fit over a single sealed public dataset."""

import copy
from pathlib import Path

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.conservative_cohort_actor import ConservativeCohortActor, ConservativeCohortActorSettings
from src.rl.conservative_cohort_collection import make_current_policy_branch
from src.rl.conservative_cohort_plan import cohort_ids, validate_config
from src.rl.paired_cohort_actor import _actor_logits
from src.rl.paired_cohort_collection import validate_context
from src.rl.paired_cohort_recording import PairedBranchRecorder
from src.rl.paired_cohort_training import actor_examples
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.networks import torch


def collect_round(root, config, streams, block, round_index, contexts, template,
                  producer_factory, reference, continuation, budget, *, admit,
                  verifier, recorder_type=PairedBranchRecorder,
                  branch_factory=make_current_policy_branch):
    validate_config(config)
    if budget.active != f"branches/round{round_index}/block{block}" or block not in config["blocks"]:
        raise PermissionError("prespecified round/block branch owner required")
    expected = [(c, t) for c in cohort_ids(round_index) for t in config["context_after_prefix_steps"]]
    if set(contexts) != set(expected) or not callable(admit) or not callable(verifier):
        raise ValueError("exact six current-round states and independent reader required")
    root, index, recorder, run = Path(root).resolve(), [], None, None
    before = state_digest(continuation.state_dict())
    try:
        for c, t in expected:
            saved, bank = validate_context(contexts[c, t])
            if (saved["block"] != block or saved["cohort"] != c or saved["after_prefix_steps"] != t
                    or saved["source_id"] != streams["namespace"] or len(bank.requests) > 6):
                raise ValueError("context round/block/public support changed")
            seeds = streams["conditional_future"][f"block{block}/cohort{c}/after{t}"]
            if len(seeds) != 4 or len(set(seeds)) != 4:
                raise ValueError("four distinct prescribed future replications required")
            for replication, seed in enumerate(seeds):
                for candidate in range(len(bank.class_keys)):
                    budget.check()
                    name = f"branches/round{round_index}/block{block}/cohort{c}/after{t}/rep{replication}/class{candidate}"
                    run = branch_factory(template, producer_factory, reference, contexts[c, t],
                        continuation=continuation, round_index=round_index, before_forward=budget.check,
                        future_seed=seed, candidate_index=candidate, replication=replication, branch_id=name,
                        before_clone=lambda: admit("conditional_branch_clone"), record=lambda *a: None, enabled=True)
                    recorder = recorder_type(root, "payload/" + name, run)
                    run.record = recorder.append
                    while not run.closed:
                        admit("branch_step")
                        run.step(before_step=lambda: budget.debit_environment("clone"))
                    if len(run.rows) != 63 - t or state_digest(continuation.state_dict()) != before:
                        raise ValueError("incomplete branch or mutated continuation")
                    index.append(recorder.finish(run, verifier=verifier))
                    recorder = None
                    write_json_once(root / f"launcher/branch-boundaries/round{round_index}/block{block}/{len(index):04d}.json",
                        dict(branch_id=name, index=index[-1], continuation_sha256=before, budget=budget.snapshot()))
        write_json_once(root / f"payload/branches/round{round_index}/block{block}/index.json", index)
        return index
    except BaseException as error:
        if recorder is not None:
            recorder.close_partial()
        directory = root / f"launcher/failures/branches/round{round_index}/block{block}"
        write_json_once(directory / "failure.json", dict(error=repr(error), completed=len(index),
            refunded=False, automatic_retry=False, budget=budget.snapshot()))
        if run is not None:
            save_envelope(directory / "state.pt", run.state_dict())
        raise


def fit_round(root, config, block, round_index, arm, prototype, contract, dataset, budget, *, admit,
              owner_type=ConservativeCohortActor, decode=actor_examples, logits=_actor_logits):
    """Exactly 64 steps; caller supplies this arm's own preceding boundary.

    q is computed exactly once from the same round-start actor on every public
    example, not from a stale logits file or from the other arm's model. The
    original same-start requirement is enforced by the campaign, not guessed here.
    """
    validate_config(config)
    phase = {"paired_cost": "paired_actor", "bc_continue": "bc_actor"}.get(arm)
    if (phase is None or block not in config["blocks"] or round_index not in (0, 1)
            or budget.active != f"{phase}/round{round_index}/block{block}"):
        raise PermissionError("declared arm/round/block update owner required")
    if dataset["block"] != block or dataset["round"] != round_index:
        raise ValueError("wrong round/block dataset")
    examples, sha = decode(dataset, contract)
    if len(examples) != 6:
        raise ValueError("six shared current-round examples required")
    settings = ConservativeCohortActorSettings()
    if (config["actor_optimizer"] != dict(name="Adam", lr=settings.learning_rate, betas=list(settings.betas),
            eps=settings.eps, weight_decay=settings.weight_decay, gradient_norm_cap=settings.max_grad_norm)
            or config["kl_coefficient"] != settings.kl_coefficient or config["cost_scale"] != settings.cost_scale):
        raise ValueError("config/actor settings differ")
    root = Path(root).resolve()
    directory = root / f"payload/models/round{round_index}/block{block}/{arm}"
    before = prototype.snapshot_sha256()
    admit("actor_fork")
    q = []
    with torch.no_grad():
        for example in examples:
            admit("round_start_logits")
            q.append(logits(prototype, example).detach().to(dtype=torch.float64, device="cpu"))
    if prototype.snapshot_sha256() != before:
        raise ValueError("q inference changed the prototype")
    owner = owner_type(prototype, contract, settings, examples, dataset_sha256=sha,
        arm=arm, reference_logits=tuple(q), round_index=round_index + 1, enabled=True)
    save_envelope(directory / "initial.pt", owner.state_dict())
    write_json_once(directory / "binding.json", dict(dataset_sha256=sha, initial_policy_sha256=before,
        round=round_index, arm=arm, optimizer_reset=True, q_sha256=state_digest(tuple(q)),
        scientific_model_selection=False))
    try:
        while owner.steps < config["actor_updates_per_round_arm_block"]:
            admit("actor_update")
            receipt = owner.update(before_optimizer_step=budget.debit_optimizer, before_compute=budget.check)
            if receipt["critic_optimizer_steps"] != 0 or prototype.snapshot_sha256() != before:
                raise ValueError("critic update or prototype mutation")
            save_envelope(directory / f"step{owner.steps:03d}.pt", owner.state_dict())
            write_json_once(directory / f"step{owner.steps:03d}.json", dict(receipt=receipt,
                budget=budget.snapshot(), dataset_sha256=sha))
        save_envelope(directory / "final.pt", owner.state_dict())
        return owner, file_record(root, directory / "final.pt")
    except BaseException as error:
        save_envelope(directory / "failure.pt", owner.state_dict())
        write_json_once(directory / "failure.json", dict(error=repr(error), refunded=False,
            automatic_retry=False, budget=budget.snapshot()))
        raise
