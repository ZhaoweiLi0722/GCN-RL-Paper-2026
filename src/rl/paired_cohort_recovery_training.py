"""Complete only the committed remainder of the original conditional matrix."""

from dataclasses import replace
from pathlib import Path

from src.env.cohort_followup import CohortTailSpec, cohort_environment_class, same_payload
from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.paired_cohort_collection import make_branch, validate_context
from src.rl.paired_cohort_recording import PairedBranchRecorder


def restore_template(backend, block, saved, *, admit):
    """One non-stepped template, restored to the original complete prefix state."""
    from src.env.patient_capacity_planning import patient_env_config_from_dict
    from src.rl.experiment import apply_graph_ablation
    from src.rl.frozen_value_probe import assert_scenario
    context, _ = validate_context(saved)
    admit("template_build")
    rng = global_rng_state()
    try:
        runtime, _, reference = backend.contexts[block]
        cfg = dict(runtime.get("env", {}))
        ablation = cfg.pop("graph_ablation", runtime.get("graph_ablation", "full_graph"))
        scenario = cfg.pop("scenario_name", runtime.get("scenario", "default"))
        typed = patient_env_config_from_dict(cfg)
        typed = replace(typed, base=apply_graph_ablation(typed.base, ablation))
        env = cohort_environment_class()(typed, seed=context["environment"]["scalars"]["_episode_seed"],
            cohort_spec=CohortTailSpec(**context["cohort_spec"]), enabled=True)
        env.scenario_name, env.graph_ablation = scenario, ablation
        assert_scenario(env, runtime, backend.config["objective"]["scenario"])
        env.load_state_dict(context["environment"])
        if (not same_payload(env.state_dict(), context["environment"])
                or env.t != context["after_prefix_steps"] or env._cohort_closed
                or env._cohort_steps != 0 or reference.checkpoint_sha256 != context["reference_sha256"]):
            raise ValueError("saved prefix template/reference did not restore exactly")
        backend.branch_producer(block, env).check_environment(env)
        return env
    finally:
        restore_global_rng(rng)


def collect_remaining(root, config, streams, block, contexts, template, producer_factory,
                      reference, budget, *, remaining, admit, verifier,
                      recorder_type=PairedBranchRecorder, branch_factory=make_branch):
    if budget.active != f"paired_branches/block{block}" or not remaining:
        raise PermissionError("active finite remaining-branch owner required")
    root = Path(root).resolve()
    index, run, recorder = [], None, None
    seen = set()
    try:
        for row in remaining:
            cohort, t = row["cohort"], row["after_prefix_steps"]
            context, bank = validate_context(contexts[cohort, t])
            rep, candidate, name = row["replication"], row["candidate_index"], row["branch_id"]
            canonical = f"branches/block{block}/cohort{cohort}/after{t}/rep{rep}/class{candidate}"
            future = streams["conditional_future"][f"block{block}/cohort{cohort}/after{t}"][rep]
            if (row["block"] != block or context["block"] != block or name != canonical
                    or name in seen or not 0 <= candidate < len(bank.class_keys)
                    or row["future_seed"] != str(future)
                    or row["environment_calls"] != config["economic_endpoint"] - t):
                raise ValueError("remaining branch identity/support/seed changed")
            seen.add(name)
            budget.check()
            run = branch_factory(template, producer_factory, reference, contexts[cohort, t],
                future_seed=future, candidate_index=candidate, replication=rep, branch_id=name,
                before_clone=lambda: admit("conditional_branch_clone"), record=lambda *args: None, enabled=True)
            recorder = recorder_type(root, "payload/" + name, run)
            run.record = recorder.append
            while not run.closed:
                budget.check()
                admit("branch_step")
                run.step(before_step=lambda: budget.debit_environment("clone"))
            if len(run.rows) != row["environment_calls"]:
                raise ValueError("remaining branch length differs")
            index.append(recorder.finish(run, verifier=verifier))
            recorder = None
            write_json_once(root / f"launcher/branch-boundaries/block{block}/{len(index):04d}.json",
                dict(branch_id=name, index=index[-1], budget=budget.snapshot(), recovery_new_branch=True))
        write_json_once(root / f"payload/branches/block{block}/recovery-index.json", index)
        return index
    except BaseException as error:
        if recorder is not None:
            recorder.close_partial()
        write_json_once(root / f"launcher/branch-failure-block{block}.json",
            dict(error=repr(error), completed=len(index), refunded=False, budget=budget.snapshot()))
        if run is not None:
            save_envelope(root / f"launcher/branch-failure-block{block}.pt", run.state_dict())
        raise
