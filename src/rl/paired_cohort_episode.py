"""Whole reference/evaluation cohorts and counted same-start clone dispatch."""

import copy
from pathlib import Path

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.cohort_collection import CohortCollection
from src.rl.cohort_recording import CohortRecorder
from src.rl.paired_cohort_collection import capture_context, validate_context
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays


def recycle_prefix_template(env, context):
    """Reuse the completed owned instance, not an uncharged simulator clone."""
    state, _ = validate_context(context)
    if (not env._cohort_closed or env._cohort_failure is not None
            or env._cohort_steps != env.cohort_spec.accounting_steps):
        raise ValueError("only a successfully recorded completed cohort may be recycled")
    for name in ("_cohort_ids", "_cohort_enrolled", "_cohort_resolution_step", "_cohort_prefix_state"):
        setattr(env, name, None)
    env._cohort_closed, env._cohort_steps, env._cohort_costs = False, 0, []
    env.load_state_dict(state["environment"])
    if evidence_digest(env.state_dict()) != state["environment_sha256"]:
        raise ValueError("recycled template differs from saved prefix")
    return env


def run_episode(root, backend_config, config, streams, backend, model, budget, *, admit,
                block, role, world, seed, split, selection, name, clone=False,
                recorder_type=CohortRecorder, collection_type=CohortCollection,
                capture=capture_context, on_boundary=None):
    root = Path(root).resolve()
    frozen_hash = state_digest(model.state_dict())
    prefix = backend.session(block, model, seed, trajectory=name, split=split, selection=selection)
    writers, runs, contexts = [], [], {}
    def attach(session, path):
        writer = recorder_type(root, path, session, backend_config, block=block, role=role,
            world_index=world, seed=seed, representation="reference" if role in ("r4", "full_mdl2") else "graph")
        writers.append(writer)
        run = collection_type(session, enabled=True, objective="none", split=split, trajectory_id=name,
            followup_action=lambda env: env.common_followup_request(session.producer.anchor_config),
            finish_prefix=writer.finish_prefix, record_prefix=writer.record_prefix, record_tail=writer.record_tail)
        runs.append(run)
        return run
    try:
        original = attach(prefix, "payload/episodes/" + name)
        save_envelope(root / f"payload/episodes/{name}/initial.pt", original.state_dict())
        if clone:
            if split != "preflight":
                raise ValueError("same-start clone belongs only to the counted preflight")
            admit("preflight_clone")
            twin_prefix = copy.deepcopy(prefix)
            twin_prefix.load_state_dict(prefix.state_dict())
            twin = attach(twin_prefix, "payload/preflight-clones/" + name)
            save_envelope(root / f"payload/preflight-clones/{name}/initial.pt", twin.state_dict())
        while not original.closed:
            budget.check()
            admit("episode_step")
            event = original.step(before_step=lambda: budget.debit_environment("trajectory"))
            if clone:
                admit("clone_step")
                other = twin.step(before_step=lambda: budget.debit_environment("clone"))
                if (state_digest(encode_arrays(event)) != state_digest(encode_arrays(other))
                        or evidence_digest(original.env.state_dict()) != evidence_digest(twin.env.state_dict())
                        or state_digest(encode_arrays(original.env.followup_state_dict())) !=
                        state_digest(encode_arrays(twin.env.followup_state_dict()))):
                    raise ValueError("same-start clone raw event/state/RNG differs")
            if split == "training" and original.index in config["context_after_prefix_steps"]:
                t = original.index
                context_id = f"block{block}/cohort{world}/after{t}"
                contexts[world, t] = capture(original.env, prefix.producer, prefix.reference, prefix.options,
                    context_id=context_id, block=block, cohort=world, allowed_times=config["context_after_prefix_steps"],
                    source_id=streams["namespace"])
                save_envelope(root / f"payload/contexts/{context_id}.pt", contexts[world, t])
                save_envelope(root / f"payload/episodes/{name}/after{t}.pt", original.state_dict())
                if on_boundary is not None:
                    on_boundary(original, writers[0])
        if original.index != config["economic_endpoint"] or state_digest(model.state_dict()) != frozen_hash:
            raise ValueError("incomplete endpoint or mutated fixed model")
        indexes = [writer.finish(run) for writer, run in zip(writers, runs)]
        write_json_once(root / f"payload/episodes/{name}/completion.json", dict(index=indexes[0],
            clone_index=indexes[1] if clone else None, budget=budget.snapshot(),
            model_sha256=frozen_hash, contexts=[list(key) for key in contexts]))
        return dict(index=indexes[0], clone_index=indexes[1] if clone else None,
                    contexts=contexts, environment=original.env)
    except BaseException as error:
        for writer in writers:
            if not writer.closed:
                writer.close_partial()
        for i, run in enumerate(runs):
            save_envelope(root / f"launcher/episode-failures/{name}/{i}.pt", run.state_dict())
        write_json_once(root / f"launcher/episode-failures/{name}/failure.json",
            dict(error=repr(error), budget=budget.snapshot(), refunded=False, automatic_retry=False))
        raise
