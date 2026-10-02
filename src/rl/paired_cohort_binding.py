"""Explicit saved initializer/PPO binding; no scientific call on import."""

from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.cohort_evaluation_models import restore_evaluation_owner
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_saved_qualification import restore_saved_policy
from src.rl.paired_cohort_backend import PairedCohortBackend
from src.rl.paired_cohort_resources import runtime_streams
from src.rl.prospective_ddpg_kernel import state_digest


def frozen_owner(view, seed):
    # Inert compatibility settings: frozen mode creates no optimizers. These
    # values cannot authorize fitting and do not change action probabilities.
    settings = CandidatePPOSettings(3e-4, .2, 0., 0., .5, 1., False, 1, 1, 52, 1, 1)
    owner = DynamicCandidatePPOKernel(view.policy, view.contract, settings, enabled=True,
        mode="frozen", sampling_seed=seed, shuffle_seed=seed)
    if owner.policy.snapshot_sha256() != view.policy.snapshot_sha256():
        raise ValueError("frozen evaluation wrapper changed trained weights")
    return owner


def bind_inputs(workspace, root, packet, budget, admission, *, backend_type=PairedCohortBackend,
                loader=load_envelope, initializer_restore=restore_saved_policy,
                ppo_restore=restore_evaluation_owner, wrap=frozen_owner, record=file_record):
    """One read per historical model; fixture dependency injection is explicit.

    Packet/claim authorization belongs to the sole execution entry. All native
    backend builds, historical loads and file verification are separately admitted.
    """
    metadata, cfg = packet["inherited"], packet["config"]
    streams = runtime_streams(packet["streams"])
    backend = backend_type(workspace, metadata["backend_config"], streams, admit_real_calls=admission)
    initializers, models = {}, {}
    for block in cfg["blocks"]:
        budget.check()
        receipt = backend.prepare(block, streams["environment"][str(block)]["layout"][0])
        write_json_once(Path(root) / f"payload/binding/block{block}.json", receipt)
        for role in ("initializer", "saved_cohort_ppo"):
            key = f"block{block}/{role}"
            expected = metadata["model_inputs"][key]
            admission("input_hash_verification")
            if record(Path(workspace), expected["path"]) != expected:
                raise ValueError("pinned model changed: " + key)
            admission("checkpoint_load")
            state = loader(Path(workspace) / expected["path"])
            if role == "initializer":
                view = initializer_restore(state, packet["initializer_config"], packet["initializer_streams"], block)
                qualification = packet["qualified_blocks"][str(block)]
                if (qualification["passed"] is not True
                        or qualification["kernel_sha256"] != state_digest(view.state_dict())):
                    raise ValueError("saved qualification does not bind this initializer")
                initializers[block] = view
                owner = wrap(view, streams["neural"][f"block{block}/paired_cost"])
                key = f"block{block}/own_frozen"
            else:
                owner = ppo_restore(state)
            if owner.contract != backend.producer(block).contract:
                raise ValueError("restored owner and layout input contracts differ")
            models[key] = owner
    write_json_once(Path(root) / "payload/binding/models.json", dict(inputs=metadata["model_inputs"],
        models={k: state_digest(v.state_dict()) for k, v in models.items()}, initializer_refitted=False,
        initializer_requalified=False, historical_model_loads=len(models)))
    return backend, initializers, models, streams
