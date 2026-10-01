"""Explicit prospective model/qualification bindings; no launch or data collection."""

import numpy as np

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_imitation import ImitationSettings, decode_example
from src.rl.candidate_ppo_kernel import CandidatePPOSettings
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.prospective_ddpg_kernel import state_digest


def ppo_settings(config):
    opt, schedule = config["optimizer"], config["continuation"]
    ppo = schedule["ppo"]
    return CandidatePPOSettings(
        learning_rate=opt["learning_rate"], clip_ratio=ppo["clip_ratio"],
        value_loss_coef=ppo["value_loss_coef"], entropy_coef=ppo["entropy_coef"],
        max_grad_norm=opt["gradient_norm_cap_each_owner"], gae_lambda=config["objective"]["gae_lambda"],
        normalize_advantages=ppo["normalize_advantages_once_per_full_rollout"],
        epochs=schedule["epochs"], batch_size=schedule["batch_size"],
        max_rollout_steps=schedule["rows_per_rollout"], max_updates=schedule["rollouts_per_arm_per_block"],
        max_optimizer_steps=ppo["actor_adam_calls_per_block"] + ppo["critic_adam_calls_per_block"])


def dynamic_template(producer, config, streams, block):
    spec = config["model_proposal"]
    if config.get("candidate_message_graph") != "specimen_routes" or spec["graph"] != "specimen_routes":
        raise ValueError("prospective explicit specimen-routes proposal required; no graph fallback")
    base = f"block{block}/graph/"
    neural = streams["neural"]
    policy = DynamicCandidatePolicy(producer.contract.inputs, enabled=True,
        architecture="graph", message_mode="physical", encoder_width=spec["encoder_width"],
        head_width=spec["head_width"], actor_seed=neural[base + "actor_initialization"],
        critic_seed=neural[base + "critic_initialization"],
        initial_reference_bias=spec["initial_reference_bias"])
    for owner in ("actor", "critic"):
        declared = spec.get(owner + "_parameter_count")
        actual = sum(p.numel() for p in getattr(policy, owner + "_parameters")())
        if declared is not None and declared != actual:
            raise ValueError("static model dimensions differ from the live public contract")
    return DynamicCandidatePPOKernel(policy, producer.contract, ppo_settings(config), enabled=True,
        mode="frozen", sampling_seed=neural[base + "continuation/sample"],
        shuffle_seed=neural[base + "ppo/shuffle"])


def dynamic_initializer(template, config, streams, block):
    opt, init = config["optimizer"], config["initialization"]
    return DynamicCandidateImitationKernel(template.policy, template.contract,
        ImitationSettings(opt["learning_rate"], opt["gradient_norm_cap_each_owner"],
                          init["batch_size"], init["actor_adam_calls_per_block"], 1, "demonstration"),
        enabled=True, shuffle_seed=streams["neural"][f"block{block}/graph/bc_init/shuffle"])


def qualify_dynamic_path(kernel, examples, config, *, before_forward):
    """Re-score both independently collected paths without fitting or hidden inputs."""
    cfg = config["qualification"]
    expected = cfg["fresh_worlds_per_block"] * config["objective"]["horizon"]
    if (len(examples) != expected or not examples
            or len({e["identity"] for e in examples}) != expected):
        raise ValueError("exact distinct independent qualification rows required")
    before = state_digest(kernel.state_dict())
    hits, multi, multi_hits = 0, 0, 0
    for example in examples:
        before_forward()
        if example["split"] != "qualification":
            raise ValueError("qualification cannot consume demonstrations or test rows")
        observation, bank = decode_example(example, kernel.contract)
        evaluation = evaluate_dynamic_policy(kernel.policy, observation, bank, kernel.contract)
        hit = int(np.argmax(evaluation.log_probs)) == bank.reference_class
        hits += int(hit)
        if len(bank.class_keys) > 1:
            multi += 1
            multi_hits += int(hit)
    if state_digest(kernel.state_dict()) != before:
        raise ValueError("qualification modified model, optimizer or RNG")
    agreement, multi_agreement = hits / expected, multi_hits / multi if multi else None
    return {"rows": expected, "agreement": agreement, "multiclass_rows": multi,
            "multiclass_agreement": multi_agreement, "kernel_sha256": before,
            "passed": agreement >= cfg["minimum_reference_agreement_each_path"]
            and multi >= cfg["minimum_multiclass_rows_each_path"]
            and multi_agreement is not None
            and multi_agreement >= cfg["minimum_multiclass_reference_agreement_each_path"]}


def qualify_dynamic_outcomes(outcomes, config, block):
    """Uncertainty-free rejection screen, explicitly not clinical noninferiority."""
    count = config["qualification"]["fresh_worlds_per_block"]
    pairs = {}
    for row in outcomes:
        if row["block"] != block or row["role"] not in ("r4", "initializer_greedy"):
            raise ValueError("unexpected qualification outcome")
        key = (row["world_index"], row["role"])
        if key in pairs:
            raise ValueError("duplicate qualification outcome")
        pairs[key] = row
    if set(pairs) != {(w, role) for w in range(count) for role in ("r4", "initializer_greedy")}:
        raise ValueError("missing paired qualification outcome")
    deltas = {key: [] for key in ("cost", "losses", "completions", "terminal_active")}
    for world in range(count):
        reference, own = pairs[world, "r4"], pairs[world, "initializer_greedy"]
        if reference["seed"] != own["seed"]:
            raise ValueError("qualification world start mismatch")
        for metric in deltas:
            values = [own[metric], reference[metric]]
            if not all(np.isfinite(v) for v in values):
                raise ValueError("nonfinite qualification outcome")
            deltas[metric].append(values[0] - values[1])
    means = {key: float(np.mean(value)) for key, value in deltas.items()}
    return {"paired_deltas": deltas, "block_mean_deltas": means,
            "passed": all(means[m] <= 0 for m in ("cost", "losses", "terminal_active"))
            and means["completions"] >= 0, "clinical_noninferiority_claim": False}


def fork_dynamic_initializer(initializer, qualification, config, streams, block):
    if (qualification.get("passed") is not True
            or qualification["kernel_sha256"] != state_digest(initializer.state_dict())
            or initializer.steps != config["initialization"]["actor_adam_calls_per_block"]):
        raise ValueError("unchanged complete qualified initializer required")
    base = f"block{block}/graph/"
    neural, opt, cont = streams["neural"], config["optimizer"], config["continuation"]
    result = {role: DynamicCandidatePPOKernel(initializer.policy, initializer.contract,
        ppo_settings(config), enabled=True, mode="frozen" if role == "own_frozen" else "online",
        sampling_seed=neural[base + "continuation/sample"], shuffle_seed=neural[base + "ppo/shuffle"])
        for role in ("own_frozen", "own_ppo")}
    result["own_bc_continue"] = DynamicCandidateImitationKernel(initializer.policy, initializer.contract,
        ImitationSettings(opt["learning_rate"], opt["gradient_norm_cap_each_owner"], cont["batch_size"],
            cont["bc_continue"]["actor_adam_calls_per_block"], cont["rollouts_per_arm_per_block"], "training"),
        enabled=True, sampling_seed=neural[base + "continuation/sample"],
        shuffle_seed=neural[base + "bc_continue/shuffle"])
    for kernel in result.values():
        optimizers = kernel.optimizers.values() if type(kernel) is DynamicCandidatePPOKernel else [kernel.optimizer]
        if (kernel.policy.snapshot_sha256() != initializer.policy.snapshot_sha256()
                or any(o is not None and o.state_dict()["state"] for o in optimizers)):
            raise ValueError("policy fork mismatch or inherited initializer moments")
    if (state_digest(result["own_ppo"].sampling_rng.get_state())
            != state_digest(result["own_bc_continue"].sampling_rng.get_state())):
        raise ValueError("continuation sampler starts differ")
    return result
