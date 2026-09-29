"""Bounded real-environment software check; no policy performance measurement."""

from dataclasses import asdict

import numpy as np

from src.env.patient_capacity_planning import PatientConditionCapacityEnv, patient_env_config_from_dict
from src.models.prospective_forward_agent import ProspectiveForwardAgent
from src.rl.networks import torch
from src.rl.patient_replay_collector import PatientObservationProducer, PatientReplayCollector, evidence_digest
from src.rl.prospective_adapter import completed_segment_windows, prepare_replay_batch
from src.rl.validated_returns import bellman_targets


def run_engineering_check(config):
    if (config.get("kind") != "engineering_only_no_training" or config.get("enabled") is not True
            or config.get("device") != "cpu" or config.get("fixture_seed") != 0
            or config.get("steps") != 4 or config.get("n_steps") != 3):
        raise ValueError("this harness accepts only the declared bounded CPU unit fixture")
    cases = [(architecture, mode, boundary) for architecture in ("graph", "flat")
             for mode in ("physical", "self_only") for boundary in ("collector_cutoff", "finite_terminal")]
    cases.append(("graph", "physical", "environment_truncation"))
    reports, total_routes = [], 0
    for architecture, mode, boundary in cases:
        env_config = dict(config["env"])
        env_config["episode_horizon"] = 6 if boundary == "collector_cutoff" else 4
        env = PatientConditionCapacityEnv(patient_env_config_from_dict(env_config), seed=0)
        producer = PatientObservationProducer(env, enabled=True, gamma=config["gamma"],
                                              reward_scale=config["reward_scale"])
        agent = ProspectiveForwardAgent(
            producer.contract, enabled=True, architecture=architecture, message_mode=mode,
            hidden_width=config["hidden_width"], residual_scale=config["residual_scale"], seed=0)
        case_id = "/".join((architecture, mode, boundary))
        collector = PatientReplayCollector(env, producer, enabled=True, trajectory_id=case_id,
                                           max_steps=config["steps"], horizon_end=(
                                               "terminal" if boundary == "finite_terminal" else "truncation"))
        before = agent.weights_digest()
        for component in ("actor", "critic", "gate"):
            original = getattr(agent, component).state_dict()
            target = getattr(agent, "target_" + component).state_dict()
            if any(not torch.equal(original[key], target[key]) for key in original):
                raise AssertionError("initial target copy differs")
        receipts = []
        while not collector.closed:
            obs, anchor = producer.observe(env.observation())
            receipts.append(collector.step(agent.select_action(obs, anchor)))
        records = [step.record for step in receipts]
        windows = completed_segment_windows(records, producer.contract.replay, max_steps=config["n_steps"])
        batch = prepare_replay_batch(windows, producer.contract, dtype=torch.float32,
                                     device="cpu", message_mode=mode)
        current_q, next_q, targets = agent.replay_forward(batch)
        reference = bellman_targets(batch.samples, next_q.numpy(), producer.contract.replay)
        np.testing.assert_allclose(targets.numpy(), reference, rtol=2e-6, atol=2e-6)
        if [s.n_steps for s in batch.samples] != [3, 3, 2, 1]:
            raise AssertionError("short return tails were lost")
        for window, sample in zip(windows, batch.samples):
            expected = sum(config["gamma"] ** i * r.raw_reward * config["reward_scale"]
                           for i, r in enumerate(window))
            np.testing.assert_allclose(sample.reward, expected, rtol=1e-14, atol=1e-14)
        expected_discounts = ([config["gamma"] ** 3, 0., 0., 0.]
                              if boundary == "finite_terminal" else
                              [config["gamma"] ** n for n in (3, 3, 2, 1)])
        np.testing.assert_allclose(batch.bootstrap_discounts[:, 0].numpy(), expected_discounts, rtol=1e-6)
        if before != agent.weights_digest() or any(p.requires_grad or p.grad is not None for p in agent.parameters()):
            raise AssertionError("forward-only weights/gradients changed")
        total_routes += sum(step.execution["specimen_route_count"] for step in receipts)
        reports.append({
            "case": case_id, "contract": asdict(producer.contract), "parameter_inventory": agent.inventory(),
            "weights_before_sha256": before, "weights_after_sha256": agent.weights_digest(),
            "initial_target_copies_equal": True, "all_clone_steps_exact": all(r.clone_verified for r in receipts),
            "steps": [asdict(step) for step in receipts], "n_steps": [s.n_steps for s in batch.samples],
            "rewards_scaled": batch.rewards[:, 0].tolist(),
            "bootstrap_discounts": batch.bootstrap_discounts[:, 0].tolist(),
            "current_q": current_q[:, 0].tolist(), "next_q": next_q[:, 0].tolist(),
            "bellman_targets": targets[:, 0].tolist(), "independent_target_reference_matches": True,
        })
    if total_routes <= 0:
        raise AssertionError("routing fixture did not exercise a real specimen transfer")
    return {"schema": "prospective-collector-engineering-v1", "config_sha256_canonical": evidence_digest(config),
            "evidence_kind": "software_verification_not_performance", "device": "cpu", "dtype": "float32",
            "optimizer_updates": 0, "case_count": len(reports), "decision_count": sum(len(r["steps"]) for r in reports),
            "positive_routing_exercised": True, "architecture_parameter_matching_claimed": False,
            "clinical_horizon_closure_claimed": False, "online_gain_claimed": False, "cases": reports}
