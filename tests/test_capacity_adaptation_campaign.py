"""One complete numeric fake packet, not training or patient experimentation."""

import copy
import json
from pathlib import Path
import subprocess
import sys
import unittest

from src.rl.capacity_adaptation_campaign import (
    ArtificialReceipt, BACKEND_PROTOCOL, CAPACITY_CONTROL_ROLES,
    CapacityAdaptationCampaign, PHASES, STATE_KEYS, numeric_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PROPOSAL = ROOT / "specs/2026-10-03-dynamic-capacity-adaptation/pilot-proposal.json"


def proposal():
    return json.loads(PROPOSAL.read_text())


class RecordingFake:
    """Integer tokens only; no environment, network, optimizer or solver."""

    backend_protocol = BACKEND_PROTOCOL

    def __init__(self, fail=None, unsettled=False):
        self.fail, self.unsettled = fail, unsettled
        self.campaign = None
        self.seals, self.forks, self.worlds, self.updates = {}, {}, {}, []
        self.transitions, self.first_controls, self.noise = {}, {}, {}
        self.barrier = False
        self.calls, self.plan_epochs = 0, 0
        self.last_kind = None
        self.current_role = None

    def dispatch(self, call):
        self.calls += 1
        self.last_kind = call.kind
        p, kind = call.payload, call.kind
        if kind == self.fail:
            raise RuntimeError("injected fake dispatch failure")
        if kind == "planner_epoch":
            assert call.epoch < 48
            self.plan_epochs += 1
            return None
        if kind == "initialize":
            assert type(p["seed"]) is int
            return {k: (0 if k == "replay" else p["seed"]) for k in STATE_KEYS}
        if kind == "construct":
            assert type(p["world"]["seed"]) is int
            if call.phase == PHASES[2]:
                assert self.barrier and set(self.seals) == {0, 1, 2}
            self.worlds[call.owner] = copy.deepcopy(p["world"])
        if kind == "construction_reset":
            assert p["history"] == [] and p["filter"] is None
            return {"epoch": 0, "own_action": None}
        if kind == "control":
            self.current_role = p["role"]
            assert len(p["history"]) == call.epoch
            assert p["public"]["epoch"] == call.epoch
            assert all(row["action"]["owner"] == call.owner for row in p["history"])
            if call.epoch == 0:
                assert p["filter"] is None
                self.first_controls[call.owner] = copy.deepcopy(p)
            else:
                assert p["filter"] == {"owner": call.owner, "receipts": call.epoch}
            if call.phase == PHASES[2] and p["noise_key"] is not None:
                self.noise[call.owner, call.epoch] = p["noise_key"]
            # Deliberately mutate the callback's copy to test ownership isolation.
            p["history"].append({"bad": "must not reach campaign"})
            if p["state"] is not None:
                p["state"]["actor"] = -999
            return {"owner": call.owner, "role": p["role"], "epoch": call.epoch}
        if kind in ("control_step", "tail_step"):
            if kind == "tail_step":
                assert p["action"] == {"artificial_tail": True, "requested_hours": [0, 0, 0, 0]}
            return ArtificialReceipt(call.epoch, {"epoch": call.epoch + 1,
                                      "own_action": copy.deepcopy(p["action"])},
                                     float(call.epoch + 1), call.epoch == 63 and not self.unsettled)
        if kind == "filter":
            assert len(p["history"]) == call.epoch + 1
            return {"owner": call.owner, "receipts": call.epoch + 1}
        if kind == "remember":
            state = p["state"]
            state["replay"] += 1
            return state
        if kind in ("actor_step", "critic_step"):
            if call.phase == PHASES[2]:
                assert call.owner.endswith("/online_matched_fork")
            self.updates.append((call.phase, call.owner, call.epoch, kind, copy.deepcopy(p["transition"])))
            state = p["state"]
            component = kind.removesuffix("_step")
            state[component] += 1
            state[component + "_optimizer"] += 1
            return state
        if kind == "targets":
            state = p["state"]
            state["target_actor"], state["target_critic"] = state["actor"], state["critic"]
            return state
        if kind == "seal":
            assert p["seal"]["state"]["replay"] == p["seal"]["replay_rows"] == 1152
            self.seals[call.block] = copy.deepcopy(p["seal"])
            p["seal"]["state"]["actor"] = -888
        if kind == "seal_barrier":
            assert set(self.seals) == {0, 1, 2}
            self.barrier = True
        if kind == "restore":
            assert p["state"] == self.seals[call.block]["state"]
            self.forks[call.owner] = copy.deepcopy(p["state"])
            return p["state"]
        if kind == "transition":
            self.transitions.setdefault(call.owner, []).append(copy.deepcopy(p["transition"]))
        if kind == "forward":
            assert p["batch"] in (1, 64)
            if p["module"] == "behavior_actor":
                assert call.epoch < 48
        if kind == "episode_complete":
            assert p["transitions"] == 48 and p["artificial_cost"] == sum(range(1, 65))
        return None


class CapacityCampaignTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fake = RecordingFake()
        cls.campaign = CapacityAdaptationCampaign(proposal(), cls.fake)
        cls.result = cls.campaign.run()

    def test_complete_numeric_packet_and_phase_counts(self):
        result, fake = self.result, self.fake
        self.assertEqual(result["status"], "complete")
        self.assertTrue(result["artificial_only"])
        self.assertEqual(result["counts"], numeric_contract(proposal())["limits"])
        self.assertEqual(len(result["completed"]), 288)
        self.assertEqual(result["counts"]["native_steps"], 18432)
        self.assertEqual(result["counts"]["total_native_operations"], 19008)
        self.assertEqual(result["counts"]["total_optimizer_steps"], 7872)
        self.assertEqual(result["counts"]["neural_forward_module_calls"], 25824)
        self.assertEqual(fake.plan_epochs, 1327104)
        self.assertEqual(result["counts"]["planner_candidate_rollouts"], 165888)
        self.assertEqual(result["counts"]["estimator_hypothesis_transitions"], 1843200)
        for phase in proposal()["proposed_budget"]["phases"]:
            for key, value in phase.items():
                if key != "id":
                    self.assertEqual(result["phase_counts"][phase["id"]][key], value)
        self.assertEqual(result["counts"]["extra_explicit_resets_or_clones"], 0)
        self.assertEqual(result["counts"]["historical_model_loads"], 0)

    def test_all_seals_then_same_initial_forks_and_own_history(self):
        fake = self.fake
        self.assertEqual(len(fake.forks), 108)
        self.assertEqual(len(fake.first_controls), 288)
        for owner, first in fake.first_controls.items():
            self.assertEqual(first["history"], [])
            self.assertIsNone(first["filter"])
            if owner in fake.forks:
                self.assertEqual(first["state"], fake.forks[owner])
                self.assertEqual(first["state"], fake.seals[fake.worlds[owner]["block"]]["state"])
        for block in range(3):
            self.assertEqual(self.result["seals"][str(block)], fake.seals[block])

    def test_pairing_integer_seeds_condition_order_and_exploration(self):
        fake = self.fake
        teacher = [w for key, w in fake.worlds.items() if key.startswith("teacher/0/")]
        self.assertEqual([w["condition"] for w in teacher], [0, 1, 2] * 4)
        self.assertEqual([w["replicate"] for w in teacher], [j for j in range(4) for _ in range(3)])
        self.assertEqual([w["seed"] for w in teacher[:3]], [62600000, 62600010, 62600020])
        for owner, world in fake.worlds.items():
            if owner.endswith("/online_matched_fork"):
                paired = owner.replace("online_matched_fork", "frozen_matched_exploration")
                self.assertEqual(world, fake.worlds[paired])
                for epoch in range(48):
                    self.assertEqual(fake.noise[owner, epoch], fake.noise[paired, epoch])

    def test_settled_last_transition_and_only_online_evaluation_updates(self):
        fake = self.fake
        for owner, rows in fake.transitions.items():
            self.assertEqual([r["receipt"] for r in rows], list(range(1, 49)))
            self.assertEqual(rows[-1]["raw_cost"], sum(range(48, 65)))
            self.assertEqual(rows[-1]["available_at"], 64)
            self.assertTrue(rows[-1]["done"])
            self.assertEqual(rows[-1]["reward"], -sum(range(48, 65)) / 100000)
            self.assertTrue(all(not row["done"] for row in rows[:-1]))
        online = [r for r in fake.updates if r[0] == PHASES[2]]
        self.assertEqual(len(online), 2880)
        for owner in fake.forks:
            actor = [r for r in online if r[1] == owner and r[3] == "actor_step"]
            if owner.endswith("/online_matched_fork"):
                self.assertEqual([r[4]["receipt"] for r in actor], list(range(9, 49)))
                self.assertEqual([r[2] for r in actor], list(range(8, 47)) + [63])
            else:
                self.assertEqual(actor, [])
        self.assertFalse(any(48 <= r[2] < 63 for r in fake.updates if r[2] is not None))

    def test_restored_complete_packet_is_not_resumable_or_mutably_aliased(self):
        snapshot = json.loads(json.dumps(self.result))
        fake = RecordingFake()
        restored = CapacityAdaptationCampaign(proposal(), fake)
        restored.load_state_dict(snapshot)
        self.assertEqual(restored.state_dict(), self.result)
        snapshot["counts"]["native_steps"] = 0
        self.assertEqual(restored.state_dict()["counts"]["native_steps"], 18432)
        with self.assertRaises(RuntimeError):
            restored.run()
        self.assertEqual(fake.calls, 0)
        with self.assertRaises(RuntimeError):
            self.campaign.run()


class CapacityCampaignFailureTests(unittest.TestCase):
    def test_science_and_undeclared_backends_fail_closed_before_dispatch(self):
        fake = RecordingFake()
        for changed in ("scientific_execution_authorized", "execution_packet_ready"):
            p = proposal()
            p[changed] = True
            with self.assertRaises(PermissionError):
                CapacityAdaptationCampaign(p, fake)
        p = proposal()
        p["current_scientific_allowance"]["optimizer_steps"] = 1
        with self.assertRaises(PermissionError):
            CapacityAdaptationCampaign(p, fake)
        fake.backend_protocol = "scientific"
        with self.assertRaises(PermissionError):
            CapacityAdaptationCampaign(proposal(), fake)
        self.assertEqual(fake.calls, 0)

    def test_failures_keep_distinct_precharged_calls_and_latch_across_restore(self):
        cases = (("construct", "native_constructions"),
                 ("construction_reset", "construction_triggered_resets"),
                 ("planner_epoch", "planner_total_model_epochs"),
                 ("control_step", "native_steps"))
        for operation, charged in cases:
            with self.subTest(operation=operation):
                fake = RecordingFake(fail=operation)
                campaign = CapacityAdaptationCampaign(proposal(), fake)
                empty = campaign.state_dict()
                with self.assertRaisesRegex(RuntimeError, "injected"):
                    campaign.run()
                failed = campaign.state_dict()
                self.assertEqual(failed["status"], "failed")
                self.assertEqual(failed["counts"][charged], 1)
                if operation == "planner_epoch":
                    self.assertEqual(failed["counts"]["native_steps"], 0)
                    self.assertEqual(failed["counts"]["actor_optimizer_steps"], 0)
                if operation == "construct":
                    self.assertEqual(failed["counts"]["construction_triggered_resets"], 0)
                with self.assertRaises(ValueError):
                    campaign.load_state_dict(empty)
                self.assertEqual(campaign.state_dict(), failed)
                restored_fake = RecordingFake()
                restored = CapacityAdaptationCampaign(proposal(), restored_fake)
                restored.load_state_dict(json.loads(json.dumps(failed)))
                self.assertEqual(restored.state_dict(), failed)
                for owner in (campaign, restored):
                    with self.assertRaises(RuntimeError):
                        owner.run()
                self.assertEqual(restored_fake.calls, 0)

    def test_optimizer_and_forward_debits_precede_error_without_real_steps(self):
        # Exercise the same primitive directly, avoiding repeated teacher packets.
        for operation, counter in (("actor_step", "actor_optimizer_steps"),
                                   ("critic_step", "critic_optimizer_steps"),
                                   ("forward", "neural_forward_module_calls")):
            fake = RecordingFake(fail=operation)
            campaign = CapacityAdaptationCampaign(proposal(), fake)
            campaign._status = "running"
            state = dict.fromkeys(STATE_KEYS, 0)
            with self.assertRaises(RuntimeError):
                campaign._update(state, "ddpg")
            self.assertEqual(campaign.state_dict()["counts"][counter], 1)
            before = campaign.state_dict()["counts"].copy()
            self.assertEqual(fake.last_kind, operation)
            self.assertEqual(campaign.state_dict()["counts"], before)

    def test_actor_failure_in_serial_run_retains_ledger_and_training_state(self):
        class PrechargeFake(RecordingFake):
            def dispatch(self, call):
                if call.kind == "actor_step":
                    assert self.campaign._counts["actor_optimizer_steps"] == 1
                    assert self.campaign._counts["critic_optimizer_steps"] == 0
                    assert self.campaign._counts["optimizer_example_presentations"] == 64
                return super().dispatch(call)
        fake = PrechargeFake(fail="actor_step")
        campaign = CapacityAdaptationCampaign(proposal(), fake)
        fake.campaign = campaign
        with self.assertRaisesRegex(RuntimeError, "injected"):
            campaign.run()
        failed = campaign.state_dict()
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["counts"]["actor_optimizer_steps"], 1)
        self.assertEqual(failed["training_state"]["replay"], 576)
        restored = CapacityAdaptationCampaign(proposal(), RecordingFake())
        restored.load_state_dict(json.loads(json.dumps(failed)))
        self.assertEqual(restored.state_dict(), failed)
        with self.assertRaises(RuntimeError):
            restored.run()

    def test_query_cap_rejects_before_second_dispatch_and_closes_attempt(self):
        fake = RecordingFake()
        campaign = CapacityAdaptationCampaign(proposal(), fake)
        campaign._contract["limits"]["planner_total_model_epochs"] = 1
        with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
            campaign.run()
        state = campaign.state_dict()
        self.assertEqual(state["status"], "failed")
        self.assertEqual(fake.plan_epochs, 1)
        self.assertEqual(state["counts"]["planner_total_model_epochs"], 1)
        self.assertEqual(state["counts"]["native_steps"], 0)
        self.assertEqual(state["counts"]["total_optimizer_steps"], 0)
        with self.assertRaises(RuntimeError):
            campaign.run()

    def test_fork_mutation_is_rejected_and_restoration_is_consumed(self):
        class BadForkFake(RecordingFake):
            def dispatch(self, call):
                if call.kind == "restore":
                    call.payload["state"]["critic_optimizer"] += 1
                    return call.payload["state"]
                if call.kind == "construct":
                    return None
                return super().dispatch(call)
        fake = BadForkFake()
        campaign = CapacityAdaptationCampaign(proposal(), fake)
        campaign._status, campaign._phase = "running", PHASES[2]
        world = dict(phase="evaluation", block=0, condition=0, replicate=0, seed=62610000)
        initial = dict.fromkeys(STATE_KEYS, 0)
        with self.assertRaisesRegex(ValueError, "identical complete seal"):
            campaign._episode(world, "online_matched_fork", initial)
        self.assertEqual(initial, dict.fromkeys(STATE_KEYS, 0))
        self.assertEqual(campaign.state_dict()["counts"]["learned_evaluation_arm_restorations"], 1)

    def test_unsettled_tail_stops_without_last_update_or_extra_epoch(self):
        fake = RecordingFake(unsettled=True)
        campaign = CapacityAdaptationCampaign(proposal(), fake)
        with self.assertRaisesRegex(ValueError, "unsettled"):
            campaign.run()
        state = campaign.state_dict()
        self.assertEqual(state["counts"]["native_steps"], 64)
        self.assertEqual(state["counts"]["tail_steps"], 16)
        self.assertEqual(state["counts"]["total_optimizer_steps"], 0)
        self.assertEqual(state["session"]["transitions"], 47)
        self.assertEqual(len(state["session"]["history"]), 64)
        self.assertEqual(len(state["completed"]), 0)

    def test_invalid_receipt_consumes_step_and_snapshot_zero_ledger_roundtrips(self):
        class InvalidReceiptFake(RecordingFake):
            def dispatch(self, call):
                value = super().dispatch(call)
                return ArtificialReceipt(999, {}, 1.0) if call.kind == "control_step" else value
        campaign = CapacityAdaptationCampaign(proposal(), InvalidReceiptFake())
        snapshot = json.loads(json.dumps(campaign.state_dict()))
        restored = CapacityAdaptationCampaign(proposal(), RecordingFake())
        restored.load_state_dict(snapshot)
        self.assertEqual(restored.state_dict(), snapshot)
        with self.assertRaises(RuntimeError):
            restored.run()
        with self.assertRaises(ValueError):
            campaign.run()
        self.assertEqual(campaign.state_dict()["counts"]["native_steps"], 1)

    def test_numeric_mismatch_string_seed_and_non_json_state_are_rejected(self):
        for path in (("proposed_budget", "planner_total_model_epochs"),
                     ("proposed_budget", "actor_optimizer_steps")):
            p = proposal()
            p[path[0]][path[1]] += 1
            with self.assertRaises(ValueError):
                numeric_contract(p)
        p = proposal()
        p["design"]["training_seeds"][0] = "520260301"
        with self.assertRaises(ValueError):
            numeric_contract(p)
        class BadStateFake(RecordingFake):
            def dispatch(self, call):
                return {k: object() for k in STATE_KEYS}
        campaign = CapacityAdaptationCampaign(proposal(), BadStateFake())
        with self.assertRaises(TypeError):
            campaign.run()
        self.assertEqual(campaign.state_dict()["counts"]["fresh_actor_critic_pairs"], 1)

    def test_import_and_rehearsal_do_not_import_scientific_models(self):
        # Fresh interpreter: transitive imports must not load the neural stack.
        result = subprocess.run([sys.executable, "-c",
            "import sys; import src.rl.capacity_adaptation_campaign; "
            "assert 'torch' not in sys.modules; "
            "assert 'src.rl.networks' not in sys.modules"], cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
