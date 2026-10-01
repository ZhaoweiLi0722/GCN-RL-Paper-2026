"""Invented saved states and public records; no environment or optimizer calls.

Nonzero saved optimizer counters/moments below are handwritten test payloads,
not evidence of fitting. No research checkpoint is loaded by this module.
"""

import copy
from dataclasses import asdict
import importlib
import unittest
from unittest.mock import patch

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.models.matched_inputs import InputSchema, ObservationBatch
from src.rl.candidate_imitation import ImitationSettings, public_example
from src.rl.candidate_pilot_resources import digest
from src.rl.dynamic_candidate_imitation import DynamicCandidateImitationKernel
from src.rl.dynamic_candidate_resources import dynamic_stream_manifest
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.networks import torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.routing_candidate_contract import RoutingRequestSchema, build_request_candidates
from src.rl.validated_returns import ReplaySemantics
from tests.test_dynamic_candidate_factory import proposal


def synthetic_saved_fixture():
    """Construct three independent tiny policies and invented saved boundaries."""
    config = proposal()
    streams = dynamic_stream_manifest(config)
    config["objective"].update(horizon=2, num_facilities=2, action_width=8,
                               transfer_scale=4., reward_scale=.01)
    config["model_proposal"].update(encoder_width=3, head_width=5, initial_reference_bias=10.)
    for owner in ("actor", "critic"):
        config["model_proposal"].pop(owner + "_parameter_count", None)
    config["qualification"]["fresh_worlds_per_block"] = 1
    for block in config["blocks"]:
        streams["environment"][str(block)]["qualification"] = streams["environment"][str(block)]["qualification"][:1]
    schema = InputSchema("invented-saved-qualification-v1", ("A", "B"),
                         ("demand", "specimens", "reagents", "capacity"), ("time",),
                         tuple(f"request_{i}" for i in range(8)))
    contract = ReplayInputContract(schema, ReplaySemantics(
        "absolute_environment", "invented-cost-v1", .01, 1.,
        schema.definition_id + "/actor-flat", schema.definition_id + "/action", 21, 8, True))
    states, policies, examples, raw_episodes, outcomes = {}, {}, {}, [], []
    for block in config["blocks"]:
        prefix = f"block{block}/graph"
        policy = DynamicCandidatePolicy(schema, enabled=True, architecture="graph",
            message_mode="physical", encoder_width=3, head_width=5,
            actor_seed=streams["neural"][prefix + "/actor_initialization"],
            critic_seed=streams["neural"][prefix + "/critic_initialization"],
            initial_reference_bias=10.)
        settings = ImitationSettings(config["optimizer"]["learning_rate"],
            config["optimizer"]["gradient_norm_cap_each_owner"],
            config["initialization"]["batch_size"],
            config["initialization"]["actor_adam_calls_per_block"], 1, "demonstration")
        owner = DynamicCandidateImitationKernel(policy, contract, settings, enabled=True,
            shuffle_seed=streams["neural"][prefix + "/bc_init/shuffle"])
        state = owner.state_dict()
        # These counters and zero moments describe an artificial load fixture.
        state["steps"] = settings.max_optimizer_steps
        state["history"] = [{"steps": state["steps"], "identities": [f"invented-demo/{block}/0"],
                             "examples_sha256": "a" * 64}]
        indices = state["optimizer"]["param_groups"][0]["params"]
        for index, parameter in zip(indices, policy.actor_parameters()):
            state["optimizer"]["state"][index] = {
                "step": torch.tensor(float(state["steps"])),
                "exp_avg": torch.zeros_like(parameter), "exp_avg_sq": torch.zeros_like(parameter)}
        states[prefix], policies[prefix] = state, policy
        examples[block] = {}
        seed = streams["environment"][str(block)]["qualification"][0]
        for role in ("r4", "initializer_greedy"):
            trajectory = f"qualification/block{block}/{role}/world00"
            rows, path = [], []
            for step in range(2):
                token = f"invented-token/{block}/{role}/{step}"
                observation = ObservationBatch(schema,
                    torch.tensor([[[1., 2., 3., 4.], [2., 1., 4., 3.]]]),
                    torch.tensor([[step / 2.]]), torch.tensor([[[0., 1.], [1., 0.]]]))
                bank = build_request_candidates({"state_token": token,
                    "reference_request": [.25, -.25, 0., 0., 0., 0., .5, .5],
                    "anchor_request": [0., 0., 0., 0., 0., 0., .5, .5],
                    "option_requests": [[.5, -.5, 0., 0., 0., 0., .5, .5]]},
                    RoutingRequestSchema(schema.definition_id + "/action", 2, 4., 6))
                example = public_example(observation, bank, contract, split="qualification",
                                         identity=f"{trajectory}/{step}")
                evaluation = asdict(evaluate_dynamic_policy(policy, observation, bank, contract))
                path.append(example)
                rows.append({"event": {"audit": {"record": {
                    "source_id": "invented-saved-only", "trajectory_id": trajectory,
                    "step_index": step, "state_token": token, "state": list(example["actor_state"])},
                    "decision": {"evaluation": evaluation}}}})
            header = {"block": block, "role": role, "world_index": 0, "seed": seed,
                "split": "qualification", "trajectory_id": trajectory,
                "source_id": "invented-saved-only", "policy_sha256": policy.snapshot_sha256(),
                "selection": "reference" if role == "r4" else "greedy",
                "representation": "reference" if role == "r4" else "graph",
                "session_manifest": {"contract": asdict(contract)}}
            files = {name: {"path": f"invented/{trajectory}/{name}.json",
                            "sha256": digest(value), "bytes": 1}
                     for name, value in (("header", header), ("events", rows), ("final_state", {}))}
            raw_episodes.append({"header": header, "rows": rows, "files": files})
            examples[block][role] = path
            outcomes.append({"block": block, "role": role, "world_index": 0, "seed": seed,
                "cost": 100., "losses": 1., "completions": 2., "terminal_active": 3.})
    boundary = {"format": "dynamic-campaign-v1", "config_sha256": digest(config),
        "streams_sha256": digest(streams), "failure": None, "initializers": copy.deepcopy(states),
        "qualification": copy.deepcopy(examples), "qualified": {}, "models": {}, "model_paths": {},
        "test_index": [], "payload_seal": None, "live_session": None, "clone": None,
        "continuation": None, "recorder": None,
        "raw_index": [copy.deepcopy(episode["files"]) for episode in raw_episodes],
        "work": {"job": "qualification", "cursor": len(raw_episodes),
                 "indexes": [copy.deepcopy(episode["files"]) for episode in raw_episodes],
                 "evidence": [], "updates": []},
        "sequence": {"active": "qualification"}}
    return config, streams, states, policies, examples, raw_episodes, outcomes, boundary


@unittest.skipIf(torch is None, "torch unavailable")
class SavedQualificationTests(unittest.TestCase):
    def setUp(self):
        for optimizer in (torch.optim.Adam, torch.optim.SGD):
            guard = patch.object(optimizer, "step", side_effect=AssertionError("optimizer step forbidden"))
            sentinel = guard.start()
            self.addCleanup(guard.stop)
            self.addCleanup(sentinel.assert_not_called)
        loader = patch.object(torch, "load", side_effect=AssertionError("checkpoint loading forbidden"))
        sentinel = loader.start()
        self.addCleanup(loader.stop)
        self.addCleanup(sentinel.assert_not_called)
        self.api = importlib.import_module("src.rl.dynamic_candidate_saved_qualification")
        (self.config, self.streams, self.states, self.policies, self.examples,
         self.raw, self.outcomes, self.boundary) = synthetic_saved_fixture()

    def restore(self, state=None, *, block=60, config=None, streams=None):
        return self.api.restore_saved_policy(self.states[f"block{block}/graph"] if state is None else state,
            self.config if config is None else config,
            self.streams if streams is None else streams, block)

    def test_restore_exact_state_cpu_float32_without_mutating_input_or_global_rng(self):
        saved = copy.deepcopy(self.states["block60/graph"])
        before = state_digest(saved)
        rng = torch.get_rng_state().clone()
        owner = self.restore(saved)
        self.assertEqual(state_digest(saved), before)
        self.assertEqual(state_digest(owner.state_dict()), before)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(owner.policy.snapshot_sha256(), self.policies["block60/graph"].snapshot_sha256())
        self.assertTrue(all(p.device.type == "cpu" and p.dtype == torch.float32
                            for p in owner.policy.parameters()))

    def test_restore_rejects_incomplete_counter_and_terminal_failure(self):
        for name, value in (("steps", 0), ("steps", True), ("steps", 257),
                            ("failure", {"error_type": "ArtificialFailure"})):
            saved = copy.deepcopy(self.states["block60/graph"])
            saved[name] = value
            with self.subTest(field=name, value=value), self.assertRaises(ValueError):
                self.restore(saved)

    def test_restore_rejects_wrong_block_and_declared_seeds(self):
        with self.assertRaises(ValueError):
            self.api.restore_saved_policy(self.states["block60/graph"], self.config, self.streams, 61)
        for suffix in ("actor_initialization", "critic_initialization", "bc_init/shuffle"):
            streams = copy.deepcopy(self.streams)
            streams["neural"]["block60/graph/" + suffix] += 1
            with self.subTest(seed=suffix), self.assertRaises(ValueError):
                self.restore(streams=streams)

    def test_restore_rejects_model_and_optimizer_setting_drift(self):
        for section, name, value in (("model_proposal", "encoder_width", 4),
                                      ("model_proposal", "head_width", 6),
                                      ("model_proposal", "initial_reference_bias", 11.),
                                      ("optimizer", "learning_rate", .123)):
            config = copy.deepcopy(self.config)
            config[section][name] = value
            with self.subTest(field=name), self.assertRaises(ValueError):
                self.restore(config=config)

    def test_restore_rejects_missing_extra_wrong_shape_dtype_and_nonfinite_tensor(self):
        baseline = self.states["block60/graph"]
        name = next(iter(baseline["policy"]))
        variants = []
        missing = copy.deepcopy(baseline)
        del missing["policy"][name]
        variants.append(missing)
        extra = copy.deepcopy(baseline)
        extra["policy"]["unexpected"] = torch.zeros(1)
        variants.append(extra)
        for value in (baseline["policy"][name].double(), torch.zeros(3), torch.tensor(float("nan"))):
            bad = copy.deepcopy(baseline)
            bad["policy"][name] = value
            variants.append(bad)
        for saved in variants:
            with self.subTest(keys=tuple(saved["policy"])), self.assertRaises(ValueError):
                self.restore(saved)

    def test_snapshot_reports_live_model_mutations_not_only_the_saved_copy(self):
        owner = self.restore()
        before = state_digest(owner.state_dict())
        with torch.no_grad():
            next(owner.policy.parameters()).add_(.125)
        self.assertNotEqual(state_digest(owner.state_dict()), before)
        self.assertEqual(state_digest(self.states["block60/graph"]), before)

    def test_restore_constructs_no_optimizer_and_freezes_all_parameters(self):
        with patch.object(torch.optim.Adam, "__init__", side_effect=AssertionError("optimizer construction forbidden")):
            owner = self.restore()
        self.assertFalse(hasattr(owner, "optimizer"))
        self.assertFalse(owner.policy.training)
        self.assertTrue(all(not parameter.requires_grad for parameter in owner.policy.parameters()))

    def test_restore_rejects_changed_frozen_critic(self):
        saved = copy.deepcopy(self.states["block60/graph"])
        name = next(name for name in saved["policy"] if name.startswith("critic."))
        saved["policy"][name].add_(.125)
        with self.assertRaisesRegex(ValueError, "frozen critic"):
            self.restore(saved)

    def owners(self):
        return {block: self.restore(block=block) for block in self.config["blocks"]}

    def validate(self, *, boundary=None, raw=None, owners=None):
        return self.api.validate_saved_boundary(
            self.boundary if boundary is None else boundary,
            self.owners() if owners is None else owners,
            self.raw if raw is None else raw, self.config, self.streams)

    def test_boundary_exact_lineage_validation_does_not_modify_inputs(self):
        before = state_digest([self.boundary, self.raw, self.states])
        self.assertEqual(self.validate(), self.examples)
        self.assertEqual(state_digest([self.boundary, self.raw, self.states]), before)

    def test_boundary_rejects_wrong_config_stream_or_terminal_resume(self):
        for field, value in (("config_sha256", "0" * 64), ("streams_sha256", "1" * 64),
                             ("failure", {"message": "invented terminal failure"}),
                             ("live_session", {"index": 1}), ("test_index", ["opened test"]),
                             ("models", {"continued": {}})):
            boundary = copy.deepcopy(self.boundary)
            boundary[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(boundary=boundary)

    def test_boundary_rejects_missing_extra_and_duplicate_paths(self):
        variants = [self.raw[:-1], self.raw + [copy.deepcopy(self.raw[0])]]
        repeated = copy.deepcopy(self.raw)
        repeated[1] = copy.deepcopy(repeated[0])
        variants.append(repeated)
        for raw in variants:
            with self.subTest(count=len(raw)), self.assertRaises(ValueError):
                self.validate(raw=raw)

    def test_boundary_rejects_raw_seed_split_source_and_model_lineage_drift(self):
        for field, value in (("seed", 123), ("split", "test"), ("source_id", "different-source"),
                             ("policy_sha256", "b" * 64), ("world_index", 2)):
            raw = copy.deepcopy(self.raw)
            raw[0]["header"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(raw=raw)

    def test_boundary_rejects_example_identity_split_state_token_and_support_drift(self):
        for field in ("identity", "split", "actor_state", "state_token", "requests"):
            boundary = copy.deepcopy(self.boundary)
            example = boundary["qualification"][60]["r4"][0]
            if field == "identity":
                example[field] += "-stale"
            elif field == "split":
                example[field] = "demonstration"
            elif field == "actor_state":
                example[field] = (example[field][0] + 1.,) + example[field][1:]
            elif field == "state_token":
                example["candidates"][field] = "different-state"
            else:
                requests = list(example["candidates"][field])
                requests[0] = (.5, -.5) + requests[0][2:]
                example["candidates"][field] = tuple(requests)
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(boundary=boundary)

    def test_boundary_rejects_raw_record_drift_even_when_header_is_correct(self):
        for field, value in (("source_id", "different-source"), ("trajectory_id", "different-path"),
                             ("step_index", 7), ("state_token", "different-token"), ("state", [0.])):
            raw = copy.deepcopy(self.raw)
            raw[0]["rows"][0]["event"]["audit"]["record"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.validate(raw=raw)

    def test_boundary_rejects_partial_or_changed_saved_initializer(self):
        boundary = copy.deepcopy(self.boundary)
        boundary["initializers"]["block60/graph"]["policy"]["actor.reference_bias"].add_(1.)
        with self.assertRaises(ValueError):
            self.validate(boundary=boundary)
        boundary = copy.deepcopy(self.boundary)
        boundary["work"]["cursor"] -= 1
        with self.assertRaises(ValueError):
            self.validate(boundary=boundary)

    def test_boundary_rejects_index_not_preserved_in_raw_inventory(self):
        for inventory in ("raw_index", "indexes"):
            boundary = copy.deepcopy(self.boundary)
            target = boundary if inventory == "raw_index" else boundary["work"]
            target[inventory] = target[inventory][:-1]
            with self.subTest(inventory=inventory), self.assertRaises(ValueError):
                self.validate(boundary=boundary)

    def score(self, *, owners=None, examples=None, outcomes=None, before_forward=None):
        return self.api.score_saved_qualification(
            self.owners() if owners is None else owners,
            self.examples if examples is None else examples,
            self.outcomes if outcomes is None else outcomes, self.config,
            before_forward=(lambda: None) if before_forward is None else before_forward)

    def test_scoring_debits_before_each_forward_and_preserves_every_saved_state(self):
        owners, actions = self.owners(), []
        before = {key: state_digest(owner.state_dict()) for key, owner in owners.items()}
        def debit():
            actions.append("debit")
        def forward(*args, **kwargs):
            self.assertEqual(actions[-1], "debit")
            actions.append("forward")
            return evaluate_dynamic_policy(*args, **kwargs)
        with patch("src.rl.dynamic_candidate_factory.evaluate_dynamic_policy", side_effect=forward):
            result = self.score(owners=owners, before_forward=debit)
        self.assertEqual(actions, [item for _ in range(12) for item in ("debit", "forward")])
        self.assertEqual(before, {key: state_digest(owner.state_dict()) for key, owner in owners.items()})
        self.assertIsInstance(result, dict)
        self.assertTrue(result["passed"])
        self.assertFalse(result["rl_performance_claim"])
        self.assertFalse(result["clinical_noninferiority_claim"])
        self.assertFalse(result["automatic_training_authorized"])
        for block in result["blocks"].values():
            self.assertEqual(set(block["paths"]), {"r4", "initializer_greedy"})
            for path in block["paths"].values():
                self.assertEqual(path["rows"], 2)
                self.assertEqual(path["agreement"], 1.)
                self.assertEqual(path["multiclass_agreement"], 1.)

    def test_failed_debit_prevents_forward_and_propagates_without_retry(self):
        debit = unittest.mock.Mock(side_effect=TimeoutError("artificial budget stop"))
        with patch("src.rl.dynamic_candidate_factory.evaluate_dynamic_policy") as forward:
            with self.assertRaisesRegex(TimeoutError, "artificial budget stop"):
                self.score(before_forward=debit)
        debit.assert_called_once_with()
        forward.assert_not_called()

    def test_scoring_detects_policy_mutation(self):
        owners = self.owners()
        def corrupt(policy, *args, **kwargs):
            result = evaluate_dynamic_policy(policy, *args, **kwargs)
            with torch.no_grad():
                next(policy.parameters()).add_(.125)
            return result
        with patch("src.rl.dynamic_candidate_factory.evaluate_dynamic_policy", side_effect=corrupt):
            with self.assertRaises(ValueError):
                self.score(owners=owners)

    def test_scoring_rejects_missing_or_duplicate_examples(self):
        for change in ("missing", "duplicate", "test_split"):
            examples = copy.deepcopy(self.examples)
            rows = examples[60]["r4"]
            if change == "missing":
                rows.pop()
            elif change == "duplicate":
                rows[1] = copy.deepcopy(rows[0])
            else:
                rows[0]["split"] = "test"
            with self.subTest(change=change), self.assertRaises(ValueError):
                self.score(examples=examples)

    def test_scoring_rejects_nonfinite_outcomes_and_wrong_pairs(self):
        for field, value in (("cost", float("nan")), ("seed", 0)):
            outcomes = copy.deepcopy(self.outcomes)
            outcomes[1][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.score(outcomes=outcomes)

    def test_patient_tradeoff_is_a_failed_qualification_not_hidden_by_cost_gain(self):
        outcomes = copy.deepcopy(self.outcomes)
        outcomes[1]["cost"] -= 10.
        outcomes[1]["losses"] += 1.
        result = self.score(outcomes=outcomes)
        self.assertFalse(result["passed"])
        self.assertFalse(result["blocks"]["60"]["outcomes"]["passed"])
        self.assertEqual(result["blocks"]["60"]["outcomes"]["paired_deltas"]["losses"], [1.])
        self.assertFalse(result["automatic_training_authorized"])

    def test_path_receipts_are_published_serially_and_publication_failure_stops(self):
        owners, receipts, debits = self.owners(), [], []
        def on_path(block, role, result):
            receipts.append((block, role, copy.deepcopy(result)))
            raise OSError("invented receipt publication failure")
        with self.assertRaisesRegex(OSError, "invented receipt publication failure"):
            self.api.score_saved_qualification(owners, self.examples, self.outcomes, self.config,
                before_forward=lambda: debits.append("debit"), on_path=on_path)
        self.assertEqual(len(debits), 2)
        self.assertEqual([(b, r) for b, r, _ in receipts], [(60, "r4")])
        self.assertEqual(receipts[0][2]["rows"], 2)


if __name__ == "__main__":
    unittest.main()
