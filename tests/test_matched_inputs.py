"""Synthetic contract checks only: no environment, checkpoint or optimizer."""

from dataclasses import replace
import unittest

from src.models.matched_inputs import (
    InputSchema, ObservationBatch, build_matched_inputs, parameter_inventory,
)
from src.rl.networks import torch, nn


def schema(**changes):
    values = dict(definition_id="synthetic-inputs-v1", node_ids=("A", "B", "C"),
                  node_feature_names=("pipeline_0", "pipeline_1"),
                  global_feature_names=("time",), action_names=("support_A", "support_B", "support_C"))
    return InputSchema(**(values | changes))


@unittest.skipIf(torch is None, "torch not available")
class MatchedInputTests(unittest.TestCase):
    def setUp(self):
        self.schema = schema()
        self.observation = ObservationBatch(
            self.schema, torch.arange(12, dtype=torch.float32).reshape(2, 3, 2),
            torch.tensor([[0.0], [1.0]]),
            torch.tensor([[[0., 2., 0.], [2., 0., 1.], [0., 1., 0.]],
                          [[0., 1., 0.], [1., 0., 0.], [0., 0., 0.]]]))
        self.anchor = torch.tensor([[.1, .2, .3], [.4, .5, .6]])
        self.action = self.anchor + .125

    def build(self, role="actor", **changes):
        args = dict(role=role, message_mode="physical")
        if role != "actor":
            args["action"] = self.action
        if role == "gate":
            args["detach_proposal"] = True
        args.update(changes)
        return build_matched_inputs(self.observation, self.schema, self.anchor, **args)

    def test_graph_and_flat_have_exactly_the_same_values(self):
        for role in ("actor", "critic", "gate"):
            views = self.build(role)
            self.assertTrue(torch.equal(views.nodes, self.observation.nodes))
            self.assertTrue(torch.equal(views.flat[:, :6].reshape(2, 3, 2), views.nodes))
            self.assertTrue(torch.equal(views.flat[:, 6:], views.context))
            self.assertTrue(torch.equal(views.context[:, :1], self.observation.globals))
            self.assertTrue(torch.equal(views.context[:, 1:10], self.observation.physical_links.reshape(2, -1)))
            self.assertTrue(torch.equal(views.context[:, 10:13], self.anchor))

    def test_no_pipeline_slot_is_summed_away(self):
        original = self.build()
        changed = self.observation.nodes.clone()
        changed[:, 0, 0] += 1
        changed[:, 0, 1] -= 1
        self.observation = replace(self.observation, nodes=changed)
        views = self.build()
        self.assertFalse(torch.equal(views.nodes, original.nodes))
        self.assertFalse(torch.equal(views.flat, original.flat))

    def test_critic_receives_both_anchor_and_candidate(self):
        views = self.build("critic")
        self.assertTrue(torch.equal(views.context[:, -3:], self.action))
        changed = self.build("critic", action=self.action + .1)
        self.assertTrue(torch.equal(changed.context[:, :-3], views.context[:, :-3]))
        self.assertFalse(torch.equal(changed.context[:, -3:], views.context[:, -3:]))

    def test_gate_proposal_changes_both_views_in_the_same_coordinates(self):
        views = self.build("gate")
        torch.testing.assert_close(views.context[:, -3:], self.action - self.anchor)
        changed = self.build("gate", action=self.action + .25)
        torch.testing.assert_close(changed.context[:, -3:] - views.context[:, -3:],
                                   torch.full((2, 3), .25))
        self.assertFalse(torch.equal(changed.flat, views.flat))

    def test_explicit_gate_detach_and_critic_action_gradients(self):
        for role, detach in (("critic", None), ("gate", False), ("gate", True)):
            action = self.action.clone().requires_grad_()
            settings = {"action": action}
            if role == "gate":
                settings["detach_proposal"] = detach
            views = self.build(role, **settings)
            for value in (views.context, views.flat):
                if detach:
                    self.assertFalse(value.requires_grad)
                else:
                    gradient = torch.autograd.grad(value.sum(), action, retain_graph=True)[0]
                    torch.testing.assert_close(gradient, torch.ones_like(action))

    def test_actor_rejects_circular_proposal_and_implicit_gate_policy(self):
        for kwargs in ({"action": self.action}, {"detach_proposal": True}):
            with self.assertRaisesRegex(ValueError, "actor"):
                self.build(**kwargs)
        for setting in (None, 0, "false"):
            with self.assertRaisesRegex(ValueError, "explicit"):
                self.build("gate", detach_proposal=setting)
        with self.assertRaisesRegex(ValueError, "critic"):
            self.build("critic", detach_proposal=True)

    def test_neural_ablation_keeps_physical_information_and_actions(self):
        for role in ("actor", "critic", "gate"):
            full, ablated = self.build(role), self.build(role, message_mode="self_only")
            for name in ("nodes", "context", "flat"):
                self.assertTrue(torch.equal(getattr(full, name), getattr(ablated, name)))
            self.assertFalse(torch.equal(full.message_adjacency, ablated.message_adjacency))
            torch.testing.assert_close(ablated.message_adjacency, torch.eye(3).expand(2, -1, -1))

    def test_message_normalization_agrees_with_existing_gcn_helper(self):
        from src.models.gcn import build_normalized_adjacency
        views = self.build()
        for batch in range(2):
            links = self.observation.physical_links[batch]
            edges = [(i, j) for i in range(3) for j in range(i + 1, 3) if links[i, j] > 0]
            expected = build_normalized_adjacency(3, edges, edge_weights=[links[i, j].item() for i, j in edges])
            torch.testing.assert_close(views.message_adjacency[batch], expected)

    def test_all_declared_ordering_and_definition_changes_are_rejected(self):
        alternatives = [schema(definition_id="different-scaling-v1"),
                        schema(node_ids=("B", "A", "C")),
                        schema(node_feature_names=("pipeline_1", "pipeline_0")),
                        schema(global_feature_names=("future_time",)),
                        schema(action_names=("support_C", "support_B", "support_A"))]
        for other in alternatives:
            with self.subTest(schema=other), self.assertRaisesRegex(ValueError, "schema"):
                build_matched_inputs(replace(self.observation, schema=other), self.schema,
                                     self.anchor, role="actor", message_mode="physical")

    def test_no_implicit_broadcasting_or_missing_proposals(self):
        alternatives = [replace(self.observation, nodes=self.observation.nodes[0]),
                        replace(self.observation, globals=torch.ones(1, 1)),
                        replace(self.observation, physical_links=self.observation.physical_links[0]),
                        replace(self.observation, nodes=torch.zeros(0, 3, 2))]
        for other in alternatives:
            with self.assertRaises(ValueError):
                build_matched_inputs(other, self.schema, self.anchor, role="actor", message_mode="physical")
        for action in (None, self.action[0]):
            with self.assertRaises((ValueError, TypeError)):
                self.build("critic", action=action)

    def test_bad_types_and_nonfinite_inputs_fail(self):
        for value in (self.anchor.double(), self.anchor.int(), self.anchor.tolist(),
                      torch.full_like(self.anchor, float("nan")), torch.full_like(self.anchor, float("inf"))):
            with self.subTest(value=value), self.assertRaises((ValueError, TypeError)):
                build_matched_inputs(self.observation, self.schema, value, role="actor", message_mode="physical")
        for name in ("nodes", "globals", "physical_links"):
            invalid = torch.full_like(getattr(self.observation, name), float("nan"))
            with self.assertRaisesRegex(ValueError, "finite"):
                build_matched_inputs(replace(self.observation, **{name: invalid}), self.schema,
                                     self.anchor, role="actor", message_mode="physical")
        with self.assertRaisesRegex(ValueError, "dtype/device"):
            build_matched_inputs(self.observation, self.schema, self.anchor.to("meta"),
                                 role="actor", message_mode="physical")

    def test_noncontiguous_views_keep_batch_and_feature_order(self):
        nodes = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3).transpose(1, 2)
        self.assertFalse(nodes.is_contiguous())
        self.observation = replace(self.observation, nodes=nodes)
        views = self.build()
        for batch in range(2):
            torch.testing.assert_close(views.flat[batch, :6], nodes[batch].reshape(-1))
        torch.testing.assert_close(views.context[:, 10:13], self.anchor)

    def test_prototype_does_not_silently_clip_or_scale_actions(self):
        proposed = self.action * 20
        gate = self.build("gate", action=proposed)
        critic = self.build("critic", action=proposed)
        torch.testing.assert_close(gate.context[:, -3:], proposed - self.anchor)
        torch.testing.assert_close(critic.context[:, -3:], proposed)

    def test_physical_link_contract_rejects_directed_negative_and_diagonal(self):
        for index, value in (((0, 0, 1), 3.), ((0, 0, 1), -1.), ((0, 0, 0), 1.)):
            links = self.observation.physical_links.clone()
            links[index] = value
            with self.assertRaisesRegex(ValueError, "physical_links"):
                build_matched_inputs(replace(self.observation, physical_links=links), self.schema,
                                     self.anchor, role="actor", message_mode="physical")

    def test_unknown_modes_and_legacy_inputs_rejected(self):
        for kwargs in ({"message_mode": "remove_physical_edges"}, {"role": "guess"}):
            with self.assertRaises(ValueError):
                self.build(**kwargs)
        with self.assertRaises(TypeError):
            build_matched_inputs({}, self.schema, self.anchor, role="actor", message_mode="physical")

    def test_outputs_do_not_alias_caller_data(self):
        views = self.build()
        views.nodes[0, 0, 0] = 99
        views.context[0, 0] = 99
        views.flat[:] = 99
        self.assertEqual(self.observation.nodes[0, 0, 0].item(), 0)
        self.assertEqual(self.observation.globals[0, 0].item(), 0)
        self.assertAlmostEqual(self.anchor[0, 0].item(), .1)

    def test_empty_globals_and_single_node_without_links(self):
        contract = schema(node_ids=("A",), global_feature_names=(), action_names=("support_A",))
        data = ObservationBatch(contract, torch.ones(1, 1, 2, dtype=torch.float64),
                                torch.empty(1, 0, dtype=torch.float64), torch.zeros(1, 1, 1, dtype=torch.float64))
        views = build_matched_inputs(data, contract, torch.zeros(1, 1, dtype=torch.float64),
                                     role="actor", message_mode="physical")
        self.assertEqual(views.flat.shape, (1, 4))
        self.assertEqual(views.flat.dtype, torch.float64)
        self.assertEqual(views.message_adjacency.item(), 1)

    def test_overflow_is_not_hidden_by_normalization(self):
        links = torch.full((2, 3, 3), torch.finfo(torch.float32).max)
        links.diagonal(dim1=1, dim2=2).zero_()
        with self.assertRaisesRegex(ValueError, "overflow"):
            build_matched_inputs(replace(self.observation, physical_links=links), self.schema,
                                 self.anchor, role="actor", message_mode="physical")
        with self.assertRaisesRegex(ValueError, "finite"):
            build_matched_inputs(self.observation, self.schema, -torch.full_like(self.anchor, 3e38),
                                 action=torch.full_like(self.anchor, 3e38), role="gate",
                                 detach_proposal=True, message_mode="physical")

    def test_synthetic_forward_consumer_uses_common_heads(self):
        from src.models.gcn import GraphConvolution
        layer = GraphConvolution(2, 2)
        head = nn.Linear(19, 3)
        with torch.no_grad():
            layer.linear.weight.copy_(torch.eye(2))
            layer.linear.bias.zero_()
            head.weight.fill_(.1)
            head.bias.zero_()
        identity = self.build(message_mode="self_only")
        encoded = layer(identity.nodes, identity.message_adjacency)
        graph_values = torch.cat((encoded.flatten(1), identity.context), dim=1)
        torch.testing.assert_close(head(graph_values), head(identity.flat))
        physical = self.build()
        encoded = layer(physical.nodes, physical.message_adjacency)
        self.assertFalse(torch.equal(encoded, physical.nodes))
        self.assertTrue(torch.equal(physical.flat, identity.flat))

    def test_parameter_counts_deduplicate_shared_weights_and_exclude_buffers(self):
        actor = nn.Linear(2, 3)
        critic = nn.Linear(2, 3)
        critic.weight = actor.weight
        actor.register_buffer("synthetic_metadata", torch.ones(99))
        actor.bias.requires_grad_(False)
        gate = nn.Linear(3, 1)
        counts = parameter_inventory(actor=actor, critic=critic, gate=gate)
        self.assertEqual(counts["components"], {"actor": 9, "critic": 9, "gate": 4})
        self.assertEqual(counts["total_unique"], 16)
        self.assertEqual(counts["trainable_unique"], 13)
        self.assertEqual(counts["shared_duplicate_numel"], 6)
        self.assertEqual(parameter_inventory(actor=actor, critic=actor)["total_unique"], 9)
        with self.assertRaises(TypeError):
            parameter_inventory(actor=actor, critic=None)


class InputSchemaTests(unittest.TestCase):
    def test_invalid_schema_identifiers_and_ordering_fail(self):
        for changes in ({"definition_id": ""}, {"node_ids": ("A", "A")},
                        {"node_feature_names": ()}, {"global_feature_names": ("",)},
                        {"action_names": ["A"]}, {"action_names": (1,)}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                schema(**changes)


if __name__ == "__main__":
    unittest.main()
