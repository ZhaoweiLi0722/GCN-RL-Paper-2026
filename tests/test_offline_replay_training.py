from __future__ import annotations

import unittest

from src.rl.experiment import train_offline_replay_updates


class _FakeAgent:
    def __init__(self) -> None:
        self.calls = 0

    def update(self):
        self.calls += 1
        return {
            "actor_loss": float(self.calls),
            "critic_loss": float(2 * self.calls),
        }


class _DelayedActorAgent:
    def __init__(self) -> None:
        self.calls = 0

    def update(self):
        self.calls += 1
        metrics = {"critic1_loss": float(self.calls)}
        if self.calls % 2 == 0:
            metrics["actor_loss"] = -float(self.calls)
        return metrics


class OfflineReplayTrainingTests(unittest.TestCase):
    def test_aggregates_offline_update_metrics(self) -> None:
        agent = _FakeAgent()

        summary = train_offline_replay_updates(agent, updates=3)

        self.assertEqual(agent.calls, 3)
        self.assertEqual(summary["offline_rl_updates"], 3)
        self.assertEqual(summary["offline_rl_actor_loss_mean"], 2.0)
        self.assertEqual(summary["offline_rl_actor_loss_final"], 3.0)
        self.assertEqual(summary["offline_rl_critic_loss_mean"], 4.0)
        self.assertEqual(summary["offline_rl_critic_loss_final"], 6.0)

    def test_zero_updates_is_a_noop(self) -> None:
        agent = _FakeAgent()

        summary = train_offline_replay_updates(agent, updates=0)

        self.assertEqual(summary, {"offline_rl_updates": 0})
        self.assertEqual(agent.calls, 0)

    def test_keeps_latest_metric_when_delayed_update_omits_it(self) -> None:
        summary = train_offline_replay_updates(
            _DelayedActorAgent(),
            updates=3,
        )

        self.assertEqual(summary["offline_rl_actor_loss_mean"], -2.0)
        self.assertEqual(summary["offline_rl_actor_loss_final"], -2.0)
        self.assertEqual(summary["offline_rl_critic1_loss_final"], 3.0)

    def test_rejects_empty_update_metrics(self) -> None:
        class EmptyAgent:
            def update(self):
                return {}

        with self.assertRaisesRegex(RuntimeError, "produced no metrics"):
            train_offline_replay_updates(EmptyAgent(), updates=1)

    def test_rejects_negative_update_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be non-negative"):
            train_offline_replay_updates(_FakeAgent(), updates=-1)


if __name__ == "__main__":
    unittest.main()
