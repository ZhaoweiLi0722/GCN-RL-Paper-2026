"""Objective-bound PPO with unchanged raw prefix segments and separate tail evidence.

This unregistered adapter reuses the existing two-owner transaction. It cannot
admit a segment until the whole fixed economic tail is recorded. No campaign,
patient environment, checkpoint loading or optimizer call happens on import.
"""

import copy
from dataclasses import asdict

import numpy as np

from src.rl.candidate_rollout import _compute_gae
from src.rl.cohort_objective import cohort_reward_receipt
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel
from src.rl.training_state import training_contract_sha256


def _sha(value):
    return (isinstance(value, str) and len(value) == 64
            and all(c in "0123456789abcdef" for c in value))


class CohortPPOKernel(TimeBaselinePPOKernel):
    def __init__(self, prototype, contract, settings, *, objective, accounting_steps, **kwargs):
        if (objective not in ("window", "cohort") or type(accounting_steps) is not int
                or accounting_steps < 1 or "target_method" in kwargs):
            raise ValueError("explicit window/cohort objective and fixed tail required")
        super().__init__(prototype, contract, settings, target_method="collected_value", **kwargs)
        self.objective, self.accounting_steps = objective, accounting_steps
        self.pending_cohorts = []
        self._manifest.update(format="cohort-ppo-kernel-v1", objective=objective,
                              accounting_steps=accounting_steps,
                              raw_prefix_immutable=True, tail_has_learned_actions=False)
        self.manifest_sha256 = training_contract_sha256(self._manifest)

    def _closure(self, payload, raw_rewards, provenance):
        keys = {"trajectory_id", "environment_seed", "source_id", "split", "tail_costs",
                "terminal_active", "prefix_state_sha256", "final_state_sha256", "tail_rows_sha256"}
        if (not isinstance(payload, dict) or set(payload) != keys
                or payload["trajectory_id"] != provenance["trajectory_id"]
                or type(payload["environment_seed"]) is not int
                or payload["environment_seed"] != provenance["environment_seed"]
                or payload["source_id"] != self.training_source_id
                or payload["split"] != "training"
                or any(not _sha(payload[k]) for k in
                       ("prefix_state_sha256", "final_state_sha256", "tail_rows_sha256"))):
            raise ValueError("complete training-only tail lineage required")
        receipt = cohort_reward_receipt(tuple(-r for r in raw_rewards), payload["tail_costs"],
            objective=self.objective, prefix_steps=self.episode_horizon,
            accounting_steps=self.accounting_steps, active_at_end=payload["terminal_active"],
            trajectory_id=payload["trajectory_id"], split=payload["split"])
        return receipt

    def add_segment(self, segment, *, cohort=None):
        if self._failure is not None:
            raise ValueError("failed transaction cannot admit more data")
        proposed = self._validate_pending([*self.pending, segment])
        offset = self.total_updates * self.episodes_per_rollout + len(self.pending)
        self._closure(cohort, tuple(r.raw_reward for r in proposed[-1].records),
                      self.training_manifest[offset])
        copied = copy.deepcopy(cohort)
        copied["tail_costs"] = tuple(copied["tail_costs"])
        super().add_segment(segment)
        self.pending_cohorts.append(copied)

    def _receipt(self, raw_rewards, values, closures, behavior, segment_shas, index):
        count = self.episodes_per_rollout
        provenance = self.training_manifest[index * count:(index + 1) * count]
        if (len(provenance) != count or any(len(x) != count for x in
                (raw_rewards, values, closures, behavior, segment_shas))
                or any(not _sha(x) for x in (*behavior, *segment_shas))
                or len(set(behavior)) != 1):
            raise ValueError("one complete same-behavior rollout and hashes required")
        targets, advantages, returns = [], [], []
        for raw, baseline, closure, origin in zip(raw_rewards, values, closures, provenance):
            if (len(raw) != self.episode_horizon or len(baseline) != self.episode_horizon
                    or not np.isfinite(np.asarray(baseline, dtype=np.float64)).all()):
                raise ValueError("complete finite prefix/value rows required")
            target = self._closure(closure, raw, origin)
            scaled = np.asarray([r * self.contract.replay.reward_scale
                                 for r in target["training_raw_rewards"]], dtype=np.float32)
            if not np.isfinite(scaled).all():
                raise ValueError("scaled cohort rewards overflow")
            dones = np.zeros(self.episode_horizon, dtype=np.float32)
            dones[-1] = 1.
            adv, ret = _compute_gae(rewards=scaled, dones=dones,
                values=np.asarray(baseline, dtype=np.float32), last_value=0., gamma=1., gae_lambda=1.)
            if not np.isfinite(adv).all() or not np.isfinite(ret).all():
                raise ValueError("nonfinite cohort targets")
            targets.append(target)
            advantages.append(tuple(float(x) for x in adv))
            returns.append(tuple(float(x) for x in ret))
        return dict(method="collected_value", objective=self.objective,
                    trajectory_ids=tuple(r["trajectory_id"] for r in provenance),
                    environment_seeds=tuple(r["environment_seed"] for r in provenance),
                    behavior_sha256s=tuple(behavior), terminal_flags=(True,) * count,
                    raw_rewards=tuple(tuple(r) for r in raw_rewards),
                    collected_values=tuple(tuple(v) for v in values),
                    baselines=tuple(tuple(v) for v in values),
                    returns=tuple(returns), advantages=tuple(advantages),
                    segment_sha256s=tuple(segment_shas), cohort_receipts=tuple(targets),
                    closures=copy.deepcopy(tuple(closures)))

    def target_receipt(self):
        segments = self._validate_pending(self.pending)
        if len(segments) != self.episodes_per_rollout:
            raise ValueError("complete prefix-plus-tail batch required")
        return self._receipt(tuple(tuple(r.raw_reward for r in s.records) for s in segments),
            tuple(tuple(d.evaluation.value for d in s.decisions) for s in segments),
            self.pending_cohorts, tuple(s.decisions[0].evaluation.behavior_sha256 for s in segments),
            tuple(state_digest(asdict(s)) for s in segments), self.total_updates)

    def _update_in_place(self, before_optimizer_step, before_minibatch, before_compute):
        report = super()._update_in_place(before_optimizer_step, before_minibatch, before_compute)
        self.pending_cohorts = []
        return report

    def state_dict(self):
        return super().state_dict() | {"pending_cohorts": copy.deepcopy(self.pending_cohorts)}

    def _restore(self, state):
        if not isinstance(state, dict) or not {"target_history", "pending_cohorts"} <= set(state):
            raise ValueError("objective history and pending closure evidence required")
        # The old baseline adapter intentionally rejects changed critic targets.
        # Restore its unchanged transaction base, then verify this version's
        # explicit objective arithmetic instead of weakening the old reader.
        DynamicCandidatePPOKernel._restore(self, {k: v for k, v in state.items()
            if k not in ("target_history", "pending_cohorts")})
        closures, history = state["pending_cohorts"], state["target_history"]
        if (not isinstance(closures, list) or len(closures) != len(self.pending)
                or not isinstance(history, list) or len(history) != self.total_updates):
            raise ValueError("closure evidence and update progress differ")
        offset = self.total_updates * self.episodes_per_rollout
        for i, (segment, closure) in enumerate(zip(self.pending, closures)):
            self._closure(closure, tuple(r.raw_reward for r in segment.records),
                          self.training_manifest[offset + i])
        for index, receipt in enumerate(history):
            try:
                rebuilt = self._receipt(receipt["raw_rewards"], receipt["collected_values"],
                    receipt["closures"], receipt["behavior_sha256s"], receipt["segment_sha256s"], index)
            except (KeyError, TypeError) as error:
                raise ValueError("malformed completed cohort targets") from error
            if state_digest(receipt) != state_digest(rebuilt):
                raise ValueError("completed objective arithmetic or lineage differs")
            count, width = self.episodes_per_rollout, self.episode_horizon
            provenance = self.training_manifest[index * count:(index + 1) * count]
            identities = [(self.training_source_id, r["trajectory_id"], t)
                          for r in provenance for t in range(width)]
            if (self.history[index] != count * width
                    or self.consumed[index * count * width:(index + 1) * count * width] != identities):
                raise ValueError("objective history and consumed prefix identities differ")
        self.pending_cohorts, self.target_history = copy.deepcopy(closures), copy.deepcopy(history)
