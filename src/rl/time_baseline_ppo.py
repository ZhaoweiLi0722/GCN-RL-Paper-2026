"""Versioned PPO targets over the frozen two-owner transaction machinery.

Only target selection differs between the two methods. Raw segment receipts,
critic targets, likelihoods, optimizers and failure accounting are unchanged.
No environment, saved research model or scientific launcher is invoked here.
"""

import copy
from dataclasses import asdict

import numpy as np

from src.rl.candidate_ppo_kernel import _record_id
from src.rl.candidate_ppo_objective import candidate_ppo_loss, normalize_rollout_advantages
from src.rl.candidate_rollout import _compute_gae
from src.rl.dynamic_candidate_ppo import DynamicCandidatePPOKernel, checked_gradient
from src.rl.dynamic_candidate_rollout import reevaluate_dynamic_decision
from src.rl.leave_one_episode_out_baseline import leave_one_episode_out_targets
from src.rl.networks import torch
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.training_state import training_contract_sha256


METHODS = ("collected_value", "leave_one_episode_out_time")


class TimeBaselinePPOKernel(DynamicCandidatePPOKernel):
    def __init__(self, prototype, contract, settings, *, enabled=False, mode,
                 sampling_seed, shuffle_seed, target_method, training_manifest,
                 training_source_id, episode_horizon, episodes_per_rollout):
        if target_method not in METHODS or not isinstance(training_source_id, str) or not training_source_id:
            raise ValueError("explicit target method and training source required")
        if (type(episode_horizon) is not int or episode_horizon < 1
                or type(episodes_per_rollout) is not int or episodes_per_rollout < 2
                or settings.max_rollout_steps != episode_horizon * episodes_per_rollout
                or contract.replay.gamma != 1. or settings.gae_lambda != 1.):
            raise ValueError("complete undiscounted fixed episode batches required")
        manifest = copy.deepcopy(tuple(training_manifest))
        if len(manifest) != settings.max_updates * episodes_per_rollout:
            raise ValueError("exact ordered training episode manifest required")
        for row in manifest:
            if (not isinstance(row, dict) or set(row) != {"trajectory_id", "environment_seed"}
                    or not isinstance(row["trajectory_id"], str) or not row["trajectory_id"]
                    or type(row["environment_seed"]) is not int or row["environment_seed"] < 0):
                raise ValueError("invalid training episode provenance")
        if (len({r["trajectory_id"] for r in manifest}) != len(manifest)
                or len({r["environment_seed"] for r in manifest}) != len(manifest)):
            raise ValueError("duplicate training trajectory or environment seed")
        super().__init__(prototype, contract, settings, enabled=enabled, mode=mode,
                         sampling_seed=sampling_seed, shuffle_seed=shuffle_seed)
        self.target_method, self.training_manifest = target_method, manifest
        self.training_source_id = training_source_id
        self.episode_horizon, self.episodes_per_rollout = episode_horizon, episodes_per_rollout
        self.target_history = []
        self._manifest.update(format="time-baseline-ppo-kernel-v1", target_method=target_method,
                              training_manifest=manifest, training_source_id=training_source_id,
                              episode_horizon=episode_horizon, episodes_per_rollout=episodes_per_rollout)
        self.manifest_sha256 = training_contract_sha256(self._manifest)

    def _validate_pending(self, segments):
        canonical = super()._validate_pending(segments)
        if len(canonical) > self.episodes_per_rollout:
            raise ValueError("too many complete episodes in rollout")
        offset = self.total_updates * self.episodes_per_rollout
        if offset + len(canonical) > len(self.training_manifest):
            raise ValueError("pending episodes exceed the declared training manifest")
        for i, segment in enumerate(canonical):
            expected = self.training_manifest[offset + i]
            if (len(segment.records) != self.episode_horizon or segment.bootstrap is not None
                    or not segment.records[-1].terminated
                    or any(r.truncated for r in segment.records)
                    or any(r.terminated for r in segment.records[:-1])
                    or [r.step_index for r in segment.records] != list(range(self.episode_horizon))
                    or any(r.trajectory_id != expected["trajectory_id"]
                           or r.source_id != self.training_source_id for r in segment.records)):
                raise ValueError("complete ordered training-only episode required")
        return canonical

    def target_receipt(self):
        """Build a detached receipt without replacing the canonical raw segments."""
        segments = self._validate_pending(self.pending)
        if len(segments) != self.episodes_per_rollout:
            raise ValueError("one full fixed rollout required for targets")
        offset = self.total_updates * self.episodes_per_rollout
        provenance = self.training_manifest[offset:offset + self.episodes_per_rollout]
        rows = tuple(tuple(s.returns) for s in segments)
        values = tuple(tuple(d.evaluation.value for d in s.decisions) for s in segments)
        common = dict(
            trajectory_ids=tuple(r["trajectory_id"] for r in provenance),
            environment_seeds=tuple(r["environment_seed"] for r in provenance),
            behavior_sha256s=tuple(s.decisions[0].evaluation.behavior_sha256 for s in segments),
            terminal_flags=(True,) * len(segments))
        loeo = leave_one_episode_out_targets(rows, **common)
        if self.target_method == "collected_value":
            baselines, advantages = values, tuple(tuple(s.advantages) for s in segments)
        else:
            baselines, advantages = loeo.baselines, loeo.advantages
        return dict(common, method=self.target_method, returns=rows, collected_values=values,
                    baselines=baselines, advantages=advantages,
                    raw_rewards=tuple(tuple(r.raw_reward for r in s.records) for s in segments),
                    segment_sha256s=tuple(state_digest(asdict(s)) for s in segments))

    def _update_in_place(self, before_optimizer_step, before_minibatch, before_compute):
        # Keep the frozen two-owner update order; change only detached advantages.
        before_compute()
        receipt = self.target_receipt()
        decisions = [d for s in self.pending for d in s.decisions]
        if self.total_optimizer_steps + self._required(self.pending) > self.settings.max_optimizer_steps:
            raise ValueError("two-owner optimizer cap exceeded")
        before_compute()
        advantages = normalize_rollout_advantages(torch.tensor(
            [a for row in receipt["advantages"] for a in row], dtype=self.dtype),
            enabled=self.settings.normalize_advantages)
        returns = torch.tensor([r for row in receipt["returns"] for r in row], dtype=self.dtype)
        old_logp = torch.tensor([d.old_log_prob for d in decisions], dtype=self.dtype)
        observations = []
        for decision in decisions:
            before_compute()
            observations.append(self._observation(decision.evaluation))
        logs = []
        for epoch in range(self.settings.epochs):
            before_compute()
            order = torch.randperm(len(decisions), generator=self.shuffle_rng).tolist()
            for start in range(0, len(order), self.settings.batch_size):
                indices = order[start:start + self.settings.batch_size]
                before_minibatch()
                evaluated = []
                for i in indices:
                    before_compute()
                    evaluated.append(reevaluate_dynamic_decision(self.policy, observations[i], decisions[i]))
                before_compute()
                logp, values, entropies = [torch.stack(v) for v in zip(*evaluated)]
                before_compute()
                losses = candidate_ppo_loss(
                    logp, values, entropies, old_logp[indices], advantages[indices], returns[indices],
                    clip_ratio=self.settings.clip_ratio, value_loss_coef=self.settings.value_loss_coef,
                    entropy_coef=self.settings.entropy_coef)
                self.policy.zero_grad(set_to_none=True)
                before_compute()
                (losses.policy - self.settings.entropy_coef * losses.entropy).backward()
                before_compute()
                actor_norm = checked_gradient(self.policy.actor_parameters(), self.policy.critic_parameters(),
                                              self.settings.max_grad_norm)
                before_compute()
                before_optimizer_step("actor")
                before_compute()
                self.actor_optimizer.step()
                before_compute()
                state_digest(self.policy.state_dict())
                state_digest(self.actor_optimizer.state_dict())
                self.policy.zero_grad(set_to_none=True)
                before_compute()
                (self.settings.value_loss_coef * losses.value).backward()
                before_compute()
                critic_norm = checked_gradient(self.policy.critic_parameters(), self.policy.actor_parameters(),
                                               self.settings.max_grad_norm)
                before_compute()
                before_optimizer_step("critic")
                before_compute()
                self.critic_optimizer.step()
                before_compute()
                state_digest(self.policy.state_dict())
                state_digest(self.critic_optimizer.state_dict())
                logs.append(dict(epoch=epoch, indices=indices, policy_loss=losses.policy.item(),
                                 value_loss=losses.value.item(), entropy=losses.entropy.item(),
                                 clip_fraction=losses.clip_fraction.item(),
                                 actor_grad_norm_before_clip=actor_norm,
                                 critic_grad_norm_before_clip=critic_norm))
        before_compute()
        self.history.append(len(decisions))
        self.target_history.append(receipt)
        self.consumed.extend(_record_id(r) for s in self.pending for r in s.records)
        self.pending = []
        self._set_modes()
        return dict(update=self.total_updates, optimizer_steps=self.total_optimizer_steps,
                    actor_optimizer_steps=self.total_owner_steps, critic_optimizer_steps=self.total_owner_steps,
                    rollout_steps=len(decisions), minibatches=logs, target_receipt=copy.deepcopy(receipt))

    def state_dict(self):
        return super().state_dict() | {"target_history": copy.deepcopy(self.target_history)}

    def _restore(self, state):
        if not isinstance(state, dict) or "target_history" not in state:
            raise ValueError("missing target history")
        super()._restore({k: v for k, v in state.items() if k != "target_history"})
        history = state["target_history"]
        if not isinstance(history, list) or len(history) != self.total_updates:
            raise ValueError("target history and completed updates differ")
        for index, receipt in enumerate(history):
            count = self.episodes_per_rollout
            provenance = self.training_manifest[index * count:(index + 1) * count]
            expected_keys = {"method", "trajectory_ids", "environment_seeds", "behavior_sha256s",
                             "terminal_flags", "returns", "collected_values", "baselines",
                             "advantages", "raw_rewards", "segment_sha256s"}
            if not isinstance(receipt, dict) or set(receipt) != expected_keys:
                raise ValueError("completed target receipt fields differ")
            if (receipt["method"] != self.target_method
                    or tuple(receipt["trajectory_ids"]) != tuple(r["trajectory_id"] for r in provenance)
                    or tuple(receipt["environment_seeds"]) != tuple(r["environment_seed"] for r in provenance)
                    or self.history[index] != count * self.episode_horizon):
                raise ValueError("completed target provenance mismatch")
            result = leave_one_episode_out_targets(receipt["returns"], **{
                k: receipt[k] for k in ("trajectory_ids", "environment_seeds", "behavior_sha256s", "terminal_flags")})
            if any(len(row) != self.episode_horizon for row in result.returns):
                raise ValueError("completed target horizon differs")
            for field in ("collected_values", "raw_rewards", "baselines", "advantages"):
                matrix = receipt[field]
                if (len(matrix) != count or any(len(row) != self.episode_horizon for row in matrix)
                        or not np.isfinite(np.asarray(matrix, dtype=np.float64)).all()):
                    raise ValueError("completed target matrix shape or values differ")
            shas = receipt["segment_sha256s"]
            if (len(shas) != count or any(not isinstance(s, str) or len(s) != 64
                                         or any(c not in "0123456789abcdef" for c in s) for s in shas)):
                raise ValueError("completed segment SHA256 inventory differs")
            original_advantages = []
            for rewards, values, returns in zip(receipt["raw_rewards"], receipt["collected_values"], result.returns):
                scaled = np.asarray([r * self.contract.replay.reward_scale for r in rewards], dtype=np.float32)
                dones = np.zeros(self.episode_horizon, dtype=np.float32)
                dones[-1] = 1.
                adv, expected = _compute_gae(rewards=scaled, dones=dones,
                    values=np.asarray(values, dtype=np.float32), last_value=0., gamma=1., gae_lambda=1.)
                if tuple(float(v) for v in expected) != returns:
                    raise ValueError("completed critic targets differ from unchanged reward arithmetic")
                original_advantages.append(tuple(float(v) for v in adv))
            if self.target_method == "leave_one_episode_out_time":
                if result.baselines != receipt["baselines"] or result.advantages != receipt["advantages"]:
                    raise ValueError("completed time-baseline arithmetic differs")
            elif (receipt["baselines"] != receipt["collected_values"]
                  or receipt["advantages"] != tuple(original_advantages)):
                raise ValueError("collected baseline receipt differs")
            identities = [(self.training_source_id, r["trajectory_id"], t)
                          for r in provenance for t in range(self.episode_horizon)]
            size = count * self.episode_horizon
            if self.consumed[index * size:(index + 1) * size] != identities:
                raise ValueError("target history and consumed trajectory identities differ")
        self.target_history = copy.deepcopy(history)
