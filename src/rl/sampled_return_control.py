"""Sampled artificial rewards and independent losses, not a patient runner.

Only the task generator knows the selected action's reward rule. The learner
receives observed scalar returns and public contexts, never a full Q table.
This module provides no experiment launch or automatic optimization loop.
"""

import copy
from dataclasses import dataclass
import math

from src.models.independent_artificial_value import IndependentArtificialValue
from src.rl.actor_positive_control import model, rng_snapshot
from src.rl.candidate_calibration_engineering import fixture_schema, invented_data, scores
from src.rl.candidate_ppo_objective import candidate_ppo_loss, normalize_rollout_advantages
from src.rl.networks import torch


CONFIG = "experiments/configs/candidate_sampled_return_control_20261001.json"
PROTOCOL = "specs/2026-10-01-sampled-return-control/protocol.md"


def make_models(config, bank, representation, seed):
    actor = model(config, bank, representation, seed)
    critic = IndependentArtificialValue(fixture_schema(), config["units"], output_gain=config["critic"]["output_gain"])
    if {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}:
        raise ValueError("actor and critic must not share parameter storage")
    return actor, critic


def public_fixture(config, times):
    observations, bank, _, _ = invented_data(config, times)
    return observations, bank


def observed_reward(config, observation, bank, action):
    """Task-side deterministic scalar observation, not a learner target oracle."""
    if type(action) is not int or not 0 <= action < len(bank.class_keys):
        raise ValueError("invalid sampled class")
    t, cue = observation.globals.detach()[0].tolist()
    winner = config["positive_cue_winner_request"] if cue > 0 else config["negative_cue_winner_request"]
    task = config["invented_return"]
    return (task["intercept"] + task["remaining_time_slope"]*(1-t)
            + (0. if bank.class_features[action][0] == winner else task["wrong_action_penalty"]))


@dataclass(frozen=True)
class SampledReturns:
    context_indices: object
    actions: object
    old_log_probs: object
    old_values: object
    returns: object
    advantages: object
    behavior_logits: object

    def validate(self, context_count, class_count):
        n = self.actions.numel()
        if n < 1 or self.actions.dtype != torch.int64 or self.context_indices.dtype != torch.int64:
            raise ValueError("nonempty integer context/action vectors required")
        for value in (self.actions, self.context_indices):
            if value.shape != (n,) or value.device.type != "cpu":
                raise ValueError("CPU vectors required")
        if ((self.context_indices < 0) | (self.context_indices >= context_count)).any() or ((self.actions < 0) | (self.actions >= class_count)).any():
            raise ValueError("context/action outside support")
        for value in (self.old_log_probs, self.old_values, self.returns, self.advantages):
            if value.shape != (n,) or value.dtype != torch.float32 or value.device.type != "cpu" or value.requires_grad or not torch.isfinite(value).all():
                raise ValueError("detached finite CPU float32 sample vectors required")
        logits = self.behavior_logits
        if logits.shape != (n, class_count) or logits.dtype != torch.float32 or logits.device.type != "cpu" or logits.requires_grad or not torch.isfinite(logits).all():
            raise ValueError("finite detached behavior logits required")
        selected = logits.log_softmax(1).gather(1, self.actions[:, None]).flatten()
        if not torch.allclose(selected, self.old_log_probs, atol=1e-6, rtol=0):
            raise ValueError("selected behavior probability differs from receipt")
        expected = normalize_rollout_advantages(self.returns-self.old_values, enabled=True)
        if not torch.allclose(expected, self.advantages, atol=1e-6, rtol=0):
            raise ValueError("advantages must use observed returns and pre-update values")


class ArtificialBudget:
    """Charge before work; failures never refund calls or observations."""
    def __init__(self, config):
        self.config = copy.deepcopy(config)
        self.observations, self.optimizer_calls, self.by_fit = 0, 0, {}

    def charge(self, name, kind, amount, persist):
        if kind not in ("observations", "actor", "critic") or type(amount) is not int or amount < 1:
            raise ValueError("explicit positive charge required")
        if name not in {f"{r}-{s}" for r in self.config["representations"] for s in self.config["initialization_seeds"]}:
            raise ValueError("unknown fixture")
        if kind != "observations" and amount != 1:
            raise ValueError("each optimizer call must be charged separately")
        counts = self.by_fit.get(name, {"observations": 0, "actor": 0, "critic": 0})
        per_cap = 1536 if kind == "observations" else 128
        total = self.observations if kind == "observations" else self.optimizer_calls
        cap = 13824 if kind == "observations" else 2304
        if counts[kind]+amount > per_cap or total+amount > cap:
            raise RuntimeError("artificial packet budget exhausted; no refund")
        counts = dict(counts)
        counts[kind] += amount
        self.by_fit[name] = counts
        if kind == "observations":
            self.observations += amount
        else:
            self.optimizer_calls += amount
        receipt = {"fixture": name, "kind": kind, "amount": amount, "budget": self.state()}
        persist(receipt)
        return receipt

    def state(self):
        return copy.deepcopy({"observations": self.observations, "optimizer_calls": self.optimizer_calls, "by_fit": self.by_fit})


def collect_samples(actor, critic, observations, bank, *, sampling_rng, repetitions,
                    reward_observer, charge_observations):
    if type(repetitions) is not int or repetitions < 1 or not observations or sampling_rng.device.type != "cpu":
        raise ValueError("explicit public contexts, CPU generator and repetitions required")
    indices = torch.arange(len(observations)).repeat(repetitions)
    charge_observations(indices.numel())
    with torch.no_grad():
        logits, _ = scores(actor, observations, bank)
        values = torch.stack([critic(obs, bank) for obs in observations])
        behavior = logits[indices]
        actions = torch.multinomial(behavior.softmax(1), 1, generator=sampling_rng).flatten()
        returns = torch.tensor([reward_observer(observations[i], bank, int(a)) for i, a in zip(indices.tolist(), actions)], dtype=torch.float32)
        old_values = values[indices]
        logp = behavior.log_softmax(1).gather(1, actions[:, None]).flatten()
        batch = SampledReturns(indices, actions, logp, old_values, returns,
            normalize_rollout_advantages(returns-old_values, enabled=True), behavior)
        batch.validate(len(observations), len(bank.class_keys))
        return batch


def sampled_losses(actor, critic, observations, bank, batch, selected, settings):
    batch.validate(len(observations), len(bank.class_keys))
    if selected.dtype != torch.int64 or selected.ndim != 1 or selected.numel() < 1 or selected.device.type != "cpu" or (selected < 0).any() or (selected >= batch.actions.numel()).any():
        raise ValueError("valid explicit minibatch indices required")
    if torch.unique(selected).numel() != selected.numel():
        raise ValueError("no duplicate minibatch indices")
    contexts = [observations[i] for i in batch.context_indices[selected].tolist()]
    logits, _ = scores(actor, contexts, bank)
    logp = logits.log_softmax(1)
    selected_logp = logp.gather(1, batch.actions[selected, None]).flatten()
    entropy = -(logp.exp()*logp).sum(1)
    zeros = torch.zeros_like(selected_logp)
    actor_loss = candidate_ppo_loss(selected_logp, zeros, entropy, batch.old_log_probs[selected],
        batch.advantages[selected], zeros, clip_ratio=settings["clip_ratio"], value_loss_coef=0.,
        entropy_coef=settings["entropy_coef"])
    values = torch.stack([critic(obs, bank) for obs in contexts])
    value_loss = torch.nn.functional.mse_loss(values, batch.returns[selected])
    if not torch.isfinite(value_loss):
        raise ValueError("nonfinite critic loss")
    return actor_loss, value_loss


def update_pair(actor, critic, actor_optimizer, critic_optimizer, observations, bank,
                batch, selected, settings, *, charge_optimizer):
    """Rollback weights/optimizers on error; caller's durable budget stays spent."""
    models, optimizers = (actor, critic), (actor_optimizer, critic_optimizer)
    parameters = [{id(p) for p in m.parameters()} for m in models]
    for expected, optimizer in zip(parameters, optimizers):
        actual = [id(p) for group in optimizer.param_groups for p in group["params"]]
        if set(actual) != expected or len(actual) != len(expected):
            raise ValueError("optimizer parameters must exactly match its model")
    if parameters[0] & parameters[1] or {p.data_ptr() for p in actor.parameters()} & {p.data_ptr() for p in critic.parameters()}:
        raise ValueError("actor/critic must be independent")
    before = [copy.deepcopy(x.state_dict()) for x in (*models, *optimizers)]
    try:
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        loss, value_loss = sampled_losses(actor, critic, observations, bank, batch, selected, settings)
        loss.total.backward()
        value_loss.backward()
        norms = [float(torch.nn.utils.clip_grad_norm_(m.parameters(), settings["max_grad_norm"])) for m in models]
        if not all(math.isfinite(v) for v in norms):
            raise ValueError("nonfinite gradient")
        for kind, optimizer in zip(("actor", "critic"), optimizers):
            charge_optimizer(kind)
            optimizer.step()
        if any(not torch.isfinite(p).all() for m in models for p in m.parameters()):
            raise ValueError("nonfinite updated parameter")
        return {"policy_loss": float(loss.policy.detach()), "entropy": float(loss.entropy.detach()),
                "actor_total": float(loss.total.detach()), "critic_mse": float(value_loss.detach()),
                "clip_fraction": float(loss.clip_fraction), "actor_grad_norm": norms[0], "critic_grad_norm": norms[1]}
    except BaseException:
        for obj, state in zip((*models, *optimizers), before):
            obj.load_state_dict(state)
        for opt in optimizers:
            opt.zero_grad(set_to_none=True)
        raise


def snapshot(actor, critic, optimizers, generators, *, batch, permutation, next_minibatch, budget):
    return copy.deepcopy({"actor": actor.state_dict(), "critic": critic.state_dict(),
        "actor_optimizer": optimizers[0].state_dict(), "critic_optimizer": optimizers[1].state_dict(),
        "actor_manifest": actor.manifest(), "sampling_rng": generators[0].get_state(),
        "shuffle_rng": generators[1].get_state(), "global_rng": rng_snapshot(),
        "batch": None if batch is None else vars(batch), "permutation": permutation,
        "next_minibatch": next_minibatch, "budget": budget.state(), "resume_authorized": False})
