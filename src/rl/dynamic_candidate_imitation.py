"""Actor-only initialization/continuation for an independent-critic candidate.

The existing demonstration/rollout split contract is retained. Critic weights
and optimizer ownership are independent; no historical kernel is registered or
modified. Scientific fitting still requires a separately authorized runner.
"""

import copy
from dataclasses import asdict
import math

from src.models.dynamic_candidate_policy import DynamicCandidatePolicy
from src.rl.candidate_imitation import ImitationSettings, decode_example
from src.rl.candidate_rollout import sample_candidate
from src.rl.dynamic_candidate_ppo import validate_adam_state
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.networks import require_torch, torch
from src.rl.prospective_adapter import ReplayInputContract
from src.rl.prospective_ddpg_kernel import state_digest


class DynamicCandidateImitationKernel:
    def __init__(self, prototype, contract, settings, *, enabled=False, shuffle_seed,
                 sampling_seed=None):
        require_torch()
        if (enabled is not True or type(prototype) is not DynamicCandidatePolicy
                or not isinstance(settings, ImitationSettings)
                or not isinstance(contract, ReplayInputContract) or prototype.schema != contract.inputs):
            raise ValueError("explicit matching dynamic imitation contract required")
        for seed in (shuffle_seed,):
            if type(seed) is not int or not 0 <= seed < 2**63:
                raise ValueError("explicit nonnegative private seed required")
        if sampling_seed is not None and (type(sampling_seed) is not int or not 0 <= sampling_seed < 2**63):
            raise ValueError("invalid private sampling seed")
        if any(p.dtype != torch.float32 or p.device.type != "cpu" for p in prototype.parameters()):
            raise ValueError("uniform CPU float32 required for imitation examples")
        actor = tuple(prototype.actor_parameters())
        critic = tuple(prototype.critic_parameters())
        if (not actor or not critic
                or {p.untyped_storage().data_ptr() for p in actor}.intersection(
                    p.untyped_storage().data_ptr() for p in critic)
                or {id(p) for p in (*actor, *critic)} != {id(p) for p in prototype.parameters()}):
            raise ValueError("disjoint complete parameter ownership required")
        self.policy, self.contract, self.settings = copy.deepcopy(prototype), contract, settings
        self.manifest = {"format": "dynamic-candidate-imitation-v1", "contract": asdict(contract),
                         "settings": asdict(settings), "policy": prototype.manifest(),
                         "initial_weights": prototype.snapshot_sha256(), "shuffle_seed": shuffle_seed,
                         "sampling_seed": sampling_seed, "trainable_owner": "actor"}
        self._critic_state = copy.deepcopy(self.policy.critic.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.actor_parameters(),
                                          lr=settings.learning_rate, foreach=False)
        self._groups = copy.deepcopy(self.optimizer.state_dict()["param_groups"])
        self.rng = torch.Generator(device="cpu").manual_seed(shuffle_seed)
        self.sampling_rng = (None if sampling_seed is None else
                             torch.Generator(device="cpu").manual_seed(sampling_seed))
        self.steps, self.history, self._failure = 0, [], None
        self._set_modes()

    def _set_modes(self):
        self.policy.eval()
        self.policy.actor.requires_grad_(True)
        self.policy.critic.requires_grad_(False)
        self.policy.zero_grad(set_to_none=True)

    def trainable(self):
        return tuple(self.policy.actor_parameters())

    def decide(self, observation, candidates):
        if self._failure is not None or self.sampling_rng is None:
            raise ValueError("failed/initialization kernel cannot collect")
        return sample_candidate(evaluate_dynamic_policy(self.policy, observation, candidates, self.contract),
                                generator=self.sampling_rng)

    def _fit(self, examples, *, epochs, replacement_steps, before_step):
        if self.settings.allowed_split == "demonstration":
            if epochs is not None or type(replacement_steps) is not int or replacement_steps < 1:
                raise ValueError("demonstration requires explicit replacement-update count")
            count = replacement_steps
        else:
            if replacement_steps is not None or type(epochs) is not int or epochs < 1:
                raise ValueError("continuation requires explicit whole epochs")
            count = epochs * math.ceil(len(examples) / self.settings.batch_size)
        if (not examples or self.steps + count > self.settings.max_optimizer_steps
                or len(self.history) >= self.settings.max_updates):
            raise ValueError("empty data or imitation budget exceeded")
        ids = [e["identity"] for e in examples]
        if len(set(ids)) != len(ids) or any(e["split"] != self.settings.allowed_split for e in examples):
            raise ValueError("mixed/forbidden split or duplicate examples")
        if set(ids).intersection(i for row in self.history for i in row["identities"]):
            raise ValueError("already consumed imitation rollout")
        decoded = [decode_example(e, self.contract) for e in examples]
        if replacement_steps is not None:
            batches = [torch.randint(len(examples), (self.settings.batch_size,), generator=self.rng).tolist()
                       for _ in range(replacement_steps)]
        else:
            batches = []
            for _ in range(epochs):
                order = torch.randperm(len(examples), generator=self.rng).tolist()
                batches.extend(order[i:i + self.settings.batch_size] for i in range(0, len(order), self.settings.batch_size))
        logs = []
        for indices in batches:
            losses = []
            for i in indices:
                observation, candidates = decoded[i]
                logits = self.policy(observation, candidates).logits
                losses.append(-torch.log_softmax(logits, dim=0)[candidates.reference_class])
            loss = torch.stack(losses).mean()
            if not torch.isfinite(loss).item():
                raise ValueError("nonfinite imitation loss")
            self.policy.zero_grad(set_to_none=True)
            loss.backward()
            parameters = self.trainable()
            if (any(p.grad is None or not torch.isfinite(p.grad).all().item() for p in parameters)
                    or any(p.grad is not None for p in self.policy.critic_parameters())):
                raise ValueError("nonfinite/missing/cross-owner imitation gradient")
            norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings.max_grad_norm, error_if_nonfinite=True)
            before_step()
            self.optimizer.step()
            self.steps += 1
            state_digest(self.policy.state_dict())
            state_digest(self.optimizer.state_dict())
            logs.append({"indices": indices, "cross_entropy": loss.item(), "grad_norm": norm.item()})
        if state_digest(self.policy.critic.state_dict()) != state_digest(self._critic_state):
            raise ValueError("imitation changed independent critic")
        self.history.append({"identities": ids, "examples_sha256": state_digest(examples), "steps": count})
        self._set_modes()
        return {"optimizer_steps": self.steps, "minibatches": logs}

    def fit(self, examples, *, before_step, epochs=None, replacement_steps=None):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal")
        if not callable(before_step):
            raise ValueError("explicit durable charge callback required")
        candidate, charged = copy.deepcopy(self), []

        def debit():
            before_step()
            charged.append("actor")

        try:
            candidate._restore(self.state_dict())
            result = candidate._fit(examples, epochs=epochs, replacement_steps=replacement_steps, before_step=debit)
            candidate._restore(candidate.state_dict())
        except BaseException as error:
            self._failure = {"error_type": type(error).__name__, "message": str(error),
                             "acknowledged_charge_owners": charged,
                             "resource_authority": "external_non_refundable_ledger",
                             "partial_transaction": candidate.state_dict()}
            raise
        self.__dict__.update(candidate.__dict__)
        return result

    def state_dict(self):
        return copy.deepcopy({"manifest": self.manifest, "policy": self.policy.state_dict(),
                              "optimizer": self.optimizer.state_dict(), "rng": self.rng.get_state(),
                              "sampling_rng": None if self.sampling_rng is None else self.sampling_rng.get_state(),
                              "steps": self.steps, "history": self.history, "failure": self._failure})

    def _restore(self, state):
        keys = {"manifest", "policy", "optimizer", "rng", "sampling_rng", "steps", "history", "failure"}
        if not isinstance(state, dict) or set(state) != keys or state["manifest"] != self.manifest:
            raise ValueError("imitation checkpoint manifest differs")
        if state["failure"] is not None:
            raise ValueError("failed transaction evidence cannot resume")
        state_digest(state)
        steps, history = state["steps"], state["history"]
        if (type(steps) is not int or not 0 <= steps <= self.settings.max_optimizer_steps
                or not isinstance(history, list) or len(history) > self.settings.max_updates
                or any(not isinstance(h, dict) or set(h) != {"steps", "identities", "examples_sha256"}
                       or type(h["steps"]) is not int or h["steps"] < 1
                       or not isinstance(h["identities"], list) or not h["identities"]
                       or any(not isinstance(i, str) or not i for i in h["identities"])
                       or not isinstance(h["examples_sha256"], str) or len(h["examples_sha256"]) != 64
                       or any(c not in "0123456789abcdef" for c in h["examples_sha256"]) for h in history)
                or sum(h["steps"] for h in history) != steps):
            raise ValueError("invalid imitation counters")
        ids = [i for h in history for i in h["identities"]]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate consumed imitation lineage")
        expected = self.policy.state_dict()
        if set(state["policy"]) != set(expected):
            raise ValueError("policy keys differ")
        for name, value in state["policy"].items():
            if (not isinstance(value, torch.Tensor) or value.dtype != expected[name].dtype
                    or value.shape != expected[name].shape or value.device.type != "cpu"):
                raise ValueError("policy tensor contract mismatch")
        self.policy.load_state_dict(state["policy"])
        if state_digest(self.policy.critic.state_dict()) != state_digest(self._critic_state):
            raise ValueError("frozen critic differs")
        if not steps and self.policy.snapshot_sha256() != self.manifest["initial_weights"]:
            raise ValueError("zero-step weights differ")
        validate_adam_state(self.optimizer, state["optimizer"], self.trainable(), self._groups, steps)
        for key, generator in (("rng", self.rng), ("sampling_rng", self.sampling_rng)):
            value = state[key]
            if generator is None:
                if value is not None:
                    raise ValueError("unexpected private sampling RNG")
                continue
            if (not isinstance(value, torch.Tensor) or value.dtype != torch.uint8
                    or value.device.type != "cpu" or value.shape != generator.get_state().shape):
                raise ValueError("invalid private RNG")
            generator.set_state(value)
        self.steps, self.history, self._failure = steps, copy.deepcopy(history), None
        self._set_modes()

    def load_state_dict(self, state):
        if self._failure is not None:
            raise ValueError("failed transaction is terminal; cannot restore to retry")
        candidate = copy.deepcopy(self)
        candidate._restore(copy.deepcopy(state))
        self.__dict__.update(candidate.__dict__)
