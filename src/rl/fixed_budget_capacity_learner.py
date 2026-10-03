"""Fixed-total logit actor adapter; the dual-critic training schedule is unchanged.

No environment or checkpoint-file access occurs here. Snapshots are versioned
separately because identical actor tensor shapes do not imply identical actions.
"""

from __future__ import annotations

import copy

import numpy as np

from src.models.capacity_ddpg import CapacityModelConfig, validate_features, validate_hours
from src.models.fixed_budget_capacity import FixedBudgetCapacityActor
from src.rl.patient_constrained_learner import PatientConstrainedConfig, PatientConstrainedLearner
from src.rl.networks import torch


class FixedBudgetCapacityLearner(PatientConstrainedLearner):
    format = "fixed-budget-capacity-learner-v1"

    def __init__(self, config, **kwargs):
        if config.exploration_sigma_hours != 0.25:
            raise ValueError("fixed-budget behavior requires configured logit sigma0.25")
        super().__init__(config, **kwargs)
        weights = self.actor.state_dict()
        # Replacement initialization is discarded; neither global nor saved RNG
        # streams advance, and the parent's exact zero-head start is retained.
        with torch.random.fork_rng(devices=[]):
            torch.set_rng_state(self.torch_rng_state)
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                actor = FixedBudgetCapacityActor(config.model, adjacency=kwargs.get("adjacency"))
            finally:
                torch.set_default_dtype(previous_dtype)
        actor.load_state_dict(weights, strict=True)
        self.actor = actor
        self.target_actor = copy.deepcopy(actor)
        self.target_actor.requires_grad_(False)
        self.actor_optimizer = torch.optim.Adam(actor.parameters(), lr=config.actor_lr,
            betas=config.betas, eps=config.epsilon, weight_decay=config.weight_decay, foreach=False)

    def act(self, features, *, explore=False, noise_hours=None):
        """One charged forward, returning (physical_request, physical_executed).

        ``noise_hours`` keeps the parent's call signature but now denotes logit
        innovations, added BEFORE the bounded map, never physical-hour noise.
        Supplied noise consumes no RNG; otherwise explore draws iid sigma0.25.

        NumPy float32 [sites, F] features yield two independent float64 [sites]
        arrays with native budget closure. CPU float32 batched tensor features
        yield two float32 tensors using the differentiable critic map. Neither
        return value is a logit or a legacy clipped/radially scaled request.
        """
        def act():
            numpy_input = isinstance(features, np.ndarray)
            if numpy_input:
                if features.dtype != np.float32 or features.shape != (self.config.model.num_nodes, self.config.model.node_input_dim):
                    raise ValueError("np.float32 [sites, node_input_dim] public features required")
                inputs = torch.from_numpy(features.copy()).unsqueeze(0)
            else:
                inputs = features
            validate_features(inputs, self.config.model)
            with torch.no_grad():
                logits = self._forward("behavior_actor", self.actor, inputs)
                if noise_hours is not None and not explore:
                    raise ValueError("logit noise requires explore=True")
                noise = None
                if explore:
                    noise = (self.exploration_rng.normal(0, self.config.exploration_sigma_hours, tuple(logits.shape))
                             if noise_hours is None else noise_hours)
                    noise = torch.as_tensor(noise, dtype=torch.float32, device="cpu")
                    if numpy_input and noise.ndim == 1:
                        noise = noise.unsqueeze(0)
                    validate_hours(noise, self.config.model, batch_size=logits.shape[0], legal=False)
                if numpy_input:
                    native = self.actor.project_native(logits[0], noise=None if noise is None else noise[0])
                    hours = np.asarray(native, dtype=np.float64)
                    return hours.copy(), hours.copy()
                hours = self.actor.project(logits) if noise is None else self.actor.project_noisy(logits, noise)
                return hours.clone(), hours
        return self._run(act)

    def state_dict(self):
        if (type(self.actor) is not FixedBudgetCapacityActor or type(self.target_actor) is not FixedBudgetCapacityActor
                or self.actor.metadata() != self.target_actor.metadata()):
            raise ValueError("fixed-budget online/target actor semantics required")
        state = super().state_dict()
        state["format"] = self.format
        state["actor_metadata"] = self.actor.metadata()
        return state

    def load_state_dict(self, state):
        """Validate semantics, then retain the parent's atomic full-state checks."""
        self._guard()
        if (not isinstance(state, dict) or state.get("format") != self.format
                or state.get("actor_metadata") != self.actor.metadata()
                or self.target_actor.metadata() != self.actor.metadata()):
            raise ValueError("fixed-budget snapshot format/actor metadata mismatch; legacy semantics rejected")
        base = dict(state)
        del base["actor_metadata"]
        base["format"] = "patient-constrained-learner-v1"
        # The parent checks keys against self.state_dict(). A detached parent-
        # typed view preserves that exact check without flags on the live object.
        view = object.__new__(PatientConstrainedLearner)
        view.__dict__.update(copy.deepcopy(self.__dict__, {
            id(self.before_forward): self.before_forward,
            id(self.before_optimizer): self.before_optimizer,
        }))
        PatientConstrainedLearner.load_state_dict(view, base)
        self.__dict__.update(view.__dict__)

    @classmethod
    def from_snapshot(cls, snapshot, *, before_forward, before_optimizer):
        state = snapshot.state_dict()
        if not isinstance(state, dict) or state.get("format") != cls.format or "actor_metadata" not in state:
            raise ValueError("fixed-budget snapshot required; no legacy actor migration")
        config = dict(state["config"])
        config["model"] = CapacityModelConfig(**config["model"])
        learner = cls(PatientConstrainedConfig(**config), model_seed=0, replay_seed=0,
            adjacency=state["modules"]["actor"]["encoder.adjacency"],
            before_forward=before_forward, before_optimizer=before_optimizer)
        learner.load_state_dict(state)
        learner._fresh_fork = learner.arm is None and learner.completed["warmup"] == learner.config.warmup_batches
        return learner
