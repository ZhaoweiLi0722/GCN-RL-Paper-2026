"""Scratch observed TD8 value-MPC arm with matched graph/self-only states."""

import copy
import numpy as np

from src.models.capacity_confirmation_value import CapacityConfirmationValue
from src.rl.capacity_value_learner import CapacityValueLearner
from src.rl.networks import torch


class CapacityFamilyValue(CapacityValueLearner):
    format = "capacity-family-observed-td-v1"

    def __init__(self, config, *, seed, feature_dim=31, before_forward, before_optimizer):
        if feature_dim != 31 or config["architecture"] not in ("graph", "self_only"):
            raise ValueError("matched 31-feature architecture required")
        self.config, self.feature_dim = copy.deepcopy(config), feature_dim
        self.before_forward, self.before_optimizer = before_forward, before_optimizer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.model = CapacityConfirmationValue(feature_dim, width=config["width"],
                                                    architecture=config["architecture"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], foreach=False)
        self.rng = np.random.default_rng(config["sampler_seed"])
        self.updates, self.pending = 0, None
        self.counts = dict(forwards=0, optimizer_steps=0)
