"""Architecture-bound adapter; historical TD logic and recovery stay unchanged.

Config requires architecture="graph" or "flat". Graph uses width; flat requires
flat_hidden_sizes, with [21, 23] matching the width-32 GCN within 1%.
The distinct snapshot format and exact inherited config check bind architecture
and widths before any live state is changed. Historical snapshots are not migrated.
"""

import copy

import numpy as np

from src.baselines.capacity_matched_value import CapacityMatchedValue
from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.capacity_value_features import FEATURE_NAMES
from src.rl.capacity_value_learner import CapacityValueLearner
from src.rl.networks import torch


class CapacityValueComparisonLearner(CapacityValueLearner):
    format = "capacity-value-comparison-td-v1"

    def __init__(self, config, *, seed, feature_dim, before_forward, before_optimizer):
        architecture = config.get("architecture")
        if architecture not in ("graph", "flat"):
            raise ValueError("architecture must be explicitly graph or flat")
        if type(feature_dim) is not int or feature_dim != len(FEATURE_NAMES):
            raise ValueError("the existing 31-feature public schema is required")
        if architecture == "graph":
            width = config.get("width")
            if type(width) is not int or width <= 0:
                raise ValueError("graph width must be a positive integer")
        else:
            hidden_sizes = config.get("flat_hidden_sizes")
            if (not isinstance(hidden_sizes, (tuple, list)) or not hidden_sizes
                    or any(type(width) is not int or width <= 0 for width in hidden_sizes)):
                raise ValueError("flat_hidden_sizes must be nonempty positive integer widths")

        self.config = copy.deepcopy(config)
        self.feature_dim = feature_dim
        self.before_forward, self.before_optimizer = before_forward, before_optimizer
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(int(seed))
            if architecture == "graph":
                self.model = CapacityTerminalValue(feature_dim, width=self.config["width"])
            else:
                self.model = CapacityMatchedValue(
                    feature_dim, hidden_sizes=self.config["flat_hidden_sizes"])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config["lr"], foreach=False)
        self.rng = np.random.default_rng(int(seed)+1)
        self.updates = 0
        self.counts = dict(forwards=0, optimizer_steps=0)
        self.pending = None
