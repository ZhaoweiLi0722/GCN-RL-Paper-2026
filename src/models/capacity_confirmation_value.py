"""Matched terminal values: only message-passing adjacency is ablated."""

from src.models.capacity_terminal_value import CapacityTerminalValue
from src.rl.networks import torch


class CapacityConfirmationValue(CapacityTerminalValue):
    """Reuse both GCN layers, mean pooling and head, including initialization.

    Construct both architectures under the same seed to obtain exactly equal
    trainable tensors. Identity adjacency consumes no additional random draws.
    Loading a graph state into self_only (or vice versa) is deliberately rejected.
    """

    def __init__(self, feature_dim=31, *, architecture, width=32):
        if (architecture not in ("graph", "self_only")
                or type(feature_dim) is not int or feature_dim != 31
                or type(width) is not int or width <= 0):
            raise ValueError("graph/self_only, 31 features and positive integer width required")
        super().__init__(feature_dim, width=width)
        self.architecture = architecture
        if architecture == "self_only":
            self.encoder.adjacency.copy_(torch.eye(4, dtype=torch.float32))
        self._expected_adjacency = self.encoder.adjacency.clone()

    def load_state_dict(self, state_dict, strict=True, assign=False):
        expected = self.state_dict()
        if not strict or assign or not isinstance(state_dict, dict) or set(state_dict) != set(expected):
            raise ValueError("strict, owned confirmation model state required")
        for key, template in expected.items():
            value = state_dict[key]
            if (not isinstance(value, torch.Tensor) or value.device.type != "cpu"
                    or value.dtype != template.dtype or value.shape != template.shape
                    or not torch.isfinite(value).all()):
                raise ValueError(f"invalid confirmation model tensor: {key}")
        if not torch.equal(state_dict["encoder.adjacency"], self._expected_adjacency):
            raise ValueError("checkpoint adjacency does not match architecture")
        return super().load_state_dict(state_dict, strict=True)
