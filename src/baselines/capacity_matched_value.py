"""Global flat residual using only the existing four-site public feature tensor.

Parameter arithmetic, before constructing either artificial model (biases included):
width-32 GCN: (31+1)*32 + 2*(32+1)*32 + (32+1)*1 = 3,169.
Flat 124 -> 21 -> 23 -> 1:
(124+1)*21 + (21+1)*23 + (23+1)*1 = 3,155.
The trainable gap is 14/3,169 = 0.44177974%, with no padding parameters.
This is a capacity match, not a depth match or performance evidence.
"""

from src.rl.capacity_value_features import FEATURE_NAMES
from src.rl.networks import nn, torch


MATCHED_FLAT_HIDDEN_SIZES = (21, 23)


class CapacityMatchedValue(nn.Module):
    """Site-major flattened MLP, with a zero-initialized scalar residual head."""

    def __init__(self, feature_dim, *, hidden_sizes=MATCHED_FLAT_HIDDEN_SIZES):
        super().__init__()
        if type(feature_dim) is not int or feature_dim != len(FEATURE_NAMES):
            raise ValueError("the existing 31-feature public schema is required")
        if (not isinstance(hidden_sizes, (tuple, list)) or not hidden_sizes
                or any(type(width) is not int or width <= 0 for width in hidden_sizes)):
            raise ValueError("flat hidden_sizes must be nonempty positive integer widths")
        self.feature_dim = feature_dim
        self.hidden_sizes = tuple(hidden_sizes)
        layers = []
        previous = 4 * feature_dim
        for width in self.hidden_sizes:
            layers.extend((nn.Linear(previous, width), nn.ReLU()))
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.head = nn.Linear(previous, 1)
        self.to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.head.weight.zero_()
            self.head.bias.zero_()

    def forward(self, x):
        if (not isinstance(x, torch.Tensor) or x.dtype != torch.float32
                or x.device.type != "cpu" or x.ndim != 3
                or tuple(x.shape[1:]) != (4, self.feature_dim)
                or not torch.isfinite(x).all()):
            raise ValueError("finite float32 [batch,4,31] public beliefs required")
        return self.head(self.encoder(x.flatten(start_dim=1))).squeeze(-1)
