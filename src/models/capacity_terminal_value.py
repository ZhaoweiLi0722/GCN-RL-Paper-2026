"""GCN residual above the unchanged MPC terminal heuristic; no policy head."""

from src.models.gcn import GCNEncoder
from src.rl.networks import nn, torch


class CapacityTerminalValue(nn.Module):
    def __init__(self, feature_dim, *, width=32):
        super().__init__()
        self.feature_dim = feature_dim
        self.encoder = GCNEncoder(feature_dim, (width, width), 4,
                                  tuple((i, (i+1) % 4) for i in range(4)))
        self.head = nn.Sequential(nn.Linear(width, width), nn.ReLU(), nn.Linear(width, 1))
        self.to(device="cpu", dtype=torch.float32)
        with torch.no_grad():
            self.head[-1].weight.zero_()
            self.head[-1].bias.zero_()

    def forward(self, x):
        if (x.dtype != torch.float32 or x.device.type != "cpu" or x.ndim != 3
                or tuple(x.shape[1:]) != (4, self.feature_dim) or not torch.isfinite(x).all()):
            raise ValueError("finite float32 [batch,4,features] public beliefs required")
        return self.head(self.encoder(x).mean(1)).squeeze(-1)
