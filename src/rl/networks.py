"""PyTorch neural network modules for RL baselines."""

from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # pragma: no cover - exercised only without torch installed
    torch = None
    nn = None
    F = None


def require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "PyTorch is required for RL training. Install dependencies with "
            "`python3 -m pip install -r requirements.txt`."
        )


if torch is not None:

    class MLPActor(nn.Module):
        """Deterministic actor with tanh-normalized continuous actions."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = state_dim
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, action_dim))
            layers.append(nn.Tanh())
            self.net = nn.Sequential(*layers)

        def forward(self, state):
            return self.net(state)


    class MLPCritic(nn.Module):
        """Q-network for a flat state-action pair."""

        def __init__(self, state_dim: int, action_dim: int, hidden_sizes: tuple[int, ...]):
            super().__init__()
            layers: list[nn.Module] = []
            prev_dim = state_dim + action_dim
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(prev_dim, hidden_dim))
                layers.append(nn.ReLU())
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.net = nn.Sequential(*layers)

        def forward(self, state, action):
            return self.net(torch.cat((state, action), dim=-1))

else:

    class MLPActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class MLPCritic:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
