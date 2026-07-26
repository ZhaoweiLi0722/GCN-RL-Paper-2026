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


def default_torch_device() -> str:
    """Return the fastest available PyTorch device for this workstation."""

    require_torch()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_torch_device(config_device: str | None = None):
    """Resolve a configured device name, treating missing/``auto`` as automatic."""

    require_torch()
    device_name = str(config_device or "auto")
    if device_name == "auto":
        device_name = default_torch_device()
    return torch.device(device_name)


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


    class MLPCorrectionGate(nn.Module):
        """Flat-state classifier deciding which residual groups may deploy."""

        def __init__(
            self,
            input_dim: int,
            hidden_sizes: tuple[int, ...],
            output_dim: int = 1,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = int(input_dim)
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(previous_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                previous_dim = int(hidden_dim)
            output = nn.Linear(previous_dim, int(output_dim))
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)
            layers.append(output)
            self.net = nn.Sequential(*layers)

        def forward(self, state):
            output = self.net(state)
            return output.squeeze(-1) if output.shape[-1] == 1 else output


    class MLPOptionQNetwork(nn.Module):
        """Dueling flat-state Q-network over fixed residual options."""

        def __init__(
            self,
            state_dim: int,
            hidden_sizes: tuple[int, ...],
            num_options: int,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            previous_dim = int(state_dim)
            for hidden_dim in hidden_sizes:
                layers.append(nn.Linear(previous_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                previous_dim = int(hidden_dim)
            self.encoder = nn.Sequential(*layers)
            self.value_head = nn.Linear(previous_dim, 1)
            self.advantage_head = nn.Linear(previous_dim, int(num_options))
            self.correction_gate_head = nn.Linear(
                previous_dim,
                int(num_options) - 1,
            )

        def forward(self, state):
            features = self.encoder(state)
            return self._option_values(features)

        def forward_with_gate(self, state):
            features = self.encoder(state)
            return self._option_values(features), self.correction_gate_head(features)

        def _option_values(self, features):
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage[:, 0:1]

        def correction_gate_logits(self, state):
            return self.correction_gate_head(self.encoder(state))

else:

    class MLPActor:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class MLPCritic:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class MLPCorrectionGate:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()


    class MLPOptionQNetwork:  # pragma: no cover
        def __init__(self, *args, **kwargs):
            require_torch()
