"""Online-boundary critic realignment for deterministic actor-critic agents."""

from __future__ import annotations

from typing import Any

from src.rl.networks import require_torch, torch


def zero_critic_action_input(
    critic: Any,
    critic_target: Any,
    critic_optimizer: Any,
    *,
    action_dim: int,
) -> dict[str, float | str]:
    """Remove inherited action dependence while preserving state features."""

    require_torch()
    action_width = int(action_dim)
    if action_width <= 0:
        raise ValueError("action_dim must be positive for critic realignment")

    container = getattr(critic, "head", None)
    if container is None:
        container = getattr(critic, "net", None)
    if container is None:
        raise TypeError(
            "Critic realignment requires a state-action head or net"
        )
    first_linear = next(
        (
            module
            for module in container.modules()
            if isinstance(module, torch.nn.Linear)
        ),
        None,
    )
    if first_linear is None:
        raise TypeError("Critic state-action container has no linear layer")
    if int(first_linear.in_features) <= action_width:
        raise ValueError(
            "Critic first layer does not contain a state prefix before its "
            "action inputs"
        )

    action_columns = first_linear.weight[:, -action_width:]
    optimizer_state_entries = len(critic_optimizer.state)
    before_l2 = float(action_columns.detach().norm().item())
    state_columns_before = first_linear.weight[:, :-action_width].detach().clone()
    bias_before = (
        None
        if first_linear.bias is None
        else first_linear.bias.detach().clone()
    )
    with torch.no_grad():
        action_columns.zero_()

    if not torch.equal(
        first_linear.weight[:, :-action_width].detach(),
        state_columns_before,
    ):
        raise RuntimeError("Critic realignment changed state-input weights")
    if bias_before is not None and not torch.equal(
        first_linear.bias.detach(),
        bias_before,
    ):
        raise RuntimeError("Critic realignment changed the first-layer bias")

    critic_optimizer.state.clear()
    critic_target.load_state_dict(critic.state_dict())
    after_l2 = float(action_columns.detach().norm().item())
    if after_l2 != 0.0:
        raise RuntimeError("Critic action-input realignment did not reach zero")

    return {
        "online_critic_realignment": "zero_action_columns",
        "online_critic_action_columns_l2_before": before_l2,
        "online_critic_action_columns_l2_after": after_l2,
        "online_critic_optimizer_state_entries_cleared": float(
            optimizer_state_entries
        ),
        "online_critic_action_input_width": float(action_width),
    }
