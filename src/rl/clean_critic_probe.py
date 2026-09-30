"""Fresh supervised GCN critic; no actor, replay, TD or environment updates."""

from __future__ import annotations

import copy
import random

import numpy as np
import torch

from src.models.graph_features import flat_state_to_node_features
from src.rl.critic_probe_contract import train_only_constant, select_supported


def fresh_critic(agent, seed):
    model = copy.deepcopy(agent.critic).cpu()
    # The locked critic contains Linear parameters only; graph adjacency is fixed.
    parameter_ids = set()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                module.reset_parameters()
                parameter_ids.update(id(p) for p in module.parameters(recurse=False))
    if parameter_ids != {id(p) for p in model.parameters()}:
        raise ValueError("Unspecified parameter initialization in locked critic")
    return model.to(agent.device).train().requires_grad_(True)


def tensor_inputs(agent, public):
    states = torch.tensor(np.stack([x.observation for x in public]), dtype=torch.float32,
                          device=agent.device)
    actions = torch.tensor(np.stack([x.requests for x in public]), dtype=torch.float32,
                           device=agent.device)
    nodes = flat_state_to_node_features(states, agent.graph_spec).detach()
    shape = actions.shape
    actions = agent._critic_actions_tensor(actions.reshape(-1, shape[-1])).reshape(shape).detach()
    return nodes, actions


def advantage(model, nodes, actions):
    count, choices, width = actions.shape
    expanded = nodes[:, None].expand(-1, choices, -1, -1).reshape(
        count * choices, *nodes.shape[1:])
    values = model(expanded, actions.reshape(count * choices, width)).reshape(count, choices)
    return values - values[:, :1]


def target_scale(targets, support, roles, floor):
    targets = np.asarray(targets, dtype=np.float64)
    support = np.asarray(support, dtype=bool)
    if (targets.ndim != 2 or targets.shape != support.shape or len(roles) != len(targets)
            or set(roles) != {"train"} or not support[:, 0].all()
            or not np.isfinite(targets).all() or not np.all(targets[:, 0] == 0)
            or not np.isfinite(floor) or floor <= 0):
        raise ValueError("Target scaling requires finite training-only frozen contrasts")
    return max(float(np.sqrt(np.mean(np.sum(targets ** 2 * support, axis=1)
                                        / support.sum(axis=1)))), floor)


def fit(model, agent, public, targets, support, roles, settings, budget, log):
    scale = target_scale(targets, support, roles, settings["scale_floor"])
    nodes, actions = tensor_inputs(agent, public)
    target = torch.tensor(np.asarray(targets) / scale, dtype=torch.float32, device=agent.device)
    mask = torch.tensor(np.asarray(support), dtype=torch.float32, device=agent.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings["learning_rate"])
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    history = []
    for update in range(1, settings["updates"] + 1):
        budget.check()
        optimizer.zero_grad(set_to_none=True)
        predicted = advantage(model, nodes, actions)
        loss = (((predicted - target) ** 2 * mask).sum(dim=1) / mask.sum(dim=1)).mean()
        if not torch.isfinite(loss):
            raise ValueError("Nonfinite critic loss")
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings["gradient_clip"],
                                             error_if_nonfinite=True)
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in model.parameters()):
            raise ValueError("Nonfinite critic parameters")
        if update == 1 or update % settings["log_every"] == 0:
            row = {"update": update, "pre_step_normalized_mse": float(loss.detach().cpu()),
                   "gradient_norm_before_clip": float(norm.detach().cpu())}
            history.append(row)
            log(row)
    model.eval()
    with torch.no_grad():
        final_predictions = advantage(model, nodes, actions).cpu().numpy() * scale
    changed = any(not torch.equal(initial[k], v.detach().cpu()) for k, v in model.state_dict().items())
    if not changed:
        raise ValueError("Critic did not change")
    return optimizer, {"updates": settings["updates"], "target_scale": scale,
                       "training_predictions": final_predictions.tolist(), "history": history,
                       "parameter_changed": changed, "actor_updates": 0, "ddpg_updates": 0}


def checkpoint(path, model, optimizer, *, updates, scale, seed):
    payload = {"critic": {k: v.detach().cpu() for k, v in model.state_dict().items()},
               "optimizer": optimizer.state_dict() if optimizer is not None else None,
               "updates": updates, "target_scale": scale, "init_seed": seed,
               "torch_rng": torch.get_rng_state(), "numpy_rng": np.random.get_state(),
               "python_rng": random.getstate(), "actor_updates": 0, "ddpg_updates": 0}
    if next(model.parameters()).device.type == "mps":
        payload["mps_rng"] = torch.mps.get_rng_state()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def seal_predictions(model, agent, public, support, scale, train_costs, train_support):
    constant, means = train_only_constant(train_costs, train_support, ["train"] * len(train_costs))
    with torch.no_grad():
        prediction = advantage(model, *tensor_inputs(agent, public)).cpu().numpy() * scale
    selected = select_supported(prediction, support)
    support = np.asarray(support, dtype=bool)
    return {"advantage_predictions": prediction.tolist(), "critic_actions": selected.tolist(),
            "frozen_actions": [0] * len(public),
            "mdl2_actions": np.where(support[:, 1], 1, 0).tolist(),
            "constant_actions": np.where(support[:, constant], constant, 0).tolist(),
            "training_selected_constant": constant, "training_constant_mean_costs": means.tolist(),
            "selection_uses_test_labels": False}
