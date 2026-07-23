"""Graph candidate selector distilled from the post-decision pMYO shield."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import (
    ShieldedPatientPriorityMyopicPolicy,
    get_heuristic_class,
    select_shield_candidate_index,
    shield_candidate_actions,
    shield_rollout_metrics,
)
from src.models.gcn import GraphFeatureExtractor
from src.models.graph_features import build_graph_spec, flat_state_to_node_features
from src.rl.action_projection import project_action
from src.rl.networks import nn, require_torch, resolve_torch_device, torch


if torch is not None:

    class GCNShieldSelectorNetwork(nn.Module):
        """GCN encoder with a classifier over shield candidate actions."""

        def __init__(
            self,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            edges,
            gcn_hidden_sizes,
            head_hidden_sizes,
            num_candidates: int,
            *,
            include_global_context: bool = True,
            edge_weights=None,
        ):
            super().__init__()
            self.extractor = GraphFeatureExtractor(
                node_feature_dim,
                num_facilities,
                num_nodes,
                edges,
                gcn_hidden_sizes,
                include_global_context=include_global_context,
                edge_weights=edge_weights,
            )
            layers: list[nn.Module] = []
            previous_dim = self.extractor.output_dim
            for hidden_dim in head_hidden_sizes:
                layers.append(nn.Linear(previous_dim, int(hidden_dim)))
                layers.append(nn.ReLU())
                previous_dim = int(hidden_dim)
            layers.append(nn.Linear(previous_dim, int(num_candidates)))
            self.head = nn.Sequential(*layers)

        def forward(self, node_features):
            return self.head(self.extractor(node_features))


class GCNShieldSelectorAgent:
    """Fast graph policy that learns the online shield's candidate choice.

    The slow teacher (`pmyo_shield`) evaluates candidate corrections by rolling
    copied environments forward. This agent learns the resulting candidate index
    and deploys the selected candidate directly, giving a graph-aware learned
    policy with pMYO as an anchor and without online lookahead rollouts.
    """

    algorithm = "gcn_pmyo_shield_selector"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        self.algorithm = str(config.get("algorithm", self.algorithm))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(config.get("seed", 0))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.env_config = dict(config.get("env", {}))
        self.graph_spec = build_graph_spec(config, state_dim)
        self.device = resolve_torch_device(config.get("device"))
        selector_config = dict(config.get("shield_selector", {}))
        self.anchor_policy_name = str(selector_config.get("anchor_policy", "pmyo"))
        self.teacher_policy_config = dict(selector_config.get("teacher_policy_config", {}))
        self.teacher_policy_config.setdefault("anchor_policy", self.anchor_policy_name)
        self.teacher_policy_config.setdefault("shield_lookahead", 2)
        self.teacher_policy_config.setdefault("shield_epsilons", [0.005])
        self.teacher_policy_config.setdefault(
            "candidate_groups",
            [
                "replenishment_patient_risk_pressure",
                "replenishment_positive_pressure",
                "reagent_transfer",
                "capacity_transfer",
                "combined_transfer",
            ],
        )
        self.candidate_epsilons = tuple(
            float(value)
            for value in selector_config.get(
                "candidate_epsilons",
                self.teacher_policy_config.get("shield_epsilons", [0.005]),
            )
        )
        self.candidate_groups = tuple(
            str(group)
            for group in selector_config.get(
                "candidate_groups",
                self.teacher_policy_config.get("candidate_groups", ()),
            )
        )
        self.confidence_threshold = float(selector_config.get("confidence_threshold", 0.0))
        self.num_candidates = int(
            selector_config.get(
                "num_candidates",
                candidate_count(self.candidate_epsilons, self.candidate_groups),
            )
        )
        self.anchor_policy = get_heuristic_class(self.anchor_policy_name)(
            state_dim=state_dim,
            action_dim=action_dim,
            config=dict(selector_config.get("anchor_policy_config", {})),
        )
        self.teacher_policy = ShieldedPatientPriorityMyopicPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.teacher_policy_config,
        )
        self.rng = np.random.default_rng(self.seed + 500000)

        self.model = GCNShieldSelectorNetwork(
            self.graph_spec.node_feature_dim,
            self.graph_spec.num_facilities,
            self.graph_spec.num_nodes,
            self.graph_spec.edge_index,
            tuple(config.get("gcn_hidden_sizes", [64, 32])),
            tuple(selector_config.get("hidden_sizes", config.get("hidden_sizes", [128, 64]))),
            self.num_candidates,
            include_global_context=bool(config.get("include_global_context", True)),
            edge_weights=self.graph_spec.edge_weights,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(selector_config.get("lr", config.get("actor_lr", 1e-4))),
        )

    def reset(self) -> None:
        return None

    def select_action(self, state: np.ndarray, explore: bool = False, env=None) -> np.ndarray:
        del explore
        if env is None:
            raise ValueError("GCN shield selector requires the current environment via env=...")
        anchor_action = self.anchor_policy.select_action(state, explore=False, env=env)
        candidates = shield_candidate_actions(
            anchor_action,
            env,
            epsilons=self.candidate_epsilons,
            candidate_groups=self.candidate_groups,
        )
        if len(candidates) <= 1:
            return anchor_action
        index, confidence = self._predict_candidate_decision(state)
        if index != 0 and confidence < self.confidence_threshold:
            index = 0
        if index >= len(candidates):
            index = 0
        return project_action(candidates[index], env_state=env, action_space_info=self.action_dim).action

    def observe(self, *args, **kwargs) -> None:
        return None

    def update(self) -> dict[str, float]:
        return {}

    def pretrain_with_heuristic(self, env, settings: dict[str, Any]) -> dict[str, Any]:
        episodes = int(settings.get("episodes", 0))
        epochs = int(settings.get("epochs", 1))
        batch_size = int(settings.get("batch_size", 128))
        max_steps = int(settings.get("max_steps_per_episode", env.config.episode_horizon))
        seed = int(settings.get("seed", self.seed + 500000))
        if episodes <= 0 or epochs <= 0:
            return {"policy": self._teacher_policy_name(), "samples": 0, "final_loss": 0.0}

        states: list[np.ndarray] = []
        labels: list[int] = []
        changed = 0
        for episode in range(episodes):
            state = env.reset(seed=seed + episode)
            self.teacher_policy.reset()
            self.anchor_policy.reset()
            for _step in range(max_steps):
                label, action = self._teacher_candidate_label_and_action(state, env)
                states.append(np.asarray(state, dtype=np.float32))
                labels.append(int(label))
                changed += int(label != 0)
                state, _reward, done, _info = env.step(action)
                if done:
                    break

        if not states:
            return {"policy": self._teacher_policy_name(), "samples": 0, "final_loss": 0.0}

        summary = self.fit_label_batch(
            np.asarray(states, dtype=np.float32),
            np.asarray(labels, dtype=np.int64),
            {
                "epochs": epochs,
                "batch_size": batch_size,
                "seed": seed,
                "class_weighting": bool(settings.get("class_weighting", True)),
                "class_weight_power": float(settings.get("class_weight_power", 1.0)),
            },
        )
        label_array = np.asarray(labels, dtype=np.int64)
        label_counts = np.bincount(label_array, minlength=self.num_candidates)
        summary.update(
            {
                "policy": self._teacher_policy_name(),
                "changed_fraction": changed / max(len(labels), 1),
                "candidate_count": self.num_candidates,
                "anchor_label_fraction": float(label_counts[0] / max(label_counts.sum(), 1)),
                "non_anchor_label_fraction": float(1.0 - label_counts[0] / max(label_counts.sum(), 1)),
            }
        )
        return summary

    def fit_label_batch(
        self,
        states: np.ndarray,
        labels: np.ndarray,
        settings: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        settings = settings or {}
        state_tensor = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=self.device)
        label_tensor = torch.as_tensor(np.asarray(labels), dtype=torch.long, device=self.device)
        if state_tensor.ndim != 2 or state_tensor.shape[1] != self.state_dim:
            raise ValueError(f"Expected states shape (batch, {self.state_dim}), got {tuple(state_tensor.shape)}")
        if label_tensor.ndim != 1 or label_tensor.shape[0] != state_tensor.shape[0]:
            raise ValueError("labels must be a vector with one label per state")
        if torch.any(label_tensor < 0) or torch.any(label_tensor >= self.num_candidates):
            raise ValueError("labels contain an out-of-range shield candidate index")

        sample_count = int(state_tensor.shape[0])
        if sample_count == 0:
            return {"samples": 0, "final_loss": 0.0}
        epochs = int(settings.get("epochs", 1))
        batch_size = min(max(int(settings.get("batch_size", 128)), 1), sample_count)
        seed = int(settings.get("seed", self.seed + 500000))
        generator = torch.Generator().manual_seed(seed)
        class_weight = (
            self._class_weight(label_tensor, power=float(settings.get("class_weight_power", 1.0)))
            if bool(settings.get("class_weighting", True))
            else None
        )
        final_loss = 0.0

        self.model.train()
        for _epoch in range(max(epochs, 1)):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                logits = self.model(flat_state_to_node_features(state_tensor[indices], self.graph_spec))
                loss = torch.nn.functional.cross_entropy(
                    logits,
                    label_tensor[indices],
                    weight=class_weight,
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()
                final_loss = float(loss.item())
        self.model.eval()
        with torch.no_grad():
            logits = self.model(flat_state_to_node_features(state_tensor, self.graph_spec))
            predictions = torch.argmax(logits, dim=-1)
            accuracy = float((predictions == label_tensor).to(dtype=torch.float32).mean().item())
            anchor_prediction_fraction = float((predictions == 0).to(dtype=torch.float32).mean().item())
        return {
            "samples": sample_count,
            "final_loss": final_loss,
            "train_accuracy": accuracy,
            "anchor_prediction_fraction": anchor_prediction_fraction,
            "non_anchor_prediction_fraction": 1.0 - anchor_prediction_fraction,
        }

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_candidates": self.num_candidates,
                "model": self.model.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint.get("num_candidates", self.num_candidates)) != self.num_candidates:
            raise ValueError("Checkpoint candidate count does not match selector config")
        self.model.load_state_dict(checkpoint["model"])

    def _predict_candidate_index(self, state: np.ndarray) -> int:
        index, _confidence = self._predict_candidate_decision(state)
        return index

    def _predict_candidate_decision(self, state: np.ndarray) -> tuple[int, float]:
        self.model.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            logits = self.model(flat_state_to_node_features(state_tensor, self.graph_spec))
            probabilities = torch.softmax(logits, dim=-1)
            confidence, index = torch.max(probabilities, dim=-1)
            return int(index.item()), float(confidence.item())

    def _teacher_policy_name(self) -> str:
        return "pmyo_shield" if self.anchor_policy_name == "pmyo" else f"{self.anchor_policy_name}_shield"

    def _teacher_candidate_label_and_action(self, state: np.ndarray, env) -> tuple[int, np.ndarray]:
        anchor_action = self.anchor_policy.select_action(state, explore=False, env=env)
        candidates = shield_candidate_actions(
            anchor_action,
            env,
            epsilons=self.candidate_epsilons,
            candidate_groups=self.candidate_groups,
        )
        if len(candidates) <= 1 or self.teacher_policy.shield_lookahead <= 0:
            return 0, anchor_action
        candidate_metrics = [
            shield_rollout_metrics(
                copy.deepcopy(env),
                self.teacher_policy._anchor_policy,
                action,
                horizon=self.teacher_policy.shield_lookahead,
            )
            for action in candidates
        ]
        label = select_shield_candidate_index(self.teacher_policy, candidate_metrics)
        action = project_action(candidates[label], env_state=env, action_space_info=self.action_dim).action
        return int(label), action

    def _class_weight(self, labels, *, power: float = 1.0):
        if power <= 0.0:
            return None
        counts = torch.bincount(labels, minlength=self.num_candidates).to(dtype=torch.float32)
        positive = counts > 0
        if int(positive.sum().item()) <= 1:
            return None
        weights = torch.zeros_like(counts, device=self.device)
        weights[positive] = counts[positive].sum() / (
            float(positive.sum().item()) * counts[positive]
        )
        if power != 1.0:
            weights[positive] = weights[positive].pow(float(power))
            weights[positive] *= float(positive.sum().item()) / weights[positive].sum().clamp_min(1e-12)
        return weights


def candidate_count(epsilons, candidate_groups) -> int:
    groups = set(str(group) for group in candidate_groups)
    per_epsilon = 0
    for group in (
        "replenishment_patient_risk",
        "replenishment_patient_risk_pressure",
        "replenishment_positive_pressure",
    ):
        per_epsilon += int(group in groups)
    for group in ("reagent_transfer", "capacity_transfer", "combined_transfer"):
        per_epsilon += 2 * int(group in groups)
    return 1 + len(tuple(epsilons)) * per_epsilon
