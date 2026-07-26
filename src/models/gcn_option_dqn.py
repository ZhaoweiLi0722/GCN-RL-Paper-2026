"""GCN Double-DQN over structured residual options around an MDL-2 anchor."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from src.baselines.heuristics import (
    facility_net_action_from_state,
    get_heuristic_class,
    heuristic_settings_for_policy,
)
from src.models.gcn import GraphFeatureExtractor
from src.models.graph_features import (
    build_graph_spec,
    flat_state_to_adaptive_demand_features,
    flat_state_to_node_features,
)
from src.rl.action_projection import project_action
from src.rl.networks import (
    MLPOptionQNetwork,
    require_torch,
    resolve_torch_device,
    torch,
)
from src.rl.preprocessing import (
    FixedObservationScaler,
    reward_scale_from_config,
)
from src.rl.replay_buffer import ReplayBuffer
from src.rl.residual_options import (
    make_explicit_residual_option_specs,
    make_residual_option_specs,
    residual_option_actions_from_env,
    residual_option_labels_from_advantages,
    residual_option_labels_from_actions,
)


if torch is not None:

    class GCNOptionQNetwork(torch.nn.Module):
        """Dueling graph Q-network with advantages relative to the anchor."""

        def __init__(
            self,
            *,
            node_feature_dim: int,
            num_facilities: int,
            num_nodes: int,
            edges,
            gcn_hidden_sizes,
            head_hidden_sizes,
            num_candidates: int,
            include_global_context: bool,
            edge_weights,
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
            layers: list[torch.nn.Module] = []
            previous_dim = self.extractor.output_dim
            for hidden_dim in head_hidden_sizes:
                layers.append(torch.nn.Linear(previous_dim, int(hidden_dim)))
                layers.append(torch.nn.ReLU())
                previous_dim = int(hidden_dim)
            self.encoder = torch.nn.Sequential(*layers)
            self.value_head = torch.nn.Linear(previous_dim, 1)
            self.advantage_head = torch.nn.Linear(
                previous_dim,
                int(num_candidates),
            )
            self.correction_gate_head = torch.nn.Linear(
                previous_dim,
                int(num_candidates) - 1,
            )

        def forward(self, node_features):
            features = self.encoder(self.extractor(node_features))
            return self._option_values(features)

        def forward_with_gate(self, node_features):
            features = self.encoder(self.extractor(node_features))
            return self._option_values(features), self.correction_gate_head(features)

        def _option_values(self, features):
            value = self.value_head(features)
            advantage = self.advantage_head(features)
            return value + advantage - advantage[:, 0:1]

        def correction_gate_logits(self, node_features):
            features = self.encoder(self.extractor(node_features))
            return self.correction_gate_head(features)


class GCNResidualOptionDQNAgent:
    """Choose one graph-aware correction option and fine-tune it with Double-DQN."""

    algorithm = "gcn_residual_mdl2_option_dqn_afd"

    def __init__(self, state_dim: int, action_dim: int, config: dict[str, Any]):
        require_torch()
        self.algorithm = str(config.get("algorithm", self.algorithm))
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.seed = int(config.get("seed", 0))
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.env_config = dict(config.get("env", {}))
        if self.env_config.get("action_mode") != "facility_net":
            raise ValueError("GCN residual option DQN requires env.action_mode='facility_net'")
        self.use_graph_encoder = not self.algorithm.startswith("flat_")
        self.include_adaptive_demand_features = bool(
            config.get("include_adaptive_demand_features", False)
        )
        self.graph_spec = (
            build_graph_spec(config, state_dim)
            if self.use_graph_encoder
            else None
        )
        self.adaptive_feature_spec = (
            build_graph_spec(config, state_dim)
            if self.include_adaptive_demand_features
            else None
        )
        num_facilities = int(self.env_config.get("num_facilities", 0))
        if self.action_dim != 4 * num_facilities:
            raise ValueError("Residual option DQN requires a facility-net action layout")
        self.device = resolve_torch_device(config.get("device"))
        self.gamma = float(config.get("gamma", 0.99))
        self.batch_size = int(config.get("batch_size", 64))
        self.reward_scale = reward_scale_from_config(config)

        option_config = dict(config.get("residual_option", {}))
        self.anchor_policy_name = str(option_config.get("anchor_policy", "mdl2"))
        self.anchor_policy_config = dict(option_config.get("anchor_policy_config", {}))
        self.anchor_settings = heuristic_settings_for_policy(
            self.anchor_policy_name,
            self.anchor_policy_config,
        )
        self.anchor_policy = get_heuristic_class(self.anchor_policy_name)(
            state_dim=state_dim,
            action_dim=action_dim,
            config=self.anchor_policy_config,
        )
        residual_config = dict(config.get("residual_action", {}))
        self.include_base_action_features = bool(
            residual_config.get("enabled", False)
            and residual_config.get("include_base_action_features", False)
        )
        self.observation_scaler = (
            None
            if self.use_graph_encoder
            else FixedObservationScaler.from_config(config, state_dim)
        )
        explicit_options = option_config.get("explicit_options")
        if explicit_options is not None:
            self.option_specs = make_explicit_residual_option_specs(
                explicit_options
            )
        else:
            self.option_specs = make_residual_option_specs(
                option_config.get("epsilons", (0.32, 0.64, 1.0)),
                option_config.get(
                    "candidate_groups",
                    (
                        "replenishment_uniform",
                        "reagent_transfer",
                        "combined_transfer",
                        "reagent_replenishment",
                        "combined_network",
                    ),
                ),
                option_config.get("candidate_signs", (-1.0, 1.0)),
            )
        self.num_options = len(self.option_specs)
        self.online_reward_mode = str(
            option_config.get("online_reward_mode", "environment")
        )
        if self.online_reward_mode not in {"environment", "one_step_anchor_relative"}:
            raise ValueError(
                "residual_option.online_reward_mode must be 'environment' or "
                "'one_step_anchor_relative'"
            )
        self.anchor_q_margin = float(option_config.get("anchor_q_margin", 0.0))
        self.exploration_start = float(option_config.get("exploration_start", 0.20))
        self.exploration_end = float(option_config.get("exploration_end", 0.02))
        self.exploration_decay_steps = max(
            int(option_config.get("exploration_decay_steps", 10000)),
            1,
        )
        self.target_update_interval = max(
            int(option_config.get("target_update_interval", 250)),
            1,
        )
        self.imitation_regularization_weight = float(
            option_config.get("imitation_regularization_weight", 0.10)
        )
        self.option_class_weight_power = max(
            float(option_config.get("class_weight_power", 0.5)),
            0.0,
        )
        default_advantage_scale = (
            1.0 / self.reward_scale if self.reward_scale > 0.0 else 1.0
        )
        self.option_advantage_scale = max(
            float(
                option_config.get(
                    "option_advantage_scale",
                    default_advantage_scale,
                )
            ),
            1e-8,
        )
        self.option_advantage_clip = max(
            float(option_config.get("option_advantage_clip", 1.0)),
            1e-6,
        )
        self.option_min_advantage = max(
            float(option_config.get("option_min_advantage", 0.0)),
            0.0,
        )
        self.option_infeasible_penalty = max(
            float(option_config.get("option_infeasible_penalty", 0.10)),
            0.0,
        )
        self.option_positive_advantage_weight = max(
            float(option_config.get("positive_advantage_weight", 8.0)),
            1.0,
        )
        self.option_infeasible_loss_weight = max(
            float(option_config.get("infeasible_loss_weight", 0.25)),
            0.0,
        )
        self.option_early_stopping_patience = max(
            int(option_config.get("early_stopping_patience", 25)),
            0,
        )
        self.distillation_objective = str(
            option_config.get(
                "distillation_objective",
                "advantage_regression",
            )
        )
        if self.distillation_objective not in {
            "advantage_regression",
            "cost_sensitive_classification",
        }:
            raise ValueError(
                "residual_option.distillation_objective must be "
                "'advantage_regression' or 'cost_sensitive_classification'"
            )
        self.correction_gate_enabled = bool(
            option_config.get("correction_gate_enabled", True)
        )
        self.correction_gate_threshold = float(
            option_config.get("correction_gate_threshold", 0.70)
        )
        if not 0.0 <= self.correction_gate_threshold <= 1.0:
            raise ValueError("correction_gate_threshold must lie in [0, 1]")
        self.correction_gate_loss_weight = max(
            float(option_config.get("correction_gate_loss_weight", 0.001)),
            0.0,
        )
        self.correction_gate_target_mode = str(
            option_config.get(
                "correction_gate_target_mode",
                "any_improving",
            )
        )
        if self.correction_gate_target_mode not in {
            "any_improving",
            "teacher_best",
        }:
            raise ValueError(
                "residual_option.correction_gate_target_mode must be "
                "'any_improving' or 'teacher_best'"
            )
        self.correction_selection_mode = str(
            option_config.get(
                "correction_selection_mode",
                "q_value",
            )
        )
        if self.correction_selection_mode not in {
            "q_value",
            "gate_probability",
        }:
            raise ValueError(
                "residual_option.correction_selection_mode must be "
                "'q_value' or 'gate_probability'"
            )
        self.correction_gate_positive_weight_power = max(
            float(option_config.get("correction_gate_positive_weight_power", 0.5)),
            0.0,
        )
        self.correction_gate_positive_weight_cap = max(
            float(option_config.get("correction_gate_positive_weight_cap", 20.0)),
            1.0,
        )
        self.option_ranking_loss_weight = max(
            float(option_config.get("ranking_loss_weight", 0.10)),
            0.0,
        )
        self.option_ranking_temperature = max(
            float(option_config.get("ranking_temperature", 1.0)),
            1e-6,
        )
        self.imitation_batch_size = int(
            option_config.get("imitation_batch_size", self.batch_size)
        )
        self.rng = np.random.default_rng(self.seed + 710000)
        self.total_action_steps = 0
        self.total_updates = 0
        self.option_selection_counts = np.zeros(self.num_options, dtype=np.int64)
        self.correction_gate_acceptances = 0
        self.correction_gate_rejections = 0
        self.correction_gate_probability_sum = 0.0
        self.correction_gate_evaluations = 0
        self.anchor_relative_reward_sum = 0.0
        self.anchor_relative_reward_abs_sum = 0.0
        self.anchor_relative_reward_positive_count = 0
        self.anchor_relative_reward_count = 0
        self._pending_state: np.ndarray | None = None
        self._pending_option_index: int | None = None
        self._pending_anchor_action: np.ndarray | None = None
        self.imitation_states = None
        self.imitation_labels = None
        self.imitation_targets = None
        self.imitation_weights = None
        self.imitation_option_weights = None
        self.imitation_gate_labels = None
        self.imitation_gate_positive_weights = None
        self.imitation_class_weights = None
        self.imitation_rng = np.random.default_rng(self.seed + 720000)

        option_hidden_sizes = tuple(
            option_config.get(
                "hidden_sizes",
                config.get("hidden_sizes", (128, 64)),
            )
        )
        if self.use_graph_encoder:
            network_kwargs = {
                "node_feature_dim": self.graph_spec.node_feature_dim,
                "num_facilities": self.graph_spec.num_facilities,
                "num_nodes": self.graph_spec.num_nodes,
                "edges": self.graph_spec.edge_index,
                "gcn_hidden_sizes": tuple(config.get("gcn_hidden_sizes", (64, 32))),
                "head_hidden_sizes": option_hidden_sizes,
                "num_candidates": self.num_options,
                "include_global_context": bool(config.get("include_global_context", True)),
                "edge_weights": self.graph_spec.edge_weights,
            }
            self.q_network = GCNOptionQNetwork(**network_kwargs).to(self.device)
        else:
            flat_input_dim = (
                self.state_dim
                + 3 * num_facilities * int(self.include_adaptive_demand_features)
                + (
                    self.action_dim if self.include_base_action_features else 0
                )
            )
            self.q_network = MLPOptionQNetwork(
                flat_input_dim,
                option_hidden_sizes,
                self.num_options,
            ).to(self.device)
        self._zero_initialize_q_head(self.q_network)
        self.target_q_network = copy.deepcopy(self.q_network).to(self.device)
        self.target_q_network.eval()
        self.optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=float(option_config.get("lr", config.get("actor_lr", 1e-4))),
        )
        self.replay_buffer = ReplayBuffer(
            state_dim,
            1,
            int(config.get("replay_capacity", 100000)),
            self.seed,
        )

    @staticmethod
    def _zero_initialize_q_head(network) -> None:
        for output_layer in (
            network.value_head,
            network.advantage_head,
            network.correction_gate_head,
        ):
            torch.nn.init.zeros_(output_layer.weight)
            torch.nn.init.zeros_(output_layer.bias)

    def reset(self) -> None:
        self.anchor_policy.reset()
        self._pending_state = None
        self._pending_option_index = None
        self._pending_anchor_action = None

    def select_action(
        self,
        state: np.ndarray,
        explore: bool = False,
        env=None,
    ) -> np.ndarray:
        if env is None:
            raise ValueError("GCN residual option DQN requires the current env via env=...")
        anchor_action = self.anchor_policy.select_action(state, explore=False, env=env)
        candidates = residual_option_actions_from_env(
            anchor_action,
            env,
            self.option_specs,
        )
        if len(candidates) != self.num_options:
            raise ValueError("Residual option candidate count changed during deployment")
        option_index = self._select_option_index(state, explore=explore)
        self.option_selection_counts[option_index] += 1
        self._pending_state = np.asarray(state, dtype=np.float32).copy()
        self._pending_option_index = int(option_index)
        self._pending_anchor_action = project_action(
            candidates[0],
            env_state=env,
            action_space_info=self.action_dim,
        ).action
        return project_action(
            candidates[option_index],
            env_state=env,
            action_space_info=self.action_dim,
        ).action

    def capture_training_reward_context(
        self,
        state: np.ndarray,
        action: np.ndarray,
        env,
    ) -> dict[str, Any]:
        """Snapshot an exact-CRN MDL-2 counterfactual before the live step."""

        if self.online_reward_mode == "environment":
            return {}
        del state, action
        if self._pending_anchor_action is None:
            raise RuntimeError(
                "Anchor-relative reward requires select_action before context capture"
            )
        return {
            "anchor_env": copy.deepcopy(env),
            "anchor_action": self._pending_anchor_action.copy(),
        }

    def transform_training_reward(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        info: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> float:
        """Return the one-step reward advantage over MDL-2 under exact CRN."""

        if self.online_reward_mode == "environment":
            return float(reward)
        del state, action, next_state, done, info
        if context is None:
            raise ValueError("Anchor-relative reward context is required")
        anchor_env = context["anchor_env"]
        anchor_action = np.asarray(context["anchor_action"], dtype=np.float32)
        _anchor_state, anchor_reward, _anchor_done, _anchor_info = anchor_env.step(
            anchor_action
        )
        relative_reward = float(reward) - float(anchor_reward)
        self.anchor_relative_reward_sum += relative_reward
        self.anchor_relative_reward_abs_sum += abs(relative_reward)
        self.anchor_relative_reward_positive_count += int(relative_reward > 0.0)
        self.anchor_relative_reward_count += 1
        return relative_reward

    def reset_option_diagnostics(self) -> None:
        self.option_selection_counts.fill(0)
        self.correction_gate_acceptances = 0
        self.correction_gate_rejections = 0
        self.correction_gate_probability_sum = 0.0
        self.correction_gate_evaluations = 0
        self.anchor_relative_reward_sum = 0.0
        self.anchor_relative_reward_abs_sum = 0.0
        self.anchor_relative_reward_positive_count = 0
        self.anchor_relative_reward_count = 0

    def option_diagnostics(self) -> dict[str, Any]:
        total = int(self.option_selection_counts.sum())
        group_counts: dict[str, int] = {}
        selected = {
            f"{index}:{option.group}:{option.sign:g}:{option.epsilon:g}": int(count)
            for index, (option, count) in enumerate(
                zip(self.option_specs, self.option_selection_counts)
            )
            if int(count) > 0
        }
        for option, count in zip(
            self.option_specs,
            self.option_selection_counts,
        ):
            if int(count) > 0:
                group_counts[option.group] = (
                    group_counts.get(option.group, 0) + int(count)
                )
        return {
            "total_selections": total,
            "anchor_selections": int(self.option_selection_counts[0]),
            "correction_selections": int(
                self.option_selection_counts[1:].sum()
            ),
            "correction_rate": (
                float(self.option_selection_counts[1:].sum()) / total
                if total
                else 0.0
            ),
            "selected_option_counts": selected,
            "selected_group_counts": dict(sorted(group_counts.items())),
            "correction_gate_acceptances": self.correction_gate_acceptances,
            "correction_gate_rejections": self.correction_gate_rejections,
            "correction_gate_mean_probability": (
                self.correction_gate_probability_sum
                / self.correction_gate_evaluations
                if self.correction_gate_evaluations
                else 0.0
            ),
            "correction_gate_evaluations": self.correction_gate_evaluations,
            "anchor_relative_reward_mean": (
                self.anchor_relative_reward_sum / self.anchor_relative_reward_count
                if self.anchor_relative_reward_count
                else 0.0
            ),
            "anchor_relative_reward_mean_abs": (
                self.anchor_relative_reward_abs_sum
                / self.anchor_relative_reward_count
                if self.anchor_relative_reward_count
                else 0.0
            ),
            "anchor_relative_reward_positive_rate": (
                self.anchor_relative_reward_positive_count
                / self.anchor_relative_reward_count
                if self.anchor_relative_reward_count
                else 0.0
            ),
            "anchor_relative_reward_count": self.anchor_relative_reward_count,
        }

    def observe(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        state_array = np.asarray(state, dtype=np.float32)
        if (
            self._pending_state is not None
            and self._pending_option_index is not None
            and np.allclose(state_array, self._pending_state, rtol=0.0, atol=1e-6)
        ):
            option_index = int(self._pending_option_index)
        else:
            option_index = int(
                residual_option_labels_from_actions(
                    state_array.reshape(1, -1),
                    np.asarray(action, dtype=np.float32).reshape(1, -1),
                    self.env_config,
                    self.anchor_settings,
                    self.option_specs,
                )[0]
            )
        self.add_option_transition(
            state_array,
            option_index,
            reward,
            next_state,
            done,
        )
        self._pending_state = None
        self._pending_option_index = None
        self._pending_anchor_action = None

    def add_option_transition(
        self,
        state: np.ndarray,
        option_index: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with an explicit option label to replay."""

        index = int(option_index)
        if index < 0 or index >= self.num_options:
            raise ValueError(f"Option index {index} is outside [0, {self.num_options})")
        self.replay_buffer.add(
            np.asarray(state, dtype=np.float32),
            np.asarray([index], dtype=np.float32),
            float(reward) * self.reward_scale,
            np.asarray(next_state, dtype=np.float32),
            bool(done),
        )

    def demonstration_option_labels(
        self,
        demonstrations: dict[str, Any],
    ) -> np.ndarray:
        """Return cache-aligned labels without inferring them from actions."""

        option_indices = self._demonstration_option_indices(demonstrations)
        return residual_option_labels_from_advantages(
            np.asarray(demonstrations["option_advantages"])[:, option_indices],
            np.asarray(demonstrations["option_feasible"])[:, option_indices],
            min_advantage=self.option_min_advantage,
        )

    def update(self) -> dict[str, float]:
        if len(self.replay_buffer) < self.batch_size:
            return {}
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions[:, 0], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(
            batch.next_states,
            dtype=torch.float32,
            device=self.device,
        )
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device)

        q_values = self._q_values(self.q_network, states)
        selected_q = q_values.gather(1, actions.unsqueeze(1))
        with torch.no_grad():
            next_online_q, next_gate_logits = self._q_and_gate_values(
                self.q_network,
                next_states,
            )
            if self.correction_gate_enabled:
                next_gate_probabilities = torch.sigmoid(next_gate_logits)
                rejected = next_gate_probabilities < self.correction_gate_threshold
                if bool(rejected.any()):
                    next_online_q = next_online_q.clone()
                    next_online_q[:, 1:][rejected] = -torch.inf
            next_actions = torch.argmax(next_online_q, dim=1, keepdim=True)
            next_target_q = self._q_values(
                self.target_q_network,
                next_states,
            ).gather(1, next_actions)
            targets = rewards + self.gamma * (1.0 - dones) * next_target_q

        td_loss = torch.nn.functional.smooth_l1_loss(selected_q, targets)
        imitation_loss = self._imitation_regularization_loss()
        loss = td_loss
        if imitation_loss is not None and self.imitation_regularization_weight > 0.0:
            loss = loss + self.imitation_regularization_weight * imitation_loss
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
        self.optimizer.step()
        self.total_updates += 1
        if self.total_updates % self.target_update_interval == 0:
            self.target_q_network.load_state_dict(self.q_network.state_dict())
        return {
            "q_loss": float(td_loss.item()),
            "imitation_loss": (
                0.0 if imitation_loss is None else float(imitation_loss.item())
            ),
        }

    def fit_action_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        settings: dict[str, Any] | None = None,
        weights: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Warm-start option values from dense advantages or hard teacher actions."""

        settings = settings or {}
        state_array = np.asarray(states, dtype=np.float32)
        action_array = np.asarray(actions, dtype=np.float32)
        demonstrations = settings.get("demonstrations")
        if demonstrations is not None and "option_advantages" in demonstrations:
            return self._fit_soft_advantage_batch(
                state_array,
                action_array,
                demonstrations,
                settings,
                weights,
            )
        labels = residual_option_labels_from_actions(
            state_array,
            action_array,
            self.env_config,
            self.anchor_settings,
            self.option_specs,
        )
        state_tensor = torch.as_tensor(state_array, dtype=torch.float32, device=self.device)
        label_tensor = torch.as_tensor(labels, dtype=torch.long, device=self.device)
        sample_count = int(state_array.shape[0])
        if sample_count == 0:
            return {"samples": 0, "final_loss": 0.0}
        weight_tensor = self._fit_weights(weights, sample_count)
        class_weights = self._class_weights(label_tensor, weight_tensor)
        epochs = max(int(settings.get("epochs", 1)), 1)
        batch_size = min(
            max(int(settings.get("batch_size", self.batch_size)), 1),
            sample_count,
        )
        generator = torch.Generator().manual_seed(
            int(settings.get("seed", self.seed + 730000))
        )
        final_loss = 0.0
        self.q_network.train()
        for _epoch in range(epochs):
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                logits = self._q_values(
                    self.q_network,
                    state_tensor[indices],
                )
                per_sample = torch.nn.functional.cross_entropy(
                    logits,
                    label_tensor[indices],
                    weight=class_weights,
                    reduction="none",
                )
                batch_weights = weight_tensor[indices]
                loss = (per_sample * batch_weights).sum() / batch_weights.sum().clamp_min(
                    1e-8
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
                self.optimizer.step()
                final_loss = float(loss.item())
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()
        with torch.no_grad():
            logits = self._q_values(self.q_network, state_tensor)
            predictions = torch.argmax(logits, dim=1)
            accuracy = float(
                (predictions == label_tensor).to(dtype=torch.float32).mean().item()
            )
            correction_mask = label_tensor != 0
            correction_recall = float(
                (predictions[correction_mask] != 0)
                .to(dtype=torch.float32)
                .mean()
                .item()
            ) if bool(correction_mask.any()) else 0.0
            correction_accuracy = float(
                (predictions[correction_mask] == label_tensor[correction_mask])
                .to(dtype=torch.float32)
                .mean()
                .item()
            ) if bool(correction_mask.any()) else 0.0
            anchor_prediction_fraction = float(
                (predictions == 0).to(dtype=torch.float32).mean().item()
            )
        if bool(settings.get("retain_for_regularization", False)):
            self.imitation_states = state_tensor.detach()
            self.imitation_labels = label_tensor.detach()
            self.imitation_targets = None
            self.imitation_weights = weight_tensor.detach()
            self.imitation_option_weights = None
            gate_labels = torch.zeros(
                (sample_count, self.num_options - 1),
                dtype=torch.float32,
                device=self.device,
            )
            correction_rows = label_tensor != 0
            if bool(correction_rows.any()):
                gate_labels[
                    correction_rows,
                    label_tensor[correction_rows] - 1,
                ] = 1.0
            self.imitation_gate_labels = gate_labels.detach()
            self.imitation_gate_positive_weights = None
            self.imitation_class_weights = class_weights.detach()
        unique_labels = int(np.unique(labels).size)
        return {
            "samples": sample_count,
            "final_loss": final_loss,
            "train_accuracy": accuracy,
            "correction_recall": correction_recall,
            "correction_accuracy": correction_accuracy,
            "anchor_prediction_fraction": anchor_prediction_fraction,
            "changed_fraction": float(np.mean(labels != 0)),
            "option_count": self.num_options,
            "observed_option_count": unique_labels,
            "target_mode": "hard_label",
        }

    def _fit_soft_advantage_batch(
        self,
        state_array: np.ndarray,
        action_array: np.ndarray,
        demonstrations: dict[str, Any],
        settings: dict[str, Any],
        weights: np.ndarray | None,
    ) -> dict[str, Any]:
        """Regress all clinically feasible option advantages under CRN rollouts."""

        sample_count = int(state_array.shape[0])
        if sample_count == 0:
            return {"samples": 0, "final_loss": 0.0, "target_mode": "soft_advantage"}
        if action_array.ndim != 2 or action_array.shape[0] != sample_count:
            raise ValueError("Option advantage fitting requires aligned state-action rows")
        option_indices = self._demonstration_option_indices(demonstrations)
        cached_advantages = np.asarray(
            demonstrations["option_advantages"],
            dtype=np.float32,
        )
        cached_feasible = np.asarray(
            demonstrations["option_feasible"],
            dtype=bool,
        )
        advantages = cached_advantages[:, option_indices]
        feasible = cached_feasible[:, option_indices]
        if advantages.shape != (sample_count, self.num_options):
            raise ValueError("Option advantages do not match samples and option count")
        if feasible.shape != advantages.shape or not np.all(np.isfinite(advantages)):
            raise ValueError("Option feasibility or advantage values are invalid")
        if not np.all(feasible[:, 0]):
            raise ValueError("The anchor option must be feasible in every demonstration")

        target_labels = residual_option_labels_from_advantages(
            advantages,
            feasible,
            min_advantage=self.option_min_advantage,
        )
        adjusted = advantages.copy()
        if self.num_options > 1 and self.option_min_advantage > 0.0:
            adjusted[:, 1:] -= self.option_min_advantage
        targets = np.clip(
            adjusted / self.option_advantage_scale,
            -self.option_advantage_clip,
            self.option_advantage_clip,
        )
        targets[:, 0] = 0.0
        targets[~feasible] = -self.option_infeasible_penalty
        option_weights = np.where(
            feasible,
            1.0,
            self.option_infeasible_loss_weight,
        ).astype(np.float32)
        option_weights[targets > 0.0] *= self.option_positive_advantage_weight

        state_tensor = torch.as_tensor(
            state_array,
            dtype=torch.float32,
            device=self.device,
        )
        target_tensor = torch.as_tensor(
            targets,
            dtype=torch.float32,
            device=self.device,
        )
        option_weight_tensor = torch.as_tensor(
            option_weights,
            dtype=torch.float32,
            device=self.device,
        )
        label_tensor = torch.as_tensor(
            target_labels,
            dtype=torch.long,
            device=self.device,
        )
        if self.correction_gate_target_mode == "teacher_best":
            gate_labels = np.zeros(
                (sample_count, self.num_options - 1),
                dtype=np.float32,
            )
            correction_rows = target_labels != 0
            gate_labels[
                np.flatnonzero(correction_rows),
                target_labels[correction_rows] - 1,
            ] = 1.0
        else:
            gate_labels = (
                feasible[:, 1:]
                & (advantages[:, 1:] > self.option_min_advantage)
            ).astype(np.float32)
        gate_label_tensor = torch.as_tensor(
            gate_labels,
            dtype=torch.float32,
            device=self.device,
        )
        gate_positive_counts = gate_label_tensor.sum(dim=0)
        gate_negative_counts = float(sample_count) - gate_positive_counts
        gate_positive_weights = torch.ones_like(gate_positive_counts)
        observed_gate_options = gate_positive_counts > 0.0
        gate_positive_weights[observed_gate_options] = (
            gate_negative_counts[observed_gate_options]
            / gate_positive_counts[observed_gate_options].clamp_min(1.0)
        ).pow(self.correction_gate_positive_weight_power)
        gate_positive_weights = gate_positive_weights.clamp(
            max=self.correction_gate_positive_weight_cap
        )
        weight_tensor = self._fit_weights(weights, sample_count)
        ranking_class_weights = self._class_weights(
            label_tensor,
            weight_tensor,
        )[1:]
        classification_weights = self._class_weights(
            label_tensor,
            weight_tensor,
        )
        epochs = max(int(settings.get("epochs", 1)), 1)
        batch_size = min(
            max(int(settings.get("batch_size", self.batch_size)), 1),
            sample_count,
        )
        generator = torch.Generator().manual_seed(
            int(settings.get("seed", self.seed + 730000))
        )
        final_loss = float("inf")
        best_epoch = 0
        best_state = None
        stale_epochs = 0
        epochs_completed = 0
        self.q_network.train()
        for epoch in range(epochs):
            epoch_loss_sum = 0.0
            epoch_weight_sum = 0.0
            permutation = torch.randperm(sample_count, generator=generator)
            for start in range(0, sample_count, batch_size):
                indices = permutation[start : start + batch_size].to(self.device)
                raw_predicted, gate_logits = self._q_and_gate_values(
                    self.q_network,
                    state_tensor[indices],
                )
                batch_weights = weight_tensor[indices]
                predicted = raw_predicted - raw_predicted[:, 0:1]
                if self.distillation_objective == "cost_sensitive_classification":
                    per_sample = torch.nn.functional.cross_entropy(
                        raw_predicted,
                        label_tensor[indices],
                        weight=classification_weights,
                        reduction="none",
                    )
                else:
                    per_option = torch.nn.functional.smooth_l1_loss(
                        predicted,
                        target_tensor[indices],
                        reduction="none",
                    )
                    batch_option_weights = option_weight_tensor[indices]
                    per_sample = (
                        per_option * batch_option_weights
                    ).sum(dim=1) / batch_option_weights.sum(dim=1).clamp_min(1e-8)
                option_loss = (
                    per_sample * batch_weights
                ).sum() / batch_weights.sum().clamp_min(
                    1e-8
                )
                gate_per_option = torch.nn.functional.binary_cross_entropy_with_logits(
                    gate_logits,
                    gate_label_tensor[indices],
                    pos_weight=gate_positive_weights,
                    reduction="none",
                )
                gate_per_sample = gate_per_option.mean(dim=1)
                gate_loss = (
                    gate_per_sample * batch_weights
                ).sum() / batch_weights.sum().clamp_min(1e-8)
                correction_rows = label_tensor[indices] != 0
                if bool(correction_rows.any()):
                    ranking_logits = (
                        predicted[correction_rows, 1:]
                        / self.option_ranking_temperature
                    )
                    ranking_labels = label_tensor[indices][correction_rows] - 1
                    ranking_sample_weights = batch_weights[correction_rows]
                    ranking_per_sample = torch.nn.functional.cross_entropy(
                        ranking_logits,
                        ranking_labels,
                        weight=ranking_class_weights,
                        reduction="none",
                    )
                    ranking_loss = (
                        ranking_per_sample * ranking_sample_weights
                    ).sum() / ranking_sample_weights.sum().clamp_min(1e-8)
                else:
                    ranking_loss = option_loss.new_zeros(())
                loss = (
                    option_loss
                    + self.correction_gate_loss_weight * gate_loss
                    + self.option_ranking_loss_weight * ranking_loss
                )
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=5.0)
                self.optimizer.step()
                batch_mass = float(batch_weights.sum().item())
                epoch_loss_sum += float(loss.item()) * batch_mass
                epoch_weight_sum += batch_mass
            epochs_completed = epoch + 1
            epoch_loss = epoch_loss_sum / max(epoch_weight_sum, 1e-8)
            if epoch_loss < final_loss:
                final_loss = epoch_loss
                best_epoch = epoch + 1
                best_state = {
                    key: value.detach().clone()
                    for key, value in self.q_network.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
            if (
                self.option_early_stopping_patience > 0
                and stale_epochs >= self.option_early_stopping_patience
            ):
                break

        if best_state is not None:
            self.q_network.load_state_dict(best_state)
        self.target_q_network.load_state_dict(self.q_network.state_dict())
        self.q_network.eval()
        with torch.no_grad():
            train_q_values, train_gate_logits = self._q_and_gate_values(
                self.q_network,
                state_tensor,
            )
            predictions = torch.argmax(train_q_values, dim=1)
            prediction_indices = predictions.detach().cpu().numpy()
            accuracy = float(
                (predictions == label_tensor).to(dtype=torch.float32).mean().item()
            )
            correction_mask = label_tensor != 0
            correction_recall = (
                float(
                    (predictions[correction_mask] != 0)
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
                if bool(correction_mask.any())
                else 0.0
            )
            correction_accuracy = (
                float(
                    (predictions[correction_mask] == label_tensor[correction_mask])
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
                if bool(correction_mask.any())
                else 0.0
            )
            anchor_prediction_fraction = float(
                (predictions == 0).to(dtype=torch.float32).mean().item()
            )
            gate_probabilities = torch.sigmoid(
                train_gate_logits
            )
            gate_predictions = gate_probabilities >= self.correction_gate_threshold
            gate_truth = gate_label_tensor.to(dtype=torch.bool)
            gate_accuracy = float(
                (gate_predictions == gate_truth)
                .to(dtype=torch.float32)
                .mean()
                .item()
            )
            gate_recall = (
                float(
                    gate_predictions[gate_truth]
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
                if bool(gate_truth.any())
                else 0.0
            )
            gate_precision = (
                float(
                    gate_truth[gate_predictions]
                    .to(dtype=torch.float32)
                    .mean()
                    .item()
                )
                if bool(gate_predictions.any())
                else 0.0
            )
        row_indices = np.arange(sample_count)
        predicted_advantages = advantages[row_indices, prediction_indices]
        predicted_feasible = feasible[row_indices, prediction_indices]
        feasible_advantages = np.where(feasible, advantages, -np.inf)
        oracle_advantages = np.max(feasible_advantages, axis=1)
        predicted_effective_advantages = np.where(
            predicted_feasible,
            predicted_advantages,
            -self.option_infeasible_penalty * self.option_advantage_scale,
        )
        advantage_regret = oracle_advantages - predicted_effective_advantages
        if bool(settings.get("retain_for_regularization", False)):
            self.imitation_states = state_tensor.detach()
            if self.distillation_objective == "cost_sensitive_classification":
                self.imitation_labels = label_tensor.detach()
                self.imitation_targets = None
                self.imitation_class_weights = classification_weights.detach()
            else:
                self.imitation_labels = None
                self.imitation_targets = target_tensor.detach()
                self.imitation_class_weights = None
            self.imitation_weights = weight_tensor.detach()
            self.imitation_option_weights = option_weight_tensor.detach()
            self.imitation_gate_labels = gate_label_tensor.detach()
            self.imitation_gate_positive_weights = (
                gate_positive_weights.detach()
            )
        return {
            "samples": sample_count,
            "final_loss": final_loss,
            "train_accuracy": accuracy,
            "correction_recall": correction_recall,
            "correction_accuracy": correction_accuracy,
            "anchor_prediction_fraction": anchor_prediction_fraction,
            "changed_fraction": float(np.mean(target_labels != 0)),
            "option_count": self.num_options,
            "observed_option_count": int(np.unique(target_labels).size),
            "target_mode": (
                "soft_advantage"
                if self.distillation_objective == "advantage_regression"
                else self.distillation_objective
            ),
            "best_epoch": best_epoch,
            "epochs_completed": epochs_completed,
            "positive_advantage_weight": self.option_positive_advantage_weight,
            "infeasible_loss_weight": self.option_infeasible_loss_weight,
            "correction_gate_threshold": self.correction_gate_threshold,
            "correction_gate_target_mode": self.correction_gate_target_mode,
            "correction_gate_accuracy": gate_accuracy,
            "correction_gate_recall": gate_recall,
            "correction_gate_precision": gate_precision,
            "correction_gate_prediction_fraction": float(
                gate_predictions.any(dim=1).to(dtype=torch.float32).mean().item()
            ),
            "ranking_loss_weight": self.option_ranking_loss_weight,
            "ranking_temperature": self.option_ranking_temperature,
            "mean_best_advantage": float(
                np.mean(oracle_advantages)
            ),
            "mean_predicted_advantage": float(
                np.mean(predicted_advantages)
            ),
            "mean_advantage_regret": float(np.mean(advantage_regret)),
            "predicted_feasible_fraction": float(
                np.mean(predicted_feasible)
            ),
            "predicted_material_improvement_fraction": float(
                np.mean(predicted_advantages > self.option_min_advantage)
            ),
        }

    def _validate_option_metadata(
        self,
        demonstrations: dict[str, Any],
    ) -> None:
        indices = self._demonstration_option_indices(demonstrations)
        if (
            len(np.asarray(demonstrations["option_groups"])) != self.num_options
            or not np.array_equal(indices, np.arange(self.num_options))
        ):
            raise ValueError(
                "Checkpoint residual option ordering does not match the agent config"
            )

    def _demonstration_option_indices(
        self,
        demonstrations: dict[str, Any],
    ) -> np.ndarray:
        """Map this agent's option subset into a cache's option table."""

        groups = np.asarray(demonstrations["option_groups"], dtype="U32")
        epsilons = np.asarray(
            demonstrations["option_epsilons"],
            dtype=np.float32,
        )
        signs = np.asarray(demonstrations["option_signs"], dtype=np.float32)
        expected_groups = np.asarray(
            [option.group for option in self.option_specs],
            dtype="U32",
        )
        expected_epsilons = np.asarray(
            [option.epsilon for option in self.option_specs],
            dtype=np.float32,
        )
        expected_signs = np.asarray(
            [option.sign for option in self.option_specs],
            dtype=np.float32,
        )
        if epsilons.shape != groups.shape or signs.shape != groups.shape:
            raise ValueError("Cached residual option metadata arrays must align")
        indices: list[int] = []
        for group, epsilon, sign in zip(
            expected_groups,
            expected_epsilons,
            expected_signs,
        ):
            matches = np.flatnonzero(
                (groups == group)
                & np.isclose(epsilons, epsilon, rtol=0.0, atol=1e-6)
                & np.isclose(signs, sign, rtol=0.0, atol=1e-6)
            )
            if matches.size != 1:
                raise ValueError(
                    "Cached residual options do not contain a unique match for "
                    f"{group}:{sign:g}:{epsilon:g}"
                )
            indices.append(int(matches[0]))
        return np.asarray(indices, dtype=np.int64)

    def pretrain_with_heuristic(self, env, settings: dict[str, Any]) -> dict[str, Any]:
        del env, settings
        return {"policy": self.anchor_policy_name, "samples": 0, "final_loss": 0.0}

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "algorithm": self.algorithm,
                "state_dim": self.state_dim,
                "action_dim": self.action_dim,
                "num_options": self.num_options,
                "use_graph_encoder": self.use_graph_encoder,
                "option_groups": [
                    option.group for option in self.option_specs
                ],
                "option_epsilons": [
                    option.epsilon for option in self.option_specs
                ],
                "option_signs": [
                    option.sign for option in self.option_specs
                ],
                "q_network": self.q_network.state_dict(),
                "target_q_network": self.target_q_network.state_dict(),
            },
            output_path,
        )

    def load_actor(self, path: str | Path) -> None:
        checkpoint = torch.load(path, map_location=self.device)
        if int(checkpoint.get("num_options", self.num_options)) != self.num_options:
            raise ValueError("Checkpoint residual option count does not match config")
        if bool(
            checkpoint.get("use_graph_encoder", self.use_graph_encoder)
        ) != self.use_graph_encoder:
            raise ValueError("Checkpoint option encoder does not match config")
        checkpoint_metadata = {
            "option_groups": np.asarray(
                checkpoint.get(
                    "option_groups",
                    [option.group for option in self.option_specs],
                ),
                dtype="U32",
            ),
            "option_epsilons": np.asarray(
                checkpoint.get(
                    "option_epsilons",
                    [option.epsilon for option in self.option_specs],
                ),
                dtype=np.float32,
            ),
            "option_signs": np.asarray(
                checkpoint.get(
                    "option_signs",
                    [option.sign for option in self.option_specs],
                ),
                dtype=np.float32,
            ),
        }
        self._validate_option_metadata(checkpoint_metadata)
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_q_network.load_state_dict(
            checkpoint.get("target_q_network", checkpoint["q_network"])
        )
        self.q_network.eval()
        self.target_q_network.eval()

    def _select_option_index(self, state: np.ndarray, *, explore: bool) -> int:
        epsilon = self._exploration_epsilon()
        self.total_action_steps += int(explore)
        if explore and self.rng.random() < epsilon:
            return int(self.rng.integers(self.num_options))
        self.q_network.eval()
        with torch.no_grad():
            state_tensor = torch.as_tensor(
                np.asarray(state, dtype=np.float32),
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)
            q_values_batch, gate_logits = self._q_and_gate_values(
                self.q_network,
                state_tensor,
            )
            q_values = q_values_batch[0]
            if self.correction_gate_enabled:
                gate_probabilities = torch.sigmoid(gate_logits[0])
                self.correction_gate_probability_sum += float(
                    gate_probabilities.max().item()
                )
                self.correction_gate_evaluations += 1
                allowed = gate_probabilities >= self.correction_gate_threshold
                if not bool(allowed.any()):
                    self.correction_gate_rejections += 1
                    return 0
                if self.correction_selection_mode == "gate_probability":
                    masked_gate_probabilities = gate_probabilities.clone()
                    masked_gate_probabilities[~allowed] = -torch.inf
                    index = int(
                        torch.argmax(masked_gate_probabilities).item()
                    ) + 1
                else:
                    masked_q_values = q_values.clone()
                    masked_q_values[1:][~allowed] = -torch.inf
                    index = int(torch.argmax(masked_q_values).item())
            else:
                index = int(torch.argmax(q_values).item())
            if (
                index != 0
                and float(q_values[index].item())
                < float(q_values[0].item()) + self.anchor_q_margin
            ):
                if self.correction_gate_enabled:
                    self.correction_gate_rejections += 1
                return 0
            if self.correction_gate_enabled:
                if index == 0:
                    self.correction_gate_rejections += 1
                else:
                    self.correction_gate_acceptances += 1
            return index

    def _exploration_epsilon(self) -> float:
        fraction = min(self.total_action_steps / self.exploration_decay_steps, 1.0)
        return self.exploration_start + fraction * (
            self.exploration_end - self.exploration_start
        )

    def _fit_weights(self, weights: np.ndarray | None, sample_count: int):
        if weights is None:
            return torch.ones(sample_count, dtype=torch.float32, device=self.device)
        values = np.asarray(weights, dtype=np.float32)
        if values.shape != (sample_count,) or not np.all(np.isfinite(values)):
            raise ValueError("Option imitation weights must be a finite sample vector")
        if np.any(values < 0.0) or float(values.sum()) <= 0.0:
            raise ValueError("Option imitation weights must be non-negative with positive mass")
        values = values / float(values.mean())
        return torch.as_tensor(values, dtype=torch.float32, device=self.device)

    def _class_weights(self, labels, sample_weights):
        weights = torch.ones(
            self.num_options,
            dtype=torch.float32,
            device=self.device,
        )
        if self.option_class_weight_power <= 0.0:
            return weights
        counts = torch.zeros_like(weights)
        counts.scatter_add_(0, labels, sample_weights)
        observed = counts > 0.0
        if int(observed.sum().item()) <= 1:
            return weights
        mean_count = counts[observed].mean()
        weights[observed] = (
            mean_count / counts[observed].clamp_min(1e-8)
        ).pow(self.option_class_weight_power)
        weights[observed] /= weights[observed].mean().clamp_min(1e-8)
        weights[~observed] = 0.0
        return weights

    def _imitation_regularization_loss(self):
        if (
            self.imitation_states is None
            or self.imitation_weights is None
            or self.imitation_regularization_weight <= 0.0
        ):
            return None
        sample_count = int(self.imitation_states.shape[0])
        batch_size = min(max(self.imitation_batch_size, 1), sample_count)
        indices = self.imitation_rng.choice(sample_count, size=batch_size, replace=False)
        index_tensor = torch.as_tensor(indices, dtype=torch.long, device=self.device)
        logits, gate_logits = self._q_and_gate_values(
            self.q_network,
            self.imitation_states[index_tensor],
        )
        if self.imitation_targets is not None:
            logits = logits - logits[:, 0:1]
            per_option = torch.nn.functional.smooth_l1_loss(
                logits,
                self.imitation_targets[index_tensor],
                reduction="none",
            )
            if self.imitation_option_weights is None:
                per_sample = per_option.mean(dim=1)
            else:
                option_weights = self.imitation_option_weights[index_tensor]
                per_sample = (
                    per_option * option_weights
                ).sum(dim=1) / option_weights.sum(dim=1).clamp_min(1e-8)
            weights = self.imitation_weights[index_tensor]
            option_loss = (
                per_sample * weights
            ).sum() / weights.sum().clamp_min(1e-8)
            if self.imitation_gate_labels is None:
                return option_loss
            gate_per_sample = torch.nn.functional.binary_cross_entropy_with_logits(
                gate_logits,
                self.imitation_gate_labels[index_tensor],
                pos_weight=self.imitation_gate_positive_weights,
                reduction="none",
            )
            if gate_per_sample.ndim == 2:
                gate_per_sample = gate_per_sample.mean(dim=1)
            gate_loss = (
                gate_per_sample * weights
            ).sum() / weights.sum().clamp_min(1e-8)
            return option_loss + self.correction_gate_loss_weight * gate_loss
        if self.imitation_labels is None or self.imitation_class_weights is None:
            return None
        per_sample = torch.nn.functional.cross_entropy(
            logits,
            self.imitation_labels[index_tensor],
            weight=self.imitation_class_weights,
            reduction="none",
        )
        weights = self.imitation_weights[index_tensor]
        return (per_sample * weights).sum() / weights.sum().clamp_min(1e-8)

    def _q_values(self, network, states):
        if self.use_graph_encoder:
            return network(flat_state_to_node_features(states, self.graph_spec))
        return network(self._flat_network_inputs(states))

    def _gate_logits(self, network, states):
        return self._q_and_gate_values(network, states)[1]

    def _q_and_gate_values(self, network, states):
        if self.use_graph_encoder:
            node_features = flat_state_to_node_features(states, self.graph_spec)
            return network.forward_with_gate(node_features)
        return network.forward_with_gate(self._flat_network_inputs(states))

    def _flat_network_inputs(self, states):
        if self.observation_scaler is None:
            raise RuntimeError("Flat option policy is missing its observation scaler")
        normalized = self.observation_scaler.normalize_tensor(states)
        input_parts = [normalized]
        if self.adaptive_feature_spec is not None:
            input_parts.append(
                flat_state_to_adaptive_demand_features(
                    states,
                    self.adaptive_feature_spec,
                ).flatten(start_dim=1)
            )
        if not self.include_base_action_features:
            return torch.cat(input_parts, dim=1)
        state_rows = states.detach().cpu().numpy()
        base_actions = np.stack(
            [
                facility_net_action_from_state(
                    row,
                    self.env_config,
                    settings=self.anchor_settings,
                )
                for row in state_rows
            ],
            axis=0,
        )
        base_tensor = torch.as_tensor(
            base_actions,
            dtype=states.dtype,
            device=states.device,
        )
        input_parts.append(base_tensor)
        return torch.cat(input_parts, dim=1)
