"""Support-matched structured exploration for specimen-routing actions."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
from typing import Any, Mapping

import numpy as np

from src.rl.residual_options import (
    ResidualOptionSpec,
    make_explicit_residual_option_specs,
    residual_option_actions_from_env,
)


DEFAULT_SPECIMEN_OPTIONS = (
    {"group": "specimen_transfer", "epsilon": 0.05, "sign": -1.0},
    {"group": "specimen_transfer", "epsilon": 0.05, "sign": 1.0},
    {"group": "specimen_transfer", "epsilon": 0.10, "sign": -1.0},
    {"group": "specimen_transfer", "epsilon": 0.10, "sign": 1.0},
)


class StructuredSpecimenExplorer:
    """Occasionally replace the specimen slice with a legal anchor option.

    The deterministic actor, correction gate, and ordinary OU process are
    evaluated first. On selected behavior-policy steps, only the specimen
    slice is replaced by one uniformly sampled option around the MDL-2 anchor;
    all other action groups retain the actor path exactly.
    """

    format_version = 1

    def __init__(
        self,
        *,
        action_dim: int,
        num_facilities: int,
        seed: int,
        settings: Mapping[str, Any] | None = None,
    ) -> None:
        self.action_dim = int(action_dim)
        self.num_facilities = int(num_facilities)
        self.settings = deepcopy(dict(settings or {}))
        self.enabled = bool(self.settings.get("enabled", False))
        self.selection_probability = float(
            self.settings.get("selection_probability", 0.0)
        )
        if (
            not np.isfinite(self.selection_probability)
            or not 0.0 <= self.selection_probability <= 1.0
        ):
            raise ValueError(
                "structured specimen exploration selection_probability "
                "must lie in [0, 1]"
            )
        if self.enabled and self.selection_probability <= 0.0:
            raise ValueError(
                "enabled structured specimen exploration requires a "
                "positive selection_probability"
            )
        if self.enabled and (
            self.num_facilities <= 0
            or self.action_dim != 4 * self.num_facilities
        ):
            raise ValueError(
                "structured specimen exploration requires a facility-net "
                "action layout"
            )
        selection_mode = str(
            self.settings.get("selection_mode", "uniform")
        ).lower()
        if selection_mode != "uniform":
            raise ValueError(
                "structured specimen exploration selection_mode must be "
                "'uniform'"
            )
        if not bool(
            self.settings.get("apply_after_correction_gate", True)
        ):
            raise ValueError(
                "structured specimen exploration must be applied after the "
                "correction gate"
            )
        self.selection_mode = selection_mode
        self.distinct_tolerance = max(
            float(self.settings.get("distinct_tolerance", 1e-7)),
            0.0,
        )
        raw_options = self.settings.get(
            "options",
            DEFAULT_SPECIMEN_OPTIONS,
        )
        self.option_specs = make_explicit_residual_option_specs(raw_options)
        if any(
            not option.is_anchor and option.group != "specimen_transfer"
            for option in self.option_specs
        ):
            raise ValueError(
                "structured specimen exploration supports only the "
                "specimen_transfer group"
            )
        self.option_labels = tuple(
            self._option_label(option)
            for option in self.option_specs
        )
        seed_offset = int(self.settings.get("seed_offset", 684_211))
        self.rng = np.random.default_rng(int(seed) + seed_offset)
        self.total_option_counts: Counter[str] = Counter(
            {label: 0 for label in self.option_labels}
        )
        self.total_decisions = 0
        self.total_selections = 0
        self.total_correction_selections = 0
        self.total_behaviorally_distinct = 0
        self.total_specimen_linf_delta = 0.0
        self.max_specimen_linf_delta = 0.0
        self.reset_episode()
        self.last_decision = self._empty_last_decision()

    def reset_episode(self) -> None:
        """Reset reporting counters without altering the exploration RNG."""

        self.episode_option_counts: Counter[str] = Counter(
            {label: 0 for label in self.option_labels}
        )
        self.episode_decisions = 0
        self.episode_selections = 0
        self.episode_correction_selections = 0
        self.episode_behaviorally_distinct = 0
        self.episode_specimen_linf_delta = 0.0
        self.episode_max_specimen_linf_delta = 0.0

    def apply(
        self,
        policy_action: np.ndarray,
        anchor_action: np.ndarray,
        *,
        env: Any | None,
    ) -> np.ndarray:
        """Return the behavior action before final environment projection."""

        policy = self._action_array(policy_action, "policy_action")
        anchor = self._action_array(anchor_action, "anchor_action")
        self.total_decisions += 1
        self.episode_decisions += 1
        self.last_decision = self._empty_last_decision()
        if not self.enabled:
            return policy.copy()
        if env is None:
            raise ValueError(
                "structured specimen exploration requires the live environment"
            )
        if float(self.rng.random()) >= self.selection_probability:
            return policy.copy()

        option_actions = residual_option_actions_from_env(
            anchor,
            env,
            self.option_specs,
        )
        option_index = int(self.rng.integers(len(option_actions)))
        option = self.option_specs[option_index]
        label = self.option_labels[option_index]
        selected = policy.copy()
        selected[: self.num_facilities] = np.asarray(
            option_actions[option_index],
            dtype=np.float32,
        )[: self.num_facilities]
        self.last_decision = {
            "selected": True,
            "option_index": option_index,
            "option_label": label,
            "is_anchor": bool(option.is_anchor),
            "recorded": False,
        }
        return selected

    def record_projected_action(
        self,
        policy_action: np.ndarray,
        behavior_action: np.ndarray,
    ) -> None:
        """Record whether a selected option survived final projection."""

        if not bool(self.last_decision.get("selected", False)):
            return
        if bool(self.last_decision.get("recorded", False)):
            raise RuntimeError(
                "structured specimen exploration decision was recorded twice"
            )
        policy = self._action_array(policy_action, "policy_action")
        behavior = self._action_array(behavior_action, "behavior_action")
        specimen_delta = float(
            np.max(
                np.abs(
                    behavior[: self.num_facilities]
                    - policy[: self.num_facilities]
                )
            )
        )
        distinct = specimen_delta > self.distinct_tolerance
        label = str(self.last_decision["option_label"])
        is_anchor = bool(self.last_decision["is_anchor"])
        self.total_selections += 1
        self.episode_selections += 1
        self.total_option_counts[label] += 1
        self.episode_option_counts[label] += 1
        if not is_anchor:
            self.total_correction_selections += 1
            self.episode_correction_selections += 1
        if distinct:
            self.total_behaviorally_distinct += 1
            self.episode_behaviorally_distinct += 1
        self.total_specimen_linf_delta += specimen_delta
        self.episode_specimen_linf_delta += specimen_delta
        self.max_specimen_linf_delta = max(
            self.max_specimen_linf_delta,
            specimen_delta,
        )
        self.episode_max_specimen_linf_delta = max(
            self.episode_max_specimen_linf_delta,
            specimen_delta,
        )
        self.last_decision.update(
            {
                "recorded": True,
                "behaviorally_distinct": distinct,
                "specimen_linf_delta": specimen_delta,
            }
        )

    def summary(self) -> dict[str, Any]:
        """Return cumulative and current-episode auditable diagnostics."""

        return {
            "enabled": self.enabled,
            "selection_mode": self.selection_mode,
            "selection_probability": self.selection_probability,
            "option_labels": list(self.option_labels),
            "total_decisions": self.total_decisions,
            "total_selections": self.total_selections,
            "total_selection_rate": self.total_selections
            / max(self.total_decisions, 1),
            "total_correction_selections": (
                self.total_correction_selections
            ),
            "total_behaviorally_distinct": (
                self.total_behaviorally_distinct
            ),
            "total_behaviorally_distinct_rate": (
                self.total_behaviorally_distinct
                / max(self.total_decisions, 1)
            ),
            "total_mean_selected_specimen_linf_delta": (
                self.total_specimen_linf_delta
                / max(self.total_selections, 1)
            ),
            "max_specimen_linf_delta": self.max_specimen_linf_delta,
            "total_option_counts": dict(self.total_option_counts),
            "episode_decisions": self.episode_decisions,
            "episode_selections": self.episode_selections,
            "episode_selection_rate": self.episode_selections
            / max(self.episode_decisions, 1),
            "episode_correction_selections": (
                self.episode_correction_selections
            ),
            "episode_behaviorally_distinct": (
                self.episode_behaviorally_distinct
            ),
            "episode_mean_selected_specimen_linf_delta": (
                self.episode_specimen_linf_delta
                / max(self.episode_selections, 1)
            ),
            "episode_max_specimen_linf_delta": (
                self.episode_max_specimen_linf_delta
            ),
            "episode_option_counts": dict(self.episode_option_counts),
            "last_decision": dict(self.last_decision),
        }

    def state_dict(self) -> dict[str, Any]:
        """Capture exact RNG and diagnostic state for atomic resumption."""

        return {
            "format_version": self.format_version,
            "enabled": self.enabled,
            "selection_probability": self.selection_probability,
            "option_labels": list(self.option_labels),
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "total_decisions": self.total_decisions,
            "total_selections": self.total_selections,
            "total_correction_selections": (
                self.total_correction_selections
            ),
            "total_behaviorally_distinct": (
                self.total_behaviorally_distinct
            ),
            "total_specimen_linf_delta": self.total_specimen_linf_delta,
            "max_specimen_linf_delta": self.max_specimen_linf_delta,
            "total_option_counts": dict(self.total_option_counts),
            "episode_decisions": self.episode_decisions,
            "episode_selections": self.episode_selections,
            "episode_correction_selections": (
                self.episode_correction_selections
            ),
            "episode_behaviorally_distinct": (
                self.episode_behaviorally_distinct
            ),
            "episode_specimen_linf_delta": (
                self.episode_specimen_linf_delta
            ),
            "episode_max_specimen_linf_delta": (
                self.episode_max_specimen_linf_delta
            ),
            "episode_option_counts": dict(self.episode_option_counts),
            "last_decision": dict(self.last_decision),
        }

    def load_state_dict(
        self,
        state: Mapping[str, Any],
        *,
        allow_enabled_mismatch: bool = False,
    ) -> None:
        """Restore a state, permitting only an explicit episode-0 arm fork."""

        saved = dict(state)
        if int(saved.get("format_version", -1)) != self.format_version:
            raise ValueError(
                "unsupported structured specimen exploration state format"
            )
        if (
            bool(saved.get("enabled", False)) != self.enabled
            and not allow_enabled_mismatch
        ):
            raise ValueError(
                "structured specimen exploration enabled setting does not match"
            )
        if float(saved["selection_probability"]) != (
            self.selection_probability
        ):
            raise ValueError(
                "structured specimen exploration probability does not match"
            )
        if tuple(saved["option_labels"]) != self.option_labels:
            raise ValueError(
                "structured specimen exploration options do not match"
            )
        self.rng.bit_generator.state = deepcopy(saved["rng_state"])
        self.total_decisions = int(saved["total_decisions"])
        self.total_selections = int(saved["total_selections"])
        self.total_correction_selections = int(
            saved["total_correction_selections"]
        )
        self.total_behaviorally_distinct = int(
            saved["total_behaviorally_distinct"]
        )
        self.total_specimen_linf_delta = float(
            saved["total_specimen_linf_delta"]
        )
        self.max_specimen_linf_delta = float(
            saved["max_specimen_linf_delta"]
        )
        self.total_option_counts = self._restore_counts(
            saved["total_option_counts"]
        )
        self.episode_decisions = int(saved["episode_decisions"])
        self.episode_selections = int(saved["episode_selections"])
        self.episode_correction_selections = int(
            saved["episode_correction_selections"]
        )
        self.episode_behaviorally_distinct = int(
            saved["episode_behaviorally_distinct"]
        )
        self.episode_specimen_linf_delta = float(
            saved["episode_specimen_linf_delta"]
        )
        self.episode_max_specimen_linf_delta = float(
            saved["episode_max_specimen_linf_delta"]
        )
        self.episode_option_counts = self._restore_counts(
            saved["episode_option_counts"]
        )
        self.last_decision = dict(saved["last_decision"])

    def _restore_counts(self, raw: Mapping[str, Any]) -> Counter[str]:
        counts = Counter({label: 0 for label in self.option_labels})
        supplied = {str(key): int(value) for key, value in dict(raw).items()}
        if set(supplied) != set(self.option_labels):
            raise ValueError(
                "structured specimen exploration option counts do not match"
            )
        if any(value < 0 for value in supplied.values()):
            raise ValueError(
                "structured specimen exploration counts cannot be negative"
            )
        counts.update(supplied)
        return counts

    def _action_array(self, value: np.ndarray, name: str) -> np.ndarray:
        action = np.asarray(value, dtype=np.float32).reshape(-1)
        if action.shape != (self.action_dim,):
            raise ValueError(
                f"{name} must have shape {(self.action_dim,)}, got "
                f"{action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError(f"{name} must be finite")
        return action

    @staticmethod
    def _option_label(option: ResidualOptionSpec) -> str:
        if option.is_anchor:
            return "mdl2"
        sign = "+" if option.sign > 0.0 else "-"
        return f"{option.group}:{sign}{option.epsilon:.2f}"

    @staticmethod
    def _empty_last_decision() -> dict[str, Any]:
        return {
            "selected": False,
            "option_index": -1,
            "option_label": "",
            "is_anchor": False,
            "recorded": False,
            "behaviorally_distinct": False,
            "specimen_linf_delta": 0.0,
        }
