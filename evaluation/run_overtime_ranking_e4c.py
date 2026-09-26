"""Stage E4c (EXPLORATORY): does a nonlinear learner change the verdict?

Every Stage E4/E4b model was linear ridge. This closes the last cheap
objection -- that the optimum is a nonlinear function of the state that ridge
structurally cannot represent -- by fitting gradient-boosted trees on the same
states, folds, and labels.

With 135 states and roughly 108 per training fold, a boosted ensemble will
overfit readily. That is itself informative: if a high-capacity learner cannot
beat a linear one out of sample, the limit is the information in the state
rather than the shape of the model.

A depth-1 (stump) variant is included as a capacity control: stumps are
additive by construction, so if depth-1 matches depth-3 then interactions are
not what is missing.

EXPLORATORY. Touches no reserved data and trains no deployed policy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from evaluation.audit_overtime_headroom_e2 import (
    load_screen_config,
    sha256_path,
    validate_config,
)
from evaluation.ranking_feasibility import ANCHOR_ARM, standardize
from evaluation.run_overtime_ranking_e4 import load_outcomes
from evaluation.run_overtime_ranking_e4b import collect_all_features, stack


class RegressionTree:
    """Axis-aligned regression tree grown by recursive squared-error splits.

    Real recursion matters: composing stumps additively yields a generalized
    additive model that structurally cannot represent interactions, which is
    precisely the hypothesis this screen exists to test. An earlier additive
    implementation failed the interaction sanity check below and was replaced.
    """

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self) -> None:
        self.feature = -1
        self.threshold = 0.0
        self.left = None
        self.right = None
        self.value = 0.0

    def fit(self, x: np.ndarray, y: np.ndarray, depth: int, min_samples: int = 5):
        self.value = float(y.mean())
        if depth <= 0 or len(y) < 2 * min_samples:
            return self
        best_gain = -np.inf
        best = None
        total = y.sum()
        count = len(y)
        parent = total**2 / count
        for feature in range(x.shape[1]):
            order = np.argsort(x[:, feature], kind="stable")
            values = x[order, feature]
            targets = y[order]
            left_sum = np.cumsum(targets)[:-1]
            left_count = np.arange(1, count)
            right_sum = total - left_sum
            right_count = count - left_count
            valid = (
                (values[:-1] < values[1:])
                & (left_count >= min_samples)
                & (right_count >= min_samples)
            )
            if not valid.any():
                continue
            gain = left_sum**2 / left_count + right_sum**2 / right_count - parent
            gain = np.where(valid, gain, -np.inf)
            index = int(np.argmax(gain))
            if gain[index] > best_gain:
                best_gain = float(gain[index])
                best = (feature, float((values[index] + values[index + 1]) / 2.0))
        if best is None or best_gain <= 0.0:
            return self
        self.feature, self.threshold = best
        mask = x[:, self.feature] <= self.threshold
        self.left = RegressionTree().fit(x[mask], y[mask], depth - 1, min_samples)
        self.right = RegressionTree().fit(x[~mask], y[~mask], depth - 1, min_samples)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.feature < 0 or self.left is None:
            return np.full(x.shape[0], self.value)
        out = np.empty(x.shape[0])
        mask = x[:, self.feature] <= self.threshold
        if mask.any():
            out[mask] = self.left.predict(x[mask])
        if (~mask).any():
            out[~mask] = self.right.predict(x[~mask])
        return out


class BoostedTrees:
    """Gradient boosting on squared error with genuine depth-limited trees."""

    def __init__(self, rounds: int = 60, learning_rate: float = 0.1, depth: int = 3):
        self.rounds = rounds
        self.learning_rate = learning_rate
        self.depth = depth
        self.base = 0.0
        self.trees: list[RegressionTree] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BoostedTrees":
        self.base = float(y.mean())
        prediction = np.full(len(y), self.base)
        self.trees = []
        for _ in range(self.rounds):
            tree = RegressionTree().fit(x, y - prediction, self.depth)
            prediction = prediction + self.learning_rate * tree.predict(x)
            self.trees.append(tree)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        out = np.full(x.shape[0], self.base)
        for tree in self.trees:
            out = out + self.learning_rate * tree.predict(x)
        return out


def evaluate_model(features, seed_of, outcomes, factory) -> dict[str, Any]:
    states = sorted(features)
    ladder = sorted({arm for _, arm in outcomes if arm != ANCHOR_ARM})
    seeds = sorted({seed_of[s] for s in states})
    truth = {s: min(ladder, key=lambda a: outcomes[(s, a)]) for s in states}
    constant_totals = {a: sum(outcomes[(s, a)] for s in states) for a in ladder}
    blind = min(constant_totals, key=constant_totals.get)

    prediction: dict[str, str] = {}
    fold_top1: list[float] = []
    for held_out in seeds:
        train = [s for s in states if seed_of[s] != held_out]
        test = [s for s in states if seed_of[s] == held_out]
        x_train = np.vstack([features[s] for s in train])
        x_test = np.vstack([features[s] for s in test])
        z_train, z_test = standardize(x_train, [x_train, x_test])
        scores = np.zeros((len(test), len(ladder)))
        for index, arm in enumerate(ladder):
            targets = np.array([outcomes[(s, arm)] for s in train])
            centre = targets.mean()
            model = factory().fit(z_train, targets - centre)
            scores[:, index] = model.predict(z_test) + centre
        hits = 0
        for row, state in enumerate(test):
            chosen = ladder[int(np.argmin(scores[row]))]
            prediction[state] = chosen
            hits += int(chosen == truth[state])
        fold_top1.append(hits / len(test))

    anchor = sum(outcomes[(s, ANCHOR_ARM)] for s in states)
    constant_cost = sum(outcomes[(s, blind)] for s in states)
    model_cost = sum(outcomes[(s, prediction[s])] for s in states)
    oracle_cost = sum(outcomes[(s, truth[s])] for s in states)
    return {
        "pooled_top1": float(np.mean([prediction[s] == truth[s] for s in states])),
        "worst_fold_top1": float(min(fold_top1)),
        "fold_top1": [float(v) for v in fold_top1],
        "realized_vs_constant_pct": 100.0 * (model_cost - constant_cost) / anchor,
        "headroom_captured": float(
            (constant_cost - model_cost) / (constant_cost - oracle_cost)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(
        "experiments/configs/continuous_overtime_critic_ranking_e4.json"))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = load_screen_config(config_path)
    validate_config(config)
    outcomes = load_outcomes(Path(config["output_root"]) / "headroom_rows.csv")
    families, seed_of, _ = collect_all_features(config)

    feature_sets = {
        "aggregate": ["aggregate"],
        "aggregate+distribution+graph": ["aggregate", "distribution", "graph"],
    }
    models = {
        "boosted_depth1_r60": lambda: BoostedTrees(rounds=60, learning_rate=0.1, depth=1),
        "boosted_depth3_r60": lambda: BoostedTrees(rounds=60, learning_rate=0.1, depth=3),
        "boosted_depth3_r150": lambda: BoostedTrees(rounds=150, learning_rate=0.05, depth=3),
    }

    results: dict[str, Any] = {}
    print(f"{'features':<30}{'model':<22}{'top-1':>8}{'worst':>8}{'vs const %':>12}{'headroom':>10}")
    for feature_label, names in feature_sets.items():
        features = stack(families, names)
        results[feature_label] = {}
        for model_label, factory in models.items():
            outcome = evaluate_model(features, seed_of, outcomes, factory)
            results[feature_label][model_label] = outcome
            print(
                f"{feature_label:<30}{model_label:<22}"
                f"{outcome['pooled_top1']:>8.3f}{outcome['worst_fold_top1']:>8.3f}"
                f"{outcome['realized_vs_constant_pct']:>+12.4f}"
                f"{100 * outcome['headroom_captured']:>9.1f}%"
            )

    output = Path(config["output_root"]) / "ranking_nonlinear_e4c.json"
    output.write_text(
        json.dumps(
            {
                "name": "continuous_overtime_ranking_nonlinear_e4c",
                "experimental_role": "EXPLORATORY nonlinear-learner check; not confirmatory",
                "gate_top1": float(config["gates"]["ranking_minimum_top1"]),
                "results": results,
                "config_sha256": sha256_path(config_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\nwritten: {output}")


if __name__ == "__main__":
    main()
