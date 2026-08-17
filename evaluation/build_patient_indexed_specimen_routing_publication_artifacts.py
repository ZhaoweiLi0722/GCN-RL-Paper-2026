"""Build publication tables and figures from frozen routing evidence.

The default path reads only compact, committed evidence.  Passing
``--raw-formal-final-root`` additionally rebuilds the cost-component summary
from the immutable row-level formal evaluation outputs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable


GCN_DDPG = "gcn_residual_mdl2_network_ddpg_afd"
FLAT_DDPG = "flat_residual_mdl2_network_ddpg_afd"
GCN_TD3 = "gcn_residual_mdl2_network_td3_bc"
FLAT_TD3 = "flat_residual_mdl2_network_td3_bc"

COMPARISON_LABELS = {
    f"{GCN_DDPG}_vs_anchor": "AFR-GCN-DDPG vs MDL-2",
    f"{FLAT_DDPG}_vs_anchor": "AFR-Flat-DDPG vs MDL-2",
    f"{GCN_DDPG}_vs_{FLAT_DDPG}": "AFR-GCN-DDPG vs AFR-Flat-DDPG",
    f"{GCN_TD3}_vs_anchor": "AFR-GCN-TD3 vs MDL-2",
    f"{FLAT_TD3}_vs_anchor": "AFR-Flat-TD3 vs MDL-2",
    f"{GCN_TD3}_vs_{FLAT_TD3}": "AFR-GCN-TD3 vs AFR-Flat-TD3",
}

TOP_LEVEL_COST_COMPONENTS = (
    "base_cost",
    "patient_loss_cost",
    "expiry_cost",
    "urgency_cost",
)
OPERATING_COST_COMPONENTS = (
    "reagent_purchase_cost",
    "reagent_holding_cost",
    "reagent_shortage_cost",
    "bioreactor_holding_cost",
    "bioreactor_shortage_cost",
    "specimen_transfer_cost",
    "capacity_transfer_cost",
    "reagent_transfer_cost",
)
PDF_METADATA = {
    "Creator": "build_patient_indexed_specimen_routing_publication_artifacts.py",
    "CreationDate": None,
    "ModDate": None,
}


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative_ci(metric: dict[str, Any]) -> tuple[float, float]:
    baseline = float(metric["baseline_mean"])
    return (
        100.0 * float(metric["ci_low"]) / baseline,
        100.0 * float(metric["ci_high"]) / baseline,
    )


def metric_row(
    *,
    scope: str,
    experiment: str,
    comparison_key: str,
    metric: dict[str, Any],
    source: str,
) -> dict[str, Any]:
    ci_low_pct, ci_high_pct = relative_ci(metric)
    return {
        "scope": scope,
        "experiment": experiment,
        "comparison": COMPARISON_LABELS[comparison_key],
        "baseline_mean": metric["baseline_mean"],
        "candidate_mean": metric["candidate_mean"],
        "cost_difference": metric["mean_difference"],
        "relative_difference_pct": metric["mean_gap_pct"],
        "ci_low": metric["ci_low"],
        "ci_high": metric["ci_high"],
        "ci_low_pct": ci_low_pct,
        "ci_high_pct": ci_high_pct,
        "n_pairs": metric["n_pairs"],
        "n_training_seeds": metric["n_training_seeds"],
        "favorable_cost_interval": float(metric["ci_high"]) < 0.0,
        "source": source,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty table: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def derive_cost_component_summary(
    raw_formal_final_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Aggregate additive objective components from formal GCN row files."""

    csv.field_size_limit(sys.maxsize)
    algorithm_root = raw_formal_final_root / GCN_DDPG
    kinds = {
        "learned": "holdout_rows.csv",
        "anchor": "holdout_anchor_rows.csv",
    }
    all_components = (*TOP_LEVEL_COST_COMPONENTS, *OPERATING_COST_COMPONENTS, "total_cost")
    result: dict[str, dict[str, float]] = {}
    source_files: list[dict[str, Any]] = []
    audit: dict[str, Any] = {}

    for role, filename in kinds.items():
        paths = sorted(algorithm_root.glob(f"seed*/{filename}"))
        if len(paths) != 5:
            raise ValueError(f"expected five {role} files, found {len(paths)}")
        sums = {name: 0.0 for name in all_components}
        count = 0
        keys: set[tuple[int, str, int]] = set()
        scenarios: set[str] = set()
        training_seeds: set[int] = set()
        max_total_reconciliation_error = 0.0
        max_base_reconciliation_error = 0.0

        for path in paths:
            file_rows = 0
            with path.open(newline="", encoding="utf-8") as handle:
                for row in csv.DictReader(handle):
                    values = {name: float(row[name]) for name in all_components}
                    for name, value in values.items():
                        if not math.isfinite(value):
                            raise ValueError(f"non-finite {name} in {path}")
                        sums[name] += value
                    training_seed = int(row["training_seed"])
                    replication = int(row["replication"])
                    scenario = row["scenario"]
                    key = (training_seed, scenario, replication)
                    if key in keys:
                        raise ValueError(f"duplicate CRN key {key} in {role}")
                    keys.add(key)
                    scenarios.add(scenario)
                    training_seeds.add(training_seed)
                    total_parts = sum(values[name] for name in TOP_LEVEL_COST_COMPONENTS)
                    base_parts = sum(values[name] for name in OPERATING_COST_COMPONENTS)
                    max_total_reconciliation_error = max(
                        max_total_reconciliation_error,
                        abs(values["total_cost"] - total_parts),
                    )
                    max_base_reconciliation_error = max(
                        max_base_reconciliation_error,
                        abs(values["base_cost"] - base_parts),
                    )
                    count += 1
                    file_rows += 1
            source_files.append(
                {
                    "role": role,
                    "path": str(path.relative_to(raw_formal_final_root)),
                    "rows": file_rows,
                    "sha256": sha256_file(path),
                }
            )

        if count != 2000 or len(keys) != 2000:
            raise ValueError(f"expected 2,000 unique {role} rows, found {count}")
        if len(scenarios) != 4 or training_seeds != {10, 11, 12, 13, 14}:
            raise ValueError(f"unexpected {role} protocol: {scenarios}, {training_seeds}")
        if max_total_reconciliation_error > 1e-3 or max_base_reconciliation_error > 1e-3:
            raise ValueError("objective component reconciliation failed")

        result[role] = {name: sums[name] / count for name in all_components}
        audit[role] = {
            "rows": count,
            "unique_crn_keys": len(keys),
            "scenarios": sorted(scenarios),
            "training_seeds": sorted(training_seeds),
            "max_total_reconciliation_error": max_total_reconciliation_error,
            "max_base_reconciliation_error": max_base_reconciliation_error,
        }

    anchor_total = result["anchor"]["total_cost"]
    components: dict[str, Any] = {}
    for name in (*TOP_LEVEL_COST_COMPONENTS, *OPERATING_COST_COMPONENTS):
        parent = None if name in TOP_LEVEL_COST_COMPONENTS else "base_cost"
        components[name] = {
            "additive_to_total": name in TOP_LEVEL_COST_COMPONENTS,
            "parent_component": parent,
            "learned_mean": result["learned"][name],
            "anchor_mean": result["anchor"][name],
            "difference": result["learned"][name] - result["anchor"][name],
            "anchor_share_of_total_pct": 100.0 * result["anchor"][name] / anchor_total,
        }

    payload = {
        "schema_version": 1,
        "evidence_scope": "formal_holdout",
        "algorithm": GCN_DDPG,
        "comparator": "routing_mdl2_anchor",
        "formal_holdout_seed": 91100000,
        "components": components,
        "total_cost": {
            "learned_mean": result["learned"]["total_cost"],
            "anchor_mean": anchor_total,
            "difference": result["learned"]["total_cost"] - anchor_total,
        },
        "audit": audit,
        "source_files": source_files,
        "reporting_note": (
            "base_cost is additive to total_cost. Its operating subcomponents, "
            "including specimen_transfer_cost, must not be added again."
        ),
    }
    write_json(output_path, payload)
    return payload


def build_primary_rows(evidence_root: Path) -> list[dict[str, Any]]:
    path = evidence_root / "patient_indexed_specimen_routing_primary_ddpg/formal/final_summary.json"
    summary = read_json(path)
    keys = (
        f"{GCN_DDPG}_vs_anchor",
        f"{FLAT_DDPG}_vs_anchor",
        f"{GCN_DDPG}_vs_{FLAT_DDPG}",
    )
    return [
        metric_row(
            scope="formal_holdout",
            experiment="routing_primary_ddpg_final",
            comparison_key=key,
            metric=summary["aggregate"][key]["total_cost"],
            source=str(path.relative_to(evidence_root.parent.parent)),
        )
        for key in keys
    ]


def build_sensitivity_rows(evidence_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for experiment, filename in (
        ("specimen_availability_lead_0", "lead0_summary.json"),
        ("finished_product_return_lead_1", "return1_summary.json"),
    ):
        path = evidence_root / f"patient_indexed_specimen_routing_primary_ddpg/sensitivity/{filename}"
        summary = read_json(path)
        for key in (
            f"{GCN_DDPG}_vs_anchor",
            f"{FLAT_DDPG}_vs_anchor",
            f"{GCN_DDPG}_vs_{FLAT_DDPG}",
        ):
            rows.append(
                metric_row(
                    scope="transport_timing_sensitivity",
                    experiment=experiment,
                    comparison_key=key,
                    metric=summary["aggregate"][key]["total_cost"],
                    source=str(path.relative_to(evidence_root.parent.parent)),
                )
            )
    return rows


def build_td3_rows(evidence_root: Path) -> list[dict[str, Any]]:
    path = evidence_root / "patient_indexed_specimen_routing_stage_c_td3/evaluation/final/summary.json"
    summary = read_json(path)
    return [
        metric_row(
            scope="development_backbone_ablation",
            experiment="routing_stage_c_td3_final",
            comparison_key=key,
            metric=summary["aggregate"][key]["total_cost"],
            source=str(path.relative_to(evidence_root.parent.parent)),
        )
        for key in (
            f"{GCN_TD3}_vs_anchor",
            f"{FLAT_TD3}_vs_anchor",
            f"{GCN_TD3}_vs_{FLAT_TD3}",
        )
    ]


def build_online_attribution_rows(evidence_root: Path) -> list[dict[str, Any]]:
    primary_dir = evidence_root / "patient_indexed_specimen_routing_primary_ddpg"
    final = read_json(primary_dir / "formal/final_summary.json")
    pretrain = read_json(primary_dir / "formal/pretrain_summary.json")
    stage_b = read_json(primary_dir / "development/checkpoint_curve_summary.json")
    td3_dir = evidence_root / "patient_indexed_specimen_routing_stage_c_td3/evaluation"
    td3_final = read_json(td3_dir / "final/summary.json")
    td3_pretrain = read_json(td3_dir / "pretrain/summary.json")
    c3_path = (
        evidence_root
        / "patient_indexed_specimen_routing_ddpg_structured_exploration/comparison/structured_exploration_summary.json"
    )
    c3 = read_json(c3_path)

    def checkpoint_mean(summary: dict[str, Any], comparison: str) -> float:
        return float(summary["aggregate"][comparison]["total_cost"]["candidate_mean"])

    gcn_ddpg_key = f"{GCN_DDPG}_vs_anchor"
    gcn_td3_key = f"{GCN_TD3}_vs_anchor"
    formal_pretrain_mean = checkpoint_mean(pretrain, gcn_ddpg_key)
    formal_final_mean = checkpoint_mean(final, gcn_ddpg_key)
    td3_pretrain_mean = checkpoint_mean(td3_pretrain, gcn_td3_key)
    td3_final_mean = checkpoint_mean(td3_final, gcn_td3_key)
    late = stage_b["algorithms"][GCN_DDPG]["segments"]["episode75_to_episode100"]["metrics"]["total_cost"]
    c3_final = c3["roles"]["candidate"][GCN_DDPG]["final_vs_frozen"]["total_cost"]
    c3_control = c3["candidate_vs_control"][GCN_DDPG]["final"]["total_cost"]

    def point_row(
        experiment: str,
        scope: str,
        baseline: float,
        candidate: float,
        source: str,
        source_secondary: str,
        note: str,
    ) -> dict[str, Any]:
        difference = candidate - baseline
        return {
            "scope": scope,
            "experiment": experiment,
            "comparison": "final minus frozen pretraining",
            "baseline_mean": baseline,
            "candidate_mean": candidate,
            "cost_difference": difference,
            "relative_difference_pct": 100.0 * difference / baseline,
            "ci_low": "",
            "ci_high": "",
            "ci_low_pct": "",
            "ci_high_pct": "",
            "n_pairs": "",
            "n_training_seeds": "",
            "supports_online_gain": False,
            "note": note,
            "source": source,
            "source_secondary": source_secondary,
        }

    rows = [
        point_row(
            "routing_primary_ddpg_formal",
            "formal_holdout_checkpoint_point_estimate",
            formal_pretrain_mean,
            formal_final_mean,
            str((primary_dir / "formal/final_summary.json").relative_to(evidence_root.parent.parent)),
            str((primary_dir / "formal/pretrain_summary.json").relative_to(evidence_root.parent.parent)),
            "Compact top-level summaries preserve the point estimate; Stage B provides the paired curve interval.",
        ),
        {
            **metric_row(
                scope="development_checkpoint_curve",
                experiment="routing_primary_ddpg_episode75_to_episode100",
                comparison_key=gcn_ddpg_key,
                metric=late,
                source=str(
                    (primary_dir / "development/checkpoint_curve_summary.json").relative_to(
                        evidence_root.parent.parent
                    )
                ),
            ),
            "supports_online_gain": False,
            "note": "Late DDPG segment; paired interval crosses zero.",
        },
        point_row(
            "routing_stage_c_td3",
            "development_backbone_ablation",
            td3_pretrain_mean,
            td3_final_mean,
            str((td3_dir / "final/summary.json").relative_to(evidence_root.parent.parent)),
            str((td3_dir / "pretrain/summary.json").relative_to(evidence_root.parent.parent)),
            "Final and frozen-pretrain TD3 are statistically indistinguishable in the audited recovery report.",
        ),
        {
            **metric_row(
                scope="development_mechanism_screen",
                experiment="routing_ddpg_structured_exploration_final_vs_frozen",
                comparison_key=gcn_ddpg_key,
                metric=c3_final,
                source=str(c3_path.relative_to(evidence_root.parent.parent)),
            ),
            "comparison": "candidate final minus candidate frozen pretraining",
            "supports_online_gain": False,
            "note": "Two of three seeds favored final, but the effect gate and paired interval failed.",
        },
        {
            **metric_row(
                scope="development_mechanism_screen",
                experiment="routing_ddpg_structured_exploration_candidate_vs_control",
                comparison_key=gcn_ddpg_key,
                metric=c3_control,
                source=str(c3_path.relative_to(evidence_root.parent.parent)),
            ),
            "comparison": "structured-exploration candidate minus unchanged DDPG control",
            "supports_online_gain": False,
            "note": "No-regression check passed; paired interval crosses zero.",
        },
    ]

    # Normalize heterogeneous rows into a stable CSV schema.
    fields = list(rows[0])
    for row in rows[1:]:
        for field in row:
            if field not in fields:
                fields.append(field)
    return [{field: row.get(field, "") for field in fields} for row in rows]


def build_cost_rows(component_summary: dict[str, Any]) -> list[dict[str, Any]]:
    labels = {
        "base_cost": "Base operating cost (net)",
        "patient_loss_cost": "Patient-loss cost",
        "expiry_cost": "Expiry cost",
        "urgency_cost": "Urgency cost",
        "reagent_purchase_cost": "Reagent purchase",
        "reagent_holding_cost": "Reagent holding",
        "reagent_shortage_cost": "Reagent shortage",
        "bioreactor_holding_cost": "Bioreactor holding",
        "bioreactor_shortage_cost": "Bioreactor shortage",
        "specimen_transfer_cost": "Specimen transfer",
        "capacity_transfer_cost": "Capacity transfer",
        "reagent_transfer_cost": "Reagent transfer",
    }
    rows = []
    for name in (*TOP_LEVEL_COST_COMPONENTS, *OPERATING_COST_COMPONENTS):
        values = component_summary["components"][name]
        rows.append(
            {
                "component": name,
                "label": labels[name],
                "additive_to_total": values["additive_to_total"],
                "parent_component": values["parent_component"] or "",
                "mdl2_mean": values["anchor_mean"],
                "gcn_ddpg_mean": values["learned_mean"],
                "gcn_minus_mdl2": values["difference"],
                "mdl2_share_of_total_pct": values["anchor_share_of_total_pct"],
            }
        )
    return rows


def plot_effects(
    primary_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    td3_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    panels = [
        ("Formal holdout: primary DDPG", primary_rows),
        (
            "Transport timing sensitivity",
            [
                row
                for row in sensitivity_rows
                if row["comparison"] == "AFR-GCN-DDPG vs MDL-2"
            ],
        ),
        (
            "Development only: TD3 ablation",
            [
                row
                for row in td3_rows
                if row["comparison"] != "AFR-Flat-TD3 vs MDL-2"
            ],
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.1), constrained_layout=True)
    favorable = "#176B5B"
    unfavorable = "#B44C43"

    for ax, (title, rows) in zip(axes, panels):
        y_positions = list(range(len(rows)))
        values = [float(row["relative_difference_pct"]) for row in rows]
        lows = [float(row["ci_low_pct"]) for row in rows]
        highs = [float(row["ci_high_pct"]) for row in rows]
        labels = []
        for row in rows:
            label = row["comparison"].replace("AFR-", "").replace(" vs ", "\nvs ")
            if row["experiment"] == "specimen_availability_lead_0":
                label = "Specimen lead 0\nGCN-DDPG vs MDL-2"
            elif row["experiment"] == "finished_product_return_lead_1":
                label = "Return lead 1\nGCN-DDPG vs MDL-2"
            labels.append(label)
        for y, value, low, high in zip(y_positions, values, lows, highs):
            color = favorable if value < 0 else unfavorable
            ax.errorbar(
                value,
                y,
                xerr=[[value - low], [high - value]],
                fmt="o",
                color=color,
                ecolor=color,
                capsize=4,
                markersize=6,
                linewidth=1.6,
            )
            ax.annotate(
                f"{value:+.3f}%",
                (value, y),
                xytext=(5 if value <= 0 else -5, -12),
                textcoords="offset points",
                ha="left" if value <= 0 else "right",
                fontsize=8.5,
                color=color,
            )
        ax.axvline(0.0, color="#555555", linewidth=1.0, linestyle="--")
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel("Candidate minus comparator total cost (%)")
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8.5)

    fig.suptitle(
        "Routing-primary paired total-cost effects",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Negative values favor the candidate. Intervals are two-level paired 95% bootstrap intervals.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(output_dir / "routing_primary_cost_effects.png", dpi=300, bbox_inches="tight")
    fig.savefig(
        output_dir / "routing_primary_cost_effects.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def plot_cost_components(
    cost_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    """Plot additive total-cost components separately from base-cost detail."""

    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    by_component = {row["component"]: row for row in cost_rows}
    additive_names = (
        "patient_loss_cost",
        "expiry_cost",
        "base_cost",
        "urgency_cost",
    )
    operating_names = (
        "bioreactor_shortage_cost",
        "reagent_shortage_cost",
        "reagent_purchase_cost",
        "specimen_transfer_cost",
        "reagent_holding_cost",
        "reagent_transfer_cost",
        "capacity_transfer_cost",
        "bioreactor_holding_cost",
    )
    panels = (
        (
            "A. Additive components of total cost",
            [by_component[name] for name in additive_names],
            1_000_000.0,
            "Difference (million objective units)",
            3,
        ),
        (
            "B. Detail already included in base operating cost",
            [by_component[name] for name in operating_names],
            1_000.0,
            "Difference (thousand objective units)",
            1,
        ),
    )
    favorable = "#176B5B"
    unfavorable = "#B44C43"
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.5), constrained_layout=True)

    for ax, (title, rows, scale, xlabel, precision) in zip(axes, panels):
        values = [float(row["gcn_minus_mdl2"]) / scale for row in rows]
        labels = [row["label"] for row in rows]
        colors = [favorable if value < 0 else unfavorable for value in values]
        y_positions = list(range(len(rows)))
        ax.barh(y_positions, values, color=colors, height=0.62)
        ax.axvline(0.0, color="#555555", linewidth=1.0)
        ax.set_yticks(y_positions, labels)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=10.5, fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.7)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0, labelsize=8.5)
        ax.margins(x=0.18)
        for y, value in zip(y_positions, values):
            ax.annotate(
                f"{value:+.{precision}f}",
                (value, y),
                xytext=(4 if value >= 0 else -4, 0),
                textcoords="offset points",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.2,
                color=unfavorable if value >= 0 else favorable,
            )

    fig.suptitle(
        "Formal AFR-GCN-DDPG minus MDL-2 cost decomposition",
        fontsize=13,
        fontweight="bold",
    )
    fig.text(
        0.5,
        -0.025,
        "Negative values are savings. Panel B partitions Panel A's base operating cost and must not be added to Panel A.",
        ha="center",
        fontsize=9,
        color="#444444",
    )
    fig.savefig(
        output_dir / "routing_primary_cost_components.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "routing_primary_cost_components.pdf",
        bbox_inches="tight",
        metadata=PDF_METADATA,
    )
    plt.close(fig)


def build_evidence_map(
    repo_root: Path,
    evidence_root: Path,
    primary_rows: list[dict[str, Any]],
    sensitivity_rows: list[dict[str, Any]],
    td3_rows: list[dict[str, Any]],
    online_rows: list[dict[str, Any]],
    component_summary_path: Path,
) -> dict[str, Any]:
    source_paths = {
        row["source"]
        for rows in (primary_rows, sensitivity_rows, td3_rows, online_rows)
        for row in rows
    }
    source_paths.update(
        row["source_secondary"]
        for row in online_rows
        if row.get("source_secondary")
    )
    source_paths.add(str(component_summary_path.relative_to(repo_root)))
    source_paths = sorted(source_paths)
    sources = [
        {
            "path": path,
            "sha256": sha256_file(repo_root / path),
        }
        for path in source_paths
    ]
    return {
        "schema_version": 1,
        "frozen_on": "2026-08-17",
        "primary_method": "AFR-GCN-DDPG",
        "claims": [
            {
                "id": "anchor_improvement",
                "status": "supported_formal_holdout",
                "claim": "AFR-GCN-DDPG lowers total cost relative to routing MDL-2.",
                "effect_pct": primary_rows[0]["relative_difference_pct"],
                "source": primary_rows[0]["source"],
            },
            {
                "id": "graph_attribution",
                "status": "supported_formal_holdout",
                "claim": "AFR-GCN-DDPG lowers total cost relative to parameter-matched AFR-Flat-DDPG.",
                "effect_pct": primary_rows[2]["relative_difference_pct"],
                "source": primary_rows[2]["source"],
            },
            {
                "id": "online_learning_attribution",
                "status": "not_established",
                "claim": "The evidence does not isolate a favorable incremental effect of online actor-critic updates over frozen pretraining.",
                "sources": sorted({row["source"] for row in online_rows}),
            },
            {
                "id": "transport_timing_robustness",
                "status": "asymmetric_not_general",
                "claim": "The GCN advantage strengthens under return lead 1 but reverses under specimen availability lead 0.",
                "sources": sorted({row["source"] for row in sensitivity_rows}),
            },
            {
                "id": "td3_backbone_ablation",
                "status": "supported_development_only",
                "claim": "Matched TD3 corroborates graph and anchor value on a development stream but does not establish online attribution.",
                "sources": sorted({row["source"] for row in td3_rows}),
            },
        ],
        "prohibited_claims": [
            "Do not claim that online DDPG or TD3 updates independently created the observed gain.",
            "Do not present TD3 or Stage C3 development streams as formal holdout confirmation.",
            "Do not claim blanket transport-timing robustness.",
            "Do not call modeled weighted-objective units dollars without external calibration.",
            "Do not add specimen-transfer cost separately to base_cost; it is a base-cost subcomponent.",
        ],
        "sources": sources,
    }


def build_publication_artifacts(
    repo_root: Path,
    report_dir: Path,
    figure_dir: Path,
    evidence_map_path: Path,
) -> dict[str, Any]:
    evidence_root = repo_root / "experiments/evidence"
    component_summary_path = (
        evidence_root
        / "patient_indexed_specimen_routing_primary_ddpg/formal/cost_component_summary.json"
    )
    component_summary = read_json(component_summary_path)
    primary_rows = build_primary_rows(evidence_root)
    sensitivity_rows = build_sensitivity_rows(evidence_root)
    td3_rows = build_td3_rows(evidence_root)
    online_rows = build_online_attribution_rows(evidence_root)
    cost_rows = build_cost_rows(component_summary)

    write_csv(report_dir / "primary_comparisons.csv", primary_rows)
    write_csv(report_dir / "transport_timing_sensitivity.csv", sensitivity_rows)
    write_csv(report_dir / "td3_development_ablation.csv", td3_rows)
    write_csv(report_dir / "online_attribution.csv", online_rows)
    write_csv(report_dir / "cost_components.csv", cost_rows)
    plot_effects(primary_rows, sensitivity_rows, td3_rows, figure_dir)
    plot_cost_components(cost_rows, figure_dir)

    evidence_map = build_evidence_map(
        repo_root,
        evidence_root,
        primary_rows,
        sensitivity_rows,
        td3_rows,
        online_rows,
        component_summary_path,
    )
    write_json(evidence_map_path, evidence_map)
    return {
        "primary_rows": primary_rows,
        "sensitivity_rows": sensitivity_rows,
        "td3_rows": td3_rows,
        "online_rows": online_rows,
        "cost_rows": cost_rows,
        "evidence_map": evidence_map,
    }


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    parser.add_argument(
        "--raw-formal-final-root",
        type=Path,
        help="Optional immutable formal_final root used to refresh component evidence.",
    )
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    repo_root = args.repo_root.resolve()
    evidence_root = repo_root / "experiments/evidence"
    component_path = (
        evidence_root
        / "patient_indexed_specimen_routing_primary_ddpg/formal/cost_component_summary.json"
    )
    if args.raw_formal_final_root:
        derive_cost_component_summary(args.raw_formal_final_root.resolve(), component_path)
    build_publication_artifacts(
        repo_root,
        repo_root / "reports/patient_indexed_specimen_routing",
        repo_root
        / "paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks/figures",
        evidence_root / "patient_indexed_specimen_routing_publication_evidence_map.json",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
