"""Export saved comparison fields and draw manuscript figures; no science runs.

The published data are sufficient to rebuild every figure without local result
roots. --export is only for the local coordinator with the preserved sources.
No world resampling, significance testing, model imports or simulation occurs.
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/gcn-manuscript-mpl")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parents[1]
DATA = PROJECT / "data"
FIGURES = PROJECT / "figures"
SOURCES = {
    "primary": "results/native_graph_self_replication_20261017_eval_recovery1/payload/comparison.json",
    "messages": "results/native_gcn_bounded_communication_20261028/payload/comparison.json",
    "procurement": "specs/2026-10-30-gcn-procurement-preservation/reexecution1/terminal-readout.json",
    "latest_conditional": "specs/2026-10-30-gcn-conditional-recipient-allocation/assessment-closeout1/terminal-readout.json",
    "joint_inventory": "reports/2026-10-31-joint-request-reuse-inventory/decision.json",
}
SUPPLEMENTS = {
    "historical-graph-self.md": "reports/2026-10-17-graph-self-evidence/README.md",
    "topology-transfer.md": "specs/2026-10-21-gcn-topology-transfer/terminal-readout.md",
    "multitopology-training.md": "specs/2026-10-21-gcn-multitopology-training/terminal-readout.md",
    "size-transfer.md": "specs/2026-10-21-gcn-size-transfer/terminal-readout.md",
    "sample-efficiency.md": "specs/2026-10-21-gcn-sample-efficiency/terminal-readout.md",
    "stale-advice.json": "specs/2026-10-27-gcn-delayed-advice/terminal-readout.json",
    "common-joint-action.md": "specs/2026-10-30-gcn-common-joint-action/recovery1/terminal-readout.md",
    "procurement-archive-only.json": "specs/2026-10-30-gcn-procurement-preservation/archive-completion1/terminal-readout.json",
}
COLORS = {"operational": "#16756c", "learning": "#bb5b19", "graph": "#486ca5"}


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n")


def export():
    values, provenance = {}, []
    for name, relative in SOURCES.items():
        raw = (ROOT / relative).read_bytes()
        values[name] = json.loads(raw)
        provenance.append(dict(id=name, local_source=relative, bytes=len(raw),
                               sha256=hashlib.sha256(raw).hexdigest()))
    p = values["primary"]
    assert p["complete"] and p["paired_worlds"] == 120 and p["training_blocks"] == 5
    assert p["development_only"] and p["synthetic_only"] and len(p["pairs"]) == 6
    primary = {k: p[k] for k in ("complete", "paired_worlds", "training_blocks", "pairs", "table",
        "condition_table", "claims", "interval_method", "multiplicity_adjusted", "development_only",
        "synthetic_only", "training_accounting", "all_world_condition_block_harms_retained")}
    m = values["messages"]
    assert m["paired_worlds"] == 24 and len(m["pairs"]) == 3
    messages = {k: m[k] for k in ("label", "paired_worlds", "evaluations", "pairs", "primary", "decision",
                                  "interval_method", "clinical_safety_claim", "mature_graph_improvement")}
    messages = json.loads(json.dumps(messages))
    for contrast in messages["pairs"]:
        for scope in ("worlds", "blocks", "conditions"):
            for record in contrast[scope]:
                record.pop("extra_metrics", None)
    procurement = values["procurement"]["formal_result_excerpts"]
    DATA.mkdir(exist_ok=True)
    dump(DATA / "primary-comparison.json", primary)
    dump(DATA / "bounded-message-comparison.json", messages)
    dump(DATA / "procurement-excerpts.json", procurement)
    dump(DATA / "latest-conditional-readout.json", values["latest_conditional"])
    dump(DATA / "joint-inventory-decision.json", values["joint_inventory"])
    supplement_dir = DATA / "supplements"
    supplement_dir.mkdir(exist_ok=True)
    for name, relative in SUPPLEMENTS.items():
        raw = (ROOT / relative).read_bytes()
        (supplement_dir / name).write_bytes(raw)
        provenance.append(dict(id=name, local_source=relative, bytes=len(raw),
                               sha256=hashlib.sha256(raw).hexdigest()))
    dump(DATA / "source-manifest.json", dict(schema="manuscript-saved-comparison-export-v1",
        sources=provenance, transformations="Select saved fields; omit repeated message per-world/block/condition extra_metrics but retain aggregate components and every outcome/harm record. Fractions multiplied by100 only for display. No bootstrap, formal comparison or new simulation.",
        evidence_scope="DEVELOPMENT, synthetic-only; human scientific review pending",
        raw_simulation_records_models_and_private_messages_included=False))
    with (DATA / "primary-contrasts.csv").open("w", newline="") as f:
        fields = ["candidate", "reference", "saving_percent", "saving_lower_percent", "saving_upper_percent",
                  "extra_losses", "loss_lower", "loss_upper", "extra_completions", "patient_harmed_worlds"]
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in p["pairs"]:
            writer.writerow(dict(candidate=row["candidate"], reference=row["reference"],
                saving_percent=100*row["saving_fraction"], saving_lower_percent=100*row["saving_fraction_ci95"][0],
                saving_upper_percent=100*row["saving_fraction_ci95"][1], extra_losses=row["extra_patient_losses"],
                loss_lower=row["extra_patient_losses_ci95"][0], loss_upper=row["extra_patient_losses_ci95"][1],
                extra_completions=row["extra_completions"], patient_harmed_worlds=row["patient_harmed_worlds"]))


def pair(rows, candidate, reference):
    return next(x for x in rows if x["candidate"] == candidate and x["reference"] == reference)


def save(fig, name):
    fig.savefig(FIGURES / (name+".pdf"), bbox_inches="tight", pad_inches=.06)
    fig.savefig(FIGURES / (name+".png"), dpi=210, bbox_inches="tight", pad_inches=.06)
    plt.close(fig)


def figures():
    FIGURES.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "axes.axisbelow": True})
    p = json.loads((DATA / "primary-comparison.json").read_text())
    m = json.loads((DATA / "bounded-message-comparison.json").read_text())
    rows = [pair(p["pairs"], "local_graph", ref) for ref in ("mdl2", "structured_h8", "local_self")]
    labels = ["Full method / MDL2", "Learning / H8", "Graph / matched Self"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), gridspec_kw={"width_ratios": [1, 1.05]})
    for y, (row, kind) in enumerate(zip(rows, COLORS)):
        color = COLORS[kind]
        x = 100*row["saving_fraction"]
        lo, hi = [100*z for z in row["saving_fraction_ci95"]]
        axes[0].errorbar(x, y, xerr=[[x-lo], [hi-x]], fmt="o", color=color, capsize=3, lw=1.4)
        loss = row["extra_patient_losses"]
        lo, hi = row["extra_patient_losses_ci95"]
        axes[1].errorbar(loss, y, xerr=[[loss-lo], [hi-loss]], fmt="o", color=color, capsize=3, lw=1.4)
    for ax in axes:
        ax.axvline(0, c="#555555", lw=.8, ls="--")
        ax.set_ylim(2.6, -.6)
        ax.set_yticks(range(3))
        ax.grid(axis="x", color="#dddddd", lw=.6)
    axes[0].set_yticklabels(labels)
    axes[1].set_yticklabels([])
    axes[0].set_xlabel("Cost saving (%)\nPositive favors candidate")
    axes[1].set_xlabel("Extra patient losses / world\nNegative favors candidate")
    axes[0].set_title("A  Cost attribution", loc="left")
    axes[1].set_title("B  Patient outcomes", loc="left")
    fig.tight_layout(w_pad=1.5)
    save(fig, "review-primary-attribution")

    fig, ax = plt.subplots(figsize=(7.0, 2.8))
    for ref, label, marker, offset, color in (
        ("structured_h8", "Graph/H8", "o", -.16, COLORS["learning"]),
        (None, "Self/H8", "s", 0, "#16756c"),
        ("local_self", "Graph/Self", "^", .16, COLORS["graph"])):
        row = pair(p["pairs"], "local_self" if ref is None else "local_graph", ref or "structured_h8")
        ax.scatter([x["block"]+offset for x in row["blocks"]], [100*x["saving_fraction"] for x in row["blocks"]],
                   label=label, marker=marker, s=35, color=color)
    ax.axhline(0, color="#555555", lw=.8, ls="--")
    ax.set_xticks(range(5), [f"Block {i}" for i in range(5)])
    ax.set_ylabel("Cost saving vs named reference (%)")
    ax.grid(axis="y", color="#dddddd", lw=.6)
    ax.legend(frameon=False, ncols=3, loc="upper right")
    ax.set_ylim(-.3, 2.8)
    fig.tight_layout()
    save(fig, "review-five-blocks")

    row = pair(m["pairs"], "message_gcn", "message_mlp")
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.7))
    for ax, field, label, color in (
        (axes[0], "cost_gain_percent", "Cost saving (%)", "#16756c"),
        (axes[1], "extra_patient_losses", "Extra patient losses / world", "#b64040")):
        ax.bar([x["block"] for x in row["blocks"]], [x[field] for x in row["blocks"]], width=.55, color=color)
        ax.set_xticks(range(3), [f"Block {i}" for i in range(3)])
        ax.set_ylabel(label)
        ax.axhline(0, color="#555555", lw=.8)
        ax.grid(axis="y", color="#dddddd", lw=.6)
    axes[0].set_title("A  Matched messages: cost advantage", loc="left")
    axes[1].set_title("B  The same policies: service harm", loc="left")
    fig.tight_layout(w_pad=2)
    save(fig, "review-message-tradeoff")

    components = [x for x in row["extra_metrics"] if len(x["path"]) == 2 and x["path"][0] == "components"]
    assert components and all("saving" in x for x in components)
    fig, ax = plt.subplots(figsize=(7.0, 3.3))
    labels = [x["path"][1].replace("_cost", "").replace("_", " ") for x in components]
    values = [x["saving"]/1e6 for x in components]
    ax.barh(range(len(values)), values, color=["#16756c" if x >= 0 else "#b64040" for x in values])
    ax.set_yticks(range(len(values)), labels)
    ax.invert_yaxis()
    ax.axvline(0, color="#555555", lw=.8)
    ax.set_xlabel("Saved cost component (million simulated units / world)")
    ax.grid(axis="x", color="#dddddd", lw=.6)
    fig.tight_layout()
    save(fig, "review-message-cost-components")

    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    boxes = {
        "state": (.1, 3.3, 1.65, .9, "Public state\n+ native\nrequests"),
        "h8": (2.05, 3.3, 1.65, .9, "Shared\ncandidates\n+ H8 costs"),
        "graph": (4.1, 3.85, 1.9, .8, "Graph: adjacency\nmixing before\nReLU"),
        "self": (4.1, 2.55, 1.9, .8, "Self: identity\nsame inputs\nand readout"),
        "score": (6.4, 3.3, 1.55, .9, "Paired cost\ncorrection"),
        "action": (8.3, 3.3, 1.6, .9, "Rerank\nand execute\nfirst action"),
        "tails": (2.75, .55, 4.9, .9, "Training: paired full-tail costs to step 52\nRound-specific frozen continuation"),
    }
    for key, (x, y, w, h, label) in boxes.items():
        ax.add_patch(Rectangle((x, y), w, h, fill=False, ec="#555555", lw=.9))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=8.3)
    for start, end in [((1.75, 3.75), (2.05, 3.75)), ((3.7, 3.95), (4.1, 4.25)),
                       ((3.7, 3.55), (4.1, 2.95)), ((6, 4.25), (6.4, 3.95)),
                       ((6, 2.95), (6.4, 3.55)), ((7.95, 3.75), (8.3, 3.75)),
                       ((5.2, 1.45), (5.2, 2.5))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=9, color="#555555", lw=.9))
    ax.text(5.2, 1.85, "Same target construction; on-policy data may differ", ha="center", fontsize=8.3)
    ax.text(5.2, 4.9, "Primary interface: only neural adjacency differs", ha="center", fontsize=10)
    ax.text(5.2, .1, "The separate bounded-message study has no global H8; it is not this diagram with one box removed.",
            ha="center", fontsize=7.5)
    save(fig, "review-method-interface")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--export", action="store_true", help="Copy selected saved fields; no formal comparison recomputation")
    args = parser.parse_args()
    if args.export:
        export()
    figures()
