"""Bind saved supplement results to the manuscript without scientific calls."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

BASE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
PAPER = BASE / "paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks"
RUN = BASE / "results/capacity_fixed_reference_20261008"


def record(path):
    data = path.read_bytes()
    return {"path": str(path.relative_to(BASE)), "bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest()}


def build():
    source = RUN / "payload/comparison.json"
    result = json.loads(source.read_text())
    terminal = json.loads((RUN / "terminal.json").read_text())
    if not result["complete"] or terminal["status"] != "completed":
        raise ValueError("Only the completed saved result is publishable")
    keys = ["format", "table", "pairs", "pair_harm", "paired_worlds",
            "new_trajectories", "reused_trajectories", "new_raw_rows",
            "posthoc_supplement", "new_independent_validation", "bootstrap_resamples",
            "bootstrap_seed", "prior_primary_preserved", "prior_pair_readouts_preserved",
            "supplementary_primary_reference", "prior_primary_reference",
            "graph_attribution", "rl_attribution", "synthetic_only", "clinical_safety",
            "compute_definition", "delay_definition", "elapsed_definition"]
    evidence = {key: result[key] for key in keys}
    evidence.update(
        source=record(source), terminal=terminal,
        protocol=record(BASE / "specs/2026-10-08-mdl2-fixed-reference/protocol.md"),
        frozen_packet=record(BASE / "specs/2026-10-08-mdl2-fixed-reference/frozen.json"),
        raw_records_omitted_from_git=True, saved_data_only=True,
        scientific_calls=0, human_author_review="not recorded by this script",
    )
    with (OUT / "evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def check():
    evidence = json.loads((OUT / "evidence.json").read_text())
    text = (PAPER / "main.tex").read_text()
    section = text.split(r"\label{sec:mdl2-fixed-supplement}", 1)[1].split(
        r"\section{Discussion and Conclusions}", 1)[0]
    names = {"mdl2-fixed2": "MDL-2 fixed support", "plain_h8": "H8-MPC",
             "value_td-graph-final": "Value-TD graph-final", "sac-graph-final": "SAC graph-final"}
    pairs = {(p["candidate"], p["reference"]): p for p in evidence["pairs"]}
    checked_rows = []
    for row in evidence["table"]:
        saving = "---"
        if row["role"] != "mdl2-fixed2":
            value = pairs[(row["role"], "mdl2-fixed2")]["saving_fraction"] * 100
            saving = f"{value:.3f}" if value >= 0 else f"${value:.3f}$"
        rendered = (f"{names[row['role']]} & {row['cost']/1e6:.3f} & {saving} & "
                    f"{row['lost']:.3f} & {row['applied_hours']:.3f}")
        if rendered not in section:
            raise ValueError("Overall manuscript row mismatch: " + rendered)
        checked_rows.append(rendered)
    conditions = ["No change", "Persistent change", "Fast fluctuation"]
    for row in pairs[("value_td-graph-final", "mdl2-fixed2")]["conditions"]:
        low, high = row["cost_ci95"]
        plow, phigh = row["patient_ci95"]
        rendered = (f"{conditions[row['condition']]} & {row['saving_fraction']*100:.3f} & "
                    f"${row['cost_saving']/1e6:.3f}\\ [{low/1e6:.3f},{high/1e6:.3f}]$ & "
                    f"${row['extra_patient_losses']:.3f}\\ [{plow:.3f},{phigh:.3f}]$")
        if rendered not in section:
            raise ValueError("Condition manuscript row mismatch: " + rendered)
        checked_rows.append(rendered)
    for pair in evidence["pairs"][:3]:
        for value in pair["cost_ci95"]:
            if f"{value/1e6:.3f}" not in section:
                raise ValueError("Missing overall cost interval endpoint")
        for value in pair["patient_ci95"]:
            if f"{value:.3f}" not in section:
                raise ValueError("Missing overall loss interval endpoint")
    labels = re.findall(r"\\label\{([^}]+)\}", text)
    duplicates = [key for key, count in Counter(labels).items() if count > 1]
    refs = re.findall(r"\\(?:ref|eqref)\{([^}]+)\}", text)
    missing_labels = sorted(set(refs) - set(labels))
    cites = set()
    for group in re.findall(r"\\cite\w*\*?(?:\[[^]]*\])*\{([^}]+)\}", text):
        cites.update(key.strip() for key in group.split(","))
    bibkeys = set(re.findall(r"@\w+\s*\{\s*([^,]+),", (PAPER / "references.bib").read_text()))
    missing_cites = sorted(cites - bibkeys)
    stack = []
    for kind, env in re.findall(r"\\(begin|end)\{([^}]+)\}", text):
        if kind == "begin":
            stack.append(env)
        elif not stack or stack.pop() != env:
            raise ValueError("Mismatched LaTeX environment: " + env)
    if duplicates or missing_labels or missing_cites or stack:
        raise ValueError(str((duplicates, missing_labels, missing_cites, stack)))
    return dict(saved_data_only=True, scientific_calls=0, checked_table_rows=len(checked_rows),
                checked_interval_endpoints=12, missing_labels=missing_labels,
                duplicate_labels=duplicates, missing_citations=missing_cites,
                citation_keys=len(cites), latex_environments_balanced=True,
                human_author_review="not performed", manuscript=record(PAPER / "main.tex"),
                evidence=record(OUT / "evidence.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    if args.build:
        build()
    validation = check()
    if args.build:
        with (OUT / "validation.json").open("x", encoding="utf-8") as stream:
            json.dump(validation, stream, indent=2, sort_keys=True)
            stream.write("\n")
    print(json.dumps(validation, sort_keys=True))
