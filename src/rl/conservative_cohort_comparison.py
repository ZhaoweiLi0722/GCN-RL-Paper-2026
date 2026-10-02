"""Independent raw-cost and patient-outcome comparison for the five native roles."""

import math
from pathlib import Path

import numpy as np

from src.rl.candidate_pilot_verification import CAUSES, OPERATING, PATIENT, verify_episode, json_hash, vector
from src.rl.cohort_bundle_verification import _check_index, _read_parts, _tail
from src.rl.conservative_cohort_plan import ROLES, streams as expected_streams, validate_config
from src.rl.dynamic_candidate_verification import _action_difference
from src.rl.paired_cohort_comparison import _action, _tail_action, _turnaround
from src.rl.time_baseline_verification import _finite_report


def verify_bundle(root, index, cfg, inherited, streams):
    validate_config(cfg)
    layouts = {b: rows["layout"][0] for b, rows in streams["environment"].items()}
    if streams != expected_streams(cfg, layouts):
        raise ValueError("prospective test stream schedule differs")
    _check_index(index)
    root, n = Path(root).resolve(), inherited["objective"]["num_facilities"]
    outcomes, files, slots = [], [], {}
    for entry in index:
        (header, rows, prefix), (tail_header, tail_rows, final) = _read_parts(root, entry)
        block, role, world = (header[k] for k in ("block", "role", "world_index"))
        slot = block, role, world
        selection = "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy"
        if (block not in cfg["blocks"] or role not in ROLES or type(world) is not int or not 0 <= world < 12
                or slot in slots or header["split"] != "test" or header["source_id"] != streams["namespace"]
                or type(header["seed"]) is not int or header["seed"] != int(streams["environment"][str(block)]["test"][world])
                or header["selection"] != selection
                or header["representation"] != ("reference" if role in ("r4", "full_mdl2") else "graph")):
            raise ValueError("evaluation native role/seed/selection differs")
        window = verify_episode(header, rows, prefix, inherited)
        actions, waiting = [], 0
        for t, row in enumerate(rows):
            audit, info = row["event"]["audit"], row["event"]["info"]
            evaluation, choice = audit["decision"]["evaluation"], audit["decision"]["choice"]
            request, bank = audit["record"]["action"], evaluation["candidates"]
            if selection == "greedy" and choice["class_index"] != int(np.argmax(evaluation["log_probs"])):
                raise ValueError("final greedy action differs from canonical tie rule")
            if role == "r4" and request != bank["requests"][0]:
                raise ValueError("R4 original request differs")
            if role == "full_mdl2" and not request == bank["requests"][0] == bank["requests"][1]:
                raise ValueError("full MDL-2 anchor differs")
            actions.append(_action(request, info, n, t))
            waiting += sum(vector(info["waiting_patients"], n))
        window.update(cost=math.fsum(row["event"]["info"]["cost"] for row in rows),
            actions=actions, source_id=header["source_id"], waiting_patient_steps=waiting)
        tail = _tail(tail_header, prefix, final, tail_rows, inherited, header, entry)
        tail["actions"] = [_tail_action(row, n, inherited["objective"]["transfer_scale"]) for row in tail_rows]
        terminal = tail["final_compartments"]
        causes = {k: window["loss_causes"][k] + tail["loss_causes"][k] for k in CAUSES}
        cost = math.fsum((window["cost"], tail["tail_cost"]))
        outcome = window | dict(role=role, block=block, world_index=world, seed=header["seed"],
            cost=cost, losses=terminal["lost"], completions=terminal["delivered"], terminal_active=0,
            waiting_patient_steps=waiting + tail["tail_waiting_patient_steps"],
            expiry_losses=sum(causes[k] for k in (CAUSES[2], CAUSES[4], CAUSES[7])),
            loss_causes=causes, prefix_cost=window["cost"], tail_cost=tail["tail_cost"],
            components={k: math.fsum((window["components"][k], tail["components"][k])) for k in OPERATING + PATIENT},
            window=window, tail=tail, actions=actions + tail["actions"],
            policy_sha256=header["policy_sha256"], final_state_sha256=json_hash(final),
            raw_episode_sha256=json_hash([header, rows, prefix, tail_header, tail_rows, final]),
            **_turnaround(final, tail_rows[-1]["info"], terminal))
        slots[slot] = outcome
        outcomes.append(outcome)
        files.extend(record for part in entry.values() for record in part.values())
    expected = {(b, r, w) for b in cfg["blocks"] for r in ROLES for w in range(12)}
    if set(slots) != expected:
        raise ValueError("exact 180 complete evaluation cohorts required")
    for b in cfg["blocks"]:
        for w in range(12):
            if len({slots[b, r, w]["initial_state_sha256"] for r in ROLES}) != 1:
                raise ValueError("unpaired controller starts")
        for r in ROLES:
            if len({slots[b, r, w]["policy_sha256"] for w in range(12)}) != 1:
                raise ValueError("model changed across test worlds")
    metrics = ("cost", "losses", "completions", "waiting_patient_steps", "expiry_losses", "prefix_cost",
               "tail_cost", "average_turnaround_time", "turnaround_sum")
    contrasts = []
    for name in cfg["primary_contrasts"] + cfg["secondary_contrasts"]:
        left, right = name.split("-minus-")
        blocks, paired = [], []
        for b in cfg["blocks"]:
            worlds = []
            for w in range(12):
                l, r = slots[b, left, w], slots[b, right, w]
                worlds.append(dict(world=w, seed=l["seed"], differences={m: l[m] - r[m] for m in metrics},
                    prefix_action_differences=_action_difference(l["window"], r["window"]),
                    component_differences={k: l["components"][k] - r["components"][k] for k in OPERATING + PATIENT},
                    loss_cause_differences={k: l["loss_causes"][k] - r["loss_causes"][k] for k in CAUSES}))
            means = {m: math.fsum(row["differences"][m] for row in worlds) / 12 for m in metrics}
            base = math.fsum(slots[b, right, w]["cost"] for w in range(12)) / 12
            if base <= 0:
                raise ValueError("undefined positive-cost relative comparison")
            blocks.append(dict(block=b, mean_differences=means, relative_cost_change=means["cost"] / base))
            paired.append(dict(block=b, worlds=worlds))
        relative = math.fsum(row["relative_cost_change"] for row in blocks) / 3
        screen = (relative <= -.01 and sum(row["relative_cost_change"] < 0 for row in blocks) >= 2
                  and all(row["mean_differences"]["losses"] <= 0 for row in blocks))
        contrasts.append(dict(name=name, blocks=blocks, paired_worlds=paired, development_screen_met=screen,
            equal_block_relative_cost_change_percent=relative * 100,
            equal_block_mean_differences={m: math.fsum(row["mean_differences"][m] for row in blocks) / 3 for m in metrics}))
    passed = all(row["development_screen_met"] for row in contrasts[:2])
    result = dict(format="conservative-cohort-raw-comparison-v1", files=files, outcomes=outcomes,
        index_sha256=json_hash(index), config_sha256=json_hash(cfg), streams_sha256=json_hash(streams),
        analysis=dict(contrasts=contrasts, independent_training_blocks=3, paired_worlds_per_role=36,
            decision="promising_development_only" if passed else "close_two_round_mechanism",
            primary_development_screen_met=passed, automatic_followon=False,
            inference="descriptive_three_training_blocks_no_confirmatory_significance_claim"))
    _finite_report(result)
    return result
