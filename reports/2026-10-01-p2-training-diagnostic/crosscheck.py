"""Independent scalar-sum verification of saved P2 weights and training costs."""

from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path

from src.rl.candidate_patient_session import load_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.utils.research_archive import sha256_file


ROOT = Path.cwd()
RUN = ROOT / "results/candidate_reference_prior_pilot_20261001"
REPORT = ROOT / "reports/2026-10-01-p2-training-diagnostic/result.json"
OUTPUT = REPORT.with_name("crosscheck.json")


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    report = json.loads(REPORT.read_text())
    for relative, expected in report["input_files"].items():
        assert sha256_file(RUN / relative) == expected, relative
    results, all_components = {}, Counter()
    for key, result in report["models"].items():
        model = RUN / "payload/models" / key
        final = load_envelope(model / "final.pt")
        w = final["policy"]["score_head.2.weight"].tolist()
        assert len(w) == 1 and len(w[0]) == 32 and all(math.isfinite(x) for x in w[0])
        span = 2 * math.fsum(abs(x) for x in w[0])
        assert span < math.log(9)
        assert math.isclose(span, result["final_score_bound"]["residual_pairwise_span_upper_bound"], abs_tol=1e-12)
        vw = final["policy"]["value_head.2.weight"].tolist()[0]
        bias = final["policy"]["value_head.2.bias"].item()
        radius = math.fsum(abs(x) for x in vw)
        lo, hi = bias-radius, bias+radius
        assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip((lo, hi), result["final_value_target_range"]["output_range"]))
        train = json.loads((model / "training.json").read_text())
        returns, components, classes, values = [], Counter(), Counter(), []
        for entry in train["raw_episodes"]:
            rows = [json.loads(s)["event"] for s in (RUN / entry["events"]["path"]).read_text().splitlines()]
            costs = [e["info"]["cost"] for e in rows]
            returns.extend(-math.fsum(costs[i:])*1e-9 for i in range(len(costs)))
            for e in rows:
                components.update(dict(e["audit"]["cost_components"]))
                ev = e["audit"]["decision"]["evaluation"]
                classes[len(ev["log_probs"])] += 1
                values.append(ev["value"])
        assert {str(k): v for k, v in classes.items()} == result["class_count_histogram"]
        assert all(math.isclose(v, result["cost_components"][k], rel_tol=1e-12) for k, v in components.items())
        outside = sum(x < lo-1e-5 or x > hi+1e-5 for x in returns)
        mse_floor = math.fsum(max(lo-x, x-hi, 0)**2 for x in returns)/len(returns)
        assert outside == result["final_value_target_range"]["outside_count"]
        assert math.isclose(mse_floor, result["final_value_target_range"]["mse_lower_bound_from_output_range"], abs_tol=1e-12)
        batches = [m for u in train["updates"] for m in u["minibatches"]]
        assert len(batches) == 128 and all(m["clip_fraction"] == 0 for m in batches)
        assert sum(m["grad_norm_before_clip"] > .5 for m in batches) == result["gradient_clipped_batches"]
        all_components.update(components)
        results[key] = {"scalar_fsum_score_bound": span, "minimum_prior_remaining_gap": math.log(9)-span,
                        "scalar_fsum_value_range": [lo, hi], "targets_outside": outside,
                        "mse_floor": mse_floor, "recorded_behavior_value_range": [min(values), max(values)],
                        "cost_sum": math.fsum(components.values())}
    rollouts = [r for m in report["models"].values() for r in m["rollouts"]]
    total = math.fsum(all_components.values())
    result = {
        "format": "p2-training-diagnostic-independent-scalar-readback-v1",
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "result_sha256": sha256_file(REPORT), "crosscheck_source_sha256": sha256_file(Path(__file__)),
        "input_files_rehashed": len(report["input_files"]), "models": results,
        "aggregate": {
            "targets_outside_final_value_ranges": sum(x["targets_outside"] for x in results.values()),
            "training_targets": report["coverage"]["training_steps"],
            "ratio_clipped_minibatches": 0, "total_minibatches": 1152,
            "gradient_clipped_minibatches": sum(m["gradient_clipped_batches"] for m in report["models"].values()),
            "time_position_variation_share_range": [min(r["time_position_advantage_variation_share"] for r in rollouts), max(r["time_position_advantage_variation_share"] for r in rollouts)],
            "behavior_value_explained_variance_range": [min(r["value_explained_variance"] for r in rollouts), max(r["value_explained_variance"] for r in rollouts)],
            "positive_mean_nonreference_advantage_rollouts": sum(r["mean_normalized_advantage_nonreference"] > 0 for r in rollouts),
            "positive_time_centered_nonreference_advantage_rollouts": sum(r["time_centered_advantage_nonreference"] > 0 for r in rollouts),
            "positive_reference_logit_signal_rollouts": sum(r["reference_logit_combined_signal_mean"] > 0 for r in rollouts),
            "rollouts": len(rollouts), "raw_cost_component_shares": {k: v/total for k, v in all_components.items()},
            "return_reconstruction_max_error": max(r["return_reconstruction_max_error"] for r in rollouts),
            "first_minibatch_max_error": max(max(r["first_minibatch_errors"].values()) for r in rollouts),
        },
        "new_patient_calls": 0, "new_neural_forwards": 0, "new_neural_backwards": 0,
        "new_optimizer_steps": 0, "new_test_evaluations": 0,
        "method": "scalar math.fsum; no import from the first diagnostic; same saved inputs, not independent scientific data",
    }
    write_json_once(OUTPUT, result)
    print(json.dumps(result["aggregate"], indent=2))


if __name__ == "__main__":
    main()
