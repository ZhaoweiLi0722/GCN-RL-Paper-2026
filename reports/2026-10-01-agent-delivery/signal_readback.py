"""Training-receipt arithmetic only: no project imports, inference or fitting."""

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import uuid


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN = ROOT / "results/candidate_sampled_return_control_20261001"
FIELDS = {"context_indices", "actions", "old_log_probs", "old_values",
          "returns", "advantages", "behavior_logits"}
CONTEXT_COLUMNS = [
    "context", "n", "winner_observations", "expected_winner_observations",
    "p_winner_mean", "p_winner_min", "p_winner_max", "residual_mean",
    "raw_score_mean", "stored_A_score_mean", "oracle_centered_score_mean",
    "baseline_offset_score_mean", "oracle_expected_raw_score_mean",
    "oracle_expected_fixed_scale_score_mean",
]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    require(type(value) in (int, float) and math.isfinite(value),
            "finite scalar number required")
    return value


def vector(values):
    require(isinstance(values, list) and bool(values), "nonempty list required")
    return [finite(value) for value in values]


def integer(value, lower, upper):
    require(type(value) is int and lower <= value < upper, "invalid integer/index")
    return value


def mean(values):
    return math.fsum(values) / len(values) if values else None


def close(actual, expected, tolerance=2e-6):
    require(abs(finite(actual) - finite(expected)) <= tolerance,
            "receipt arithmetic mismatch")


def distribution(logits):
    logits = vector(logits)
    high = max(logits)
    terms = [math.exp(value - high) for value in logits]
    total = math.fsum(terms)
    return [value / total for value in terms], high + math.log(total)


def variance_partition(values, labels):
    """Population moments, weighted by observations, not by group means."""
    values = vector(values)
    require(len(values) == len(labels), "variance label length mismatch")
    groups = defaultdict(list)
    for label, value in zip(labels, values):
        groups[label].append(value)
    center = mean(values)
    total = mean([(value - center) ** 2 for value in values])
    between = math.fsum(len(group) * (mean(group) - center) ** 2
                        for group in groups.values()) / len(values)
    within = math.fsum((value - mean(group)) ** 2
                       for group in groups.values() for value in group) / len(values)
    close(total, between + within, 1e-10 * max(1.0, total))
    return {"n": len(values), "mean": center, "total": total,
            "between_context": between, "within_context": within,
            "between_share": between / total if total else None}


def context_spec(config, manifest):
    cues, times = vector(config["cues"]), vector(config["training_times"])
    requests = vector(config["request_values"])
    require(len(set(cues)) == len(cues) and all(cue != 0 for cue in cues),
            "unique nonzero cues required")
    require(len(set(times)) == len(times) and all(0 <= time <= 1 for time in times),
            "unique unit-interval training times required")
    require(len(set(requests)) == len(requests), "unique requests required")
    bank = manifest["actor_manifest"]["bank"]["representative_requests"]
    width = len(manifest["actor_manifest"]["schema"]["action_names"])
    require(isinstance(bank, list) and len(bank) == len(requests) and width > 0,
            "invalid action support")
    for row in bank:
        require(len(vector(row)) == width, "invalid request vector shape")
    ordered = [row[0] for row in bank]
    require(sorted(ordered) == sorted(requests), "manifest/config support mismatch")
    spec = config["invented_return"]
    for value in spec.values():
        finite(value)
    require(spec["wrong_action_penalty"] < 0, "unique oracle winner required")
    contexts = []
    for cue in cues:
        request = finite(config["positive_cue_winner_request" if cue > 0
                                else "negative_cue_winner_request"])
        require(request in ordered, "winner absent from action support")
        for time in times:
            contexts.append({"cue": cue, "time": time, "winner": ordered.index(request),
                             "winner_request": request,
                             "oracle_best_return": spec["intercept"]
                             + spec["remaining_time_slope"] * (1 - time)})
    return contexts, len(bank), -spec["wrong_action_penalty"]


def read_rows(receipt, config, contexts, action_count, gap):
    require(set(receipt) == FIELDS, "unexpected sample fields")
    repetitions = integer(config["sampling"]["context_repetitions"], 1, 100000)
    n = len(contexts) * repetitions
    require(all(isinstance(receipt[key], list) and len(receipt[key]) == n
                for key in FIELDS), "receipt shape mismatch")
    for key in ("old_log_probs", "old_values", "returns", "advantages"):
        vector(receipt[key])
    for index in receipt["context_indices"]:
        integer(index, 0, len(contexts))
    require(receipt["context_indices"] == list(range(len(contexts))) * repetitions,
            "context schedule mismatch")
    residuals = [reward - value for reward, value in
                 zip(receipt["returns"], receipt["old_values"])]
    center = mean(residuals)
    scale = math.sqrt(mean([(value - center) ** 2 for value in residuals])) + 1e-8
    require(config["sampling"]["normalize_advantages_once_per_rollout"] is True,
            "unsupported advantage convention")
    rows = []
    for i, context in enumerate(receipt["context_indices"]):
        action = integer(receipt["actions"][i], 0, action_count)
        logits = vector(receipt["behavior_logits"][i])
        require(len(logits) == action_count, "logit support mismatch")
        probabilities, log_norm = distribution(logits)
        close(receipt["old_log_probs"][i], logits[action] - log_norm)
        winner = contexts[context]["winner"]
        observed = int(action == winner)
        reward = receipt["returns"][i]
        close(reward, contexts[context]["oracle_best_return"] - gap * (1 - observed))
        advantage = receipt["advantages"][i]
        close(advantage, (residuals[i] - center) / scale, 1e-5)
        p = probabilities[winner]
        score = observed - p
        # This oracle is evaluator-only algebra, never a fitted baseline.
        oracle_mean = contexts[context]["oracle_best_return"] - gap * (1 - p)
        oracle_score = gap * p * (1 - p)
        rows.append({"context": context, "action": action, "win": observed, "p": p,
                     "residual": residuals[i], "raw_score": residuals[i] * score,
                     "A_score": advantage * score,
                     "oracle_centered_score": (reward - oracle_mean) * score,
                     "baseline_offset_score": (oracle_mean - receipt["old_values"][i]) * score,
                     "oracle_expected": oracle_score,
                     "oracle_fixed_scale": oracle_score / scale})
    return rows, scale


def context_table(rows, contexts):
    table = []
    for context in range(len(contexts)):
        group = [row for row in rows if row["context"] == context]
        p = [row["p"] for row in group]
        table.append([context, len(group), sum(row["win"] for row in group),
                      math.fsum(p), mean(p), min(p) if p else None, max(p) if p else None]
                     + [mean([row[key] for row in group]) for key in
                        ("residual", "raw_score", "A_score", "oracle_centered_score",
                         "baseline_offset_score", "oracle_expected", "oracle_fixed_scale")])
    return table


def summarize(rows):
    require(bool(rows), "no observations")
    result = {"observations": len(rows), "winner_observations": sum(row["win"] for row in rows),
              "p_winner_mean": mean([row["p"] for row in rows]),
              "expected_winner_observations": math.fsum(row["p"] for row in rows)}
    for key in ("raw_score", "A_score", "oracle_centered_score", "baseline_offset_score",
                "oracle_expected", "oracle_fixed_scale"):
        result[key + "_mean"] = mean([row[key] for row in rows])
    result["residual_variance"] = variance_partition(
        [row["residual"] for row in rows], [row["context"] for row in rows])
    return result


def load_json(path, run, hashes):
    relative = path.relative_to(run).as_posix()
    require(not path.is_symlink() and path.resolve().is_relative_to(run.resolve()),
            "input must remain inside the result root")
    payload = path.read_bytes()
    hashes[relative] = hashlib.sha256(payload).hexdigest()
    return json.loads(payload)


def analyze_packet(run, hashes):
    config = load_json(run / "config.json", run, hashes)
    rollouts = integer(config["sampling"]["rollouts_per_fit"], 1, 100000)
    fits = {}
    for representation in config["representations"]:
        require(representation in ("graph", "self_only", "flat"), "unknown representation")
        for seed in config["initialization_seeds"]:
            integer(seed, 0, 2**32)
            name = f"{representation}-{seed}"
            require(name not in fits, "duplicate fit")
            manifest = load_json(run / name / "training.json", run, hashes)
            contexts, action_count, gap = context_spec(config, manifest)
            batches, all_rows, coverage = [], [], Counter()
            for batch in range(rollouts):
                receipt = load_json(run / name / f"rollout-{batch:02d}" / "samples.json", run, hashes)
                rows, scale = read_rows(receipt, config, contexts, action_count, gap)
                coverage.update((row["context"], row["action"]) for row in rows)
                batches.append({"rollout": batch, "normalization_scale": scale,
                                **summarize(rows), "contexts": context_table(rows, contexts)})
                all_rows.extend(rows)
            expected_coverage = {f"{c}:{a}": coverage[(c, a)] for c in range(len(contexts))
                                 for a in range(action_count)}
            require(manifest["coverage"] == expected_coverage, "training coverage mismatch")
            require(manifest["counts"]["observations"] == len(all_rows), "training count mismatch")
            tables = [row for batch in batches for row in batch["contexts"]]
            stats = summarize(all_rows)
            # Fit-pooling includes baseline/policy drift; retain within-rollout moments too.
            pooled = {key: math.fsum(batch["observations"] * batch["residual_variance"][key]
                                    for batch in batches) / len(all_rows)
                      for key in ("total", "between_context", "within_context")}
            pooled["between_share"] = (pooled["between_context"] / pooled["total"]
                                       if pooled["total"] else None)
            stats["pooled_within_rollout_variance"] = pooled
            stats["context_batches"] = len(tables)
            stats["zero_winner_context_batches"] = sum(row[2] == 0 for row in tables)
            stats["oracle_positive_context_batches"] = sum(row[12] > 0 for row in tables)
            stats["nonpositive_A_context_batches"] = sum(row[9] <= 0 and row[12] > 0 for row in tables)
            stats["nonpositive_raw_context_batches"] = sum(row[8] <= 0 and row[12] > 0 for row in tables)
            stats["first_rollout_p_winner"] = batches[0]["p_winner_mean"]
            stats["last_rollout_p_winner"] = batches[-1]["p_winner_mean"]
            stats["first_rollout_between_share"] = batches[0]["residual_variance"]["between_share"]
            stats["last_rollout_between_share"] = batches[-1]["residual_variance"]["between_share"]
            table = context_table(all_rows, contexts)
            stats["minimum_context_winner_observations"] = min(row[2] for row in table)
            fits[name] = {"summary": stats, "context_definitions": contexts,
                          "contexts": table, "rollouts": batches}
    stats = [fit["summary"] for fit in fits.values()]
    require(bool(stats), "no fits")
    summary = {"fits": len(fits), "rollouts": len(fits) * rollouts,
               "observations": sum(item["observations"] for item in stats),
               "context_batches": sum(item["context_batches"] for item in stats),
               "oracle_positive_context_batches": sum(item["oracle_positive_context_batches"] for item in stats),
               "zero_winner_context_batches": sum(item["zero_winner_context_batches"] for item in stats),
               "nonpositive_A_context_batches": sum(item["nonpositive_A_context_batches"] for item in stats),
               "nonpositive_raw_context_batches": sum(item["nonpositive_raw_context_batches"] for item in stats),
               "consumed_files": len(hashes)}
    for label, values in {
        "fit_pooled_between_share": [item["residual_variance"]["between_share"] for item in stats],
        "within_rollout_pooled_between_share": [item["pooled_within_rollout_variance"]["between_share"] for item in stats],
        "minimum_context_winner_observations": [item["minimum_context_winner_observations"] for item in stats],
        "last_rollout_between_share": [item["last_rollout_between_share"] for item in stats],
    }.items():
        defined = [value for value in values if value is not None]
        summary[label + "_range"] = [min(defined), max(defined)] if defined else None
    return {"format": "sampled-training-signal-readback-v1", "status": "complete",
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "input_sha256": hashes, "context_columns": CONTEXT_COLUMNS,
            "summary": summary, "fits": fits,
            "method": {
                "population": "All stored training receipts; no heldout.json or held-out outcomes read.",
                "residual": "one-step stored return (reward) minus stored old_value",
                "variance": "Population SS/N; total = observation-weighted between-context + within-context. Fit-pooled includes drift; pooled_within_rollout averages batch SS/N before taking ratio.",
                "A": "Recorded once-per-rollout normalized advantage; raw_score also uses unnormalized reward-old_value.",
                "score": "mean[A * (1(action==winner)-p_winner)] from stored behavior softmax logits; hypothetical independent-logit proxy at behavior, NOT actual shared-network gradient.",
                "oracle": "Post-hoc evaluator-only oracle: gap*p_winner*(1-p_winner) > 0 for nondegenerate probabilities. Oracle-centered sampled score uses reward-E_behavior[oracle reward]; raw score = oracle-centered score + baseline-offset score.",
                "fixed_scale_oracle": "Oracle expectation divided by observed rollout std+1e-8, HOLDING sample centering/scale fixed. Not the unconditional expectation of the same-batch normalized estimator.",
                "batch": "A batch is the full stored rollout, not an optimizer minibatch. Allowed files do not identify shuffled minibatch membership or later update logits.",
                "sign_counts": "Nonpositive empirical proxy counts include only context-batches with positive oracle expected raw direction; these sparse signs are descriptive, not hypothesis tests.",
            },
            "limitations": [
                "Descriptive adaptive training histories, not causal proof or independent replications; graph and self_only may duplicate behavior.",
                "Only four draws per context per rollout: every context-batch estimate is sparse. Zero observations means unknown action effect, not zero effect; even pooled rare winners are unreliable.",
                "Between-context variation includes residual state/time means, action mix and finite sampling; within-context variation in fit pooling includes changing value/policy over training.",
                "Empirical means used for SS decomposition are descriptive summaries, not fitted baselines and are never fed to a learner.",
                "No actual parameter gradients, PPO ratios/clipping, entropy, shared Jacobian, optimizer dynamics or causal intervention are recovered.",
                "Training exposure does not establish sufficient exposure. Increasing only exposure and improving baseline remain competing hypotheses.",
                "No confidence intervals or significance tests: small adaptive dependent samples do not justify independent-sample inference.",
                "No final-model or held-out performance claim; no prior inventories, archives, source files or unconsumed checkpoint hashes included.",
            ],
            "operations": {"new_simulations": 0, "model_forwards": 0, "fits": 0,
                           "optimizer_calls": 0, "heldout_outcome_reads": 0},
            "paper_bridge": {
                "decision": "Prioritize one training-only state/time baseline credit-assignment intervention over an exposure-only extension, conditional on the descriptive readback; no run authorized here.",
                "mechanistic_hypothesis": "Residual state/time offsets combined with finite action sampling can obscure the positive within-state action signal despite some winner exposure.",
                "cannot_establish": "Neither that baseline error caused shared-network failure nor that exposure was sufficient, nor that baseline correction would yield patient-level RL gain. A fit-level correlation or independent-logit sign is not a causal component ablation.",
                "named_experiment": "Patient-routing GCN-PPO incremental-learning comparison: training-only baseline-corrected GCN candidate-policy PPO versus its tensor-identical frozen-start actor.",
                "implementation_and_run_package_required": "Implement and qualify the real dynamic-candidate interface and training-only state/time baseline before any separately approved bounded run. Use the same information, action support, reward, start actor and prespecified seed/world pairing for frozen and RL policies; keep exposure/training budget fixed and disclose compute accounting. Evaluate frozen final actors on separately approved untouched patient-world evaluation streams, with prespecified cost endpoint and clinical/service guardrails. This readback does not set or reopen numerical scope or gates.",
                "paper_claim_boundary": "That single comparison tests incremental RL benefit over the same start, not isolated baseline causality, graph superiority, guaranteed clinical benefit or publication. Report null/failed guardrails; no intervening open-ended toy tuning.",
            }}


def write_once(path, result):
    payload = json.dumps(result, allow_nan=False, separators=(",", ":")) + "\n"
    # Exclusive creation, including failures: never replace or remove any result.
    with path.open("x", encoding="utf-8") as stream:
        stream.write(payload)


def main():
    output = HERE / "signal-readback.json"
    if output.exists():
        raise FileExistsError(output)
    hashes = {}
    try:
        result = analyze_packet(RUN, hashes)
        write_once(output, result)
    except Exception as error:
        failure = HERE / f"signal-readback-error-{uuid.uuid4().hex}.json"
        write_once(failure, {"status": "error", "error": repr(error), "input_sha256": hashes,
                             "originals_preserved": True})
        raise
    print(json.dumps(result["summary"], indent=2))
    print(output)


if __name__ == "__main__":
    main()
