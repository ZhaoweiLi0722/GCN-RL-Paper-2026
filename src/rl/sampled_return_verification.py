"""Independent scalar arithmetic for saved sampled-control receipts; no Torch."""

from collections import Counter
import copy
import math


def _finite(values):
    if not values or any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
        raise ValueError("nonempty finite numeric records required")


def _close(a, b, tolerance=2e-6):
    if not math.isfinite(a) or not math.isfinite(b) or abs(a-b) > tolerance:
        raise ValueError("recorded scalar differs from independent arithmetic")


def _distribution(logits):
    _finite(logits)
    highest = max(logits)
    exponential = [math.exp(v-highest) for v in logits]
    total = math.fsum(exponential)
    return [v/total for v in exponential], highest+math.log(total)


def _oracle(config, cue, time):
    ordered = sorted(config["request_values"])
    winner_request = config["positive_cue_winner_request"] if cue > 0 else config["negative_cue_winner_request"]
    winner = ordered.index(winner_request)
    spec = config["invented_return"]
    base = spec["intercept"]+spec["remaining_time_slope"]*(1-time)
    return winner, [base+(0 if action == winner else spec["wrong_action_penalty"]) for action in range(len(ordered))]


def verify_sample_batch(receipt, config):
    keys = {"context_indices", "actions", "old_log_probs", "old_values", "returns", "advantages", "behavior_logits"}
    if set(receipt) != keys:
        raise ValueError("unexpected sampled receipt fields")
    contexts = [(cue, t) for cue in config["cues"] for t in config["training_times"]]
    count = len(contexts)*config["sampling"]["context_repetitions"]
    if any(len(receipt[k]) != count for k in keys):
        raise ValueError("missing or extra sample rows")
    if receipt["context_indices"] != list(range(len(contexts)))*config["sampling"]["context_repetitions"]:
        raise ValueError("context visitation differs from fixed schedule")
    for key in ("old_log_probs", "old_values", "returns", "advantages"):
        _finite(receipt[key])
    pairs, advantages = Counter(), []
    for i, (index, action) in enumerate(zip(receipt["context_indices"], receipt["actions"])):
        if type(index) is not int or type(action) is not int or not 0 <= action < len(config["request_values"]):
            raise ValueError("invalid integer sample identity")
        logits = receipt["behavior_logits"][i]
        if len(logits) != len(config["request_values"]):
            raise ValueError("wrong class support")
        _, log_normalizer = _distribution(logits)
        _close(logits[action]-log_normalizer, receipt["old_log_probs"][i])
        _, oracle = _oracle(config, *contexts[index])
        _close(oracle[action], receipt["returns"][i])
        pairs[(index, action)] += 1
        advantages.append(receipt["returns"][i]-receipt["old_values"][i])
    mean = math.fsum(advantages)/count
    std = math.sqrt(math.fsum((v-mean)**2 for v in advantages)/count)
    for expected, actual in zip(advantages, receipt["advantages"]):
        _close((expected-mean)/(std+1e-8), actual, 1e-5)
    return {"sample_count": count, "return_sum": math.fsum(receipt["returns"]),
            "coverage": {f"{context}:{action}": pairs[(context, action)]
                         for context in range(len(contexts)) for action in range(len(config["request_values"]))}}


def verify_charge(previous, receipt, config):
    if set(receipt) != {"fixture", "kind", "amount", "budget"}:
        raise ValueError("unexpected budget receipt fields")
    current = copy.deepcopy(previous)
    name, kind, amount = (receipt[k] for k in ("fixture", "kind", "amount"))
    names = {f"{r}-{s}" for r in config["representations"] for s in config["initialization_seeds"]}
    if name not in names or kind not in ("observations", "actor", "critic") or type(amount) is not int or amount < 1:
        raise ValueError("invalid charge identity")
    if (kind == "observations" and amount != 48) or (kind != "observations" and amount != 1):
        raise ValueError("charge does not match fixed batch/call size")
    counters = current["by_fit"].setdefault(name, {"observations": 0, "actor": 0, "critic": 0})
    counters[kind] += amount
    key = "observations" if kind == "observations" else "optimizer_calls"
    current[key] += amount
    per_cap = config["caps"]["observations_per_fit"] if kind == "observations" else config["caps"][kind+"_calls_per_fit"]
    global_cap = config["caps"]["total_training_observations"] if kind == "observations" else config["caps"]["total_optimizer_calls"]
    if counters[kind] > per_cap or current[key] > global_cap or current != receipt["budget"]:
        raise ValueError("budget overflow, refund or receipt mismatch")
    return current


def actor_metrics(logits, config):
    contexts = [(cue, t) for cue in config["cues"] for t in config["heldout_times"]]
    if len(logits) != len(contexts):
        raise ValueError("fixed final evaluation count mismatch")
    margins, correct, expected, greedy = [], [], [], []
    for row, (cue, time) in zip(logits, contexts):
        winner, oracle = _oracle(config, cue, time)
        if len(row) != len(oracle):
            raise ValueError("evaluation support mismatch")
        probabilities, _ = _distribution(row)
        choice = max(range(len(row)), key=row.__getitem__)
        correct.append(choice == winner)
        margins.append(row[winner]-max(v for i, v in enumerate(row) if i != winner))
        expected.append(math.fsum(p*q for p, q in zip(probabilities, oracle)))
        greedy.append(oracle[choice])
    return {"accuracy": sum(correct)/len(correct), "minimum_winner_margin": min(margins),
            "expected_returns": expected, "greedy_returns": greedy,
            "mean_expected_return": math.fsum(expected)/len(expected),
            "mean_greedy_return": math.fsum(greedy)/len(greedy)}


def evaluation_metrics(logits, critic_values, training_return_mean, coverage, config):
    actor = actor_metrics(logits, config)
    expected = actor["expected_returns"]
    if len(critic_values) != len(expected):
        raise ValueError("fixed critic evaluation count mismatch")
    _finite(critic_values)
    _finite([training_return_mean])
    mse = math.fsum((v-y)**2 for v, y in zip(critic_values, expected))/len(expected)
    baseline = math.fsum((training_return_mean-y)**2 for y in expected)/len(expected)
    expected_keys = {f"{i}:{j}" for i in range(len(config["cues"])*len(config["training_times"])) for j in range(len(config["request_values"]))}
    if set(coverage) != expected_keys or any(type(v) is not int or v < 0 for v in coverage.values()):
        raise ValueError("complete integer context/action coverage required")
    accuracy = actor["accuracy"]
    gates = {"accuracy": accuracy >= config["gates"]["heldout_greedy_accuracy"],
        "margin": actor["minimum_winner_margin"] >= config["gates"]["heldout_minimum_winner_logit_margin"],
        "critic": mse <= config["gates"]["critic_mse_to_training_mean_baseline_ratio_max"]*baseline,
        "coverage": min(coverage.values()) >= config["gates"]["minimum_samples_each_context_action"]}
    return {"accuracy": accuracy, "minimum_winner_margin": actor["minimum_winner_margin"],
            "critic_mse": mse, "constant_baseline_mse": baseline,
            "critic_mse_ratio": mse/baseline if baseline > 0 else None,
            "mean_expected_return": actor["mean_expected_return"],
            "mean_greedy_return": actor["mean_greedy_return"], "gates": gates, "passed": all(gates.values())}
