# Learned residual policies versus retuned heuristics on the same fresh worlds

Date: 2026-09-16. Branch `e2b-run`. **Exploratory diagnostic, not a
preregistered confirmation.** No training. No protected seed stream. No
existing artifact changed. Tool: `evaluation/learned_policy_fresh_world_comparison.py`;
unified table built from its rows plus the 2026-09-15 heuristic screens.
Artifacts: `artifacts/unified_table.{md,json}`, `artifacts/learned_eval_summary.json`.
Row files under gitignored `results/learned_policy_fresh_world_comparison/`.

## Why this exists

The anchor screens of 2026-09-15 compared MDL-2 only with its own variants.
The paper is about the graph-aware residual policy, so the learned policy has
to sit in the same table on the same worlds. The formal AFR-GCN-DDPG
checkpoints are on the collaborator's machine; the locally archived Stage C
development checkpoints (AFR-GCN-TD3 and AFR-Flat-TD3, training seeds 20–22,
final episode-100 and frozen-pretrain variants) are the matched backbone
ablation the manuscript already reports, and they were evaluated here through
the formal evaluator's own `evaluate_deployment_candidate` with the
prespecified fixed deployment (scale 1.0, checkpoint gates).

Worlds: seed stream 99.7M, 100 paired replications per routing scenario, the
same worlds as the heuristic screens. The evaluator's own MDL-2 and MDL-3
anchor rows matched the screens' rows on all 400 worlds with zero difference,
so every row below is world-paired.

## Unified table, four routing scenarios pooled, versus the executed MDL-2 anchor

Learned rows: three training seeds, two-level paired bootstrap (the
publication method). Heuristic rows: world-level paired bootstrap. Guardrail:
the formal pooled paired-CI rule (completion ≥ −0.1 pp, manufacturing
ineligibility ≤ +0.1 pp, patients lost ≤ +1.0).

| policy | Δcost (95% CI) | Δcompletion pp | Δmfg-inelig pp (CI) | Δlost | C/I/L |
|---|---:|---:|---:|---:|:--:|
| AFR-GCN-TD3 final (3 seeds) | −0.677% [−0.773, −0.579] | +0.18 | −0.471 [−0.510, −0.432] | −27.2 | ✓ ✓ ✓ |
| AFR-GCN-TD3 frozen pretrain | −0.677% [−0.779, −0.576] | +0.18 | −0.463 | −27.2 | ✓ ✓ ✓ |
| AFR-Flat-TD3 final | −0.325% [−0.380, −0.264] | +0.01 | −0.312 | −4.6 | ✓ ✓ ✓ |
| AFR-Flat-TD3 frozen pretrain | −0.328% [−0.380, −0.271] | +0.02 | −0.315 | −4.8 | ✓ ✓ ✓ |
| MDL-3, no learning | −4.143% [−4.432, −3.856] | +1.03 | +0.203 [+0.132, +0.274] | −22.2 | ✓ ✗ ✓ |
| MDL-2 order-up-to ×1.4, no learning | −3.540% [−3.831, −3.246] | +1.04 | +0.201 | −23.2 | ✓ ✗ ✓ |
| **GCN-TD3 final on MDL-3 anchor, zero-shot** | **−4.329% [−4.562, −4.106]** | **+0.99** | **−0.188 [−0.241, −0.134]** | **−31.7** | **✓ ✓ ✓** |
| Flat-TD3 final on MDL-3 anchor, zero-shot | −3.467% [−3.662, −3.265] | +0.57 | +0.077 | −1.7 | ✓ ✗ ✗ |

Relative to the MDL-3 anchor on the same worlds:

| policy | Δcost vs MDL-3 (95% CI) | Δcompletion pp | Δmfg-inelig pp | Δlost | C/I/L |
|---|---:|---:|---:|---:|:--:|
| GCN-TD3 final on MDL-3, zero-shot | −0.194% [−0.377, −0.053] | −0.03 | −0.391 | −9.5 | ✓ ✓ ✓ |
| Flat-TD3 final on MDL-3, zero-shot | +0.706% [+0.602, +0.822] | −0.46 | −0.126 | +20.5 | ✗ ✓ ✗ |

Per-seed Δcost versus MDL-3 for the GCN zero-shot row: −0.385%, −0.053%,
−0.144%. All three favorable; seed 21 fails the completion guardrail against
MDL-3 (passes against MDL-2).

## What the table says

1. **The paper's main effect reproduces out of sample.** On worlds no policy
   or protocol ever touched, AFR-GCN-TD3 beats MDL-2 by 0.68% (development
   stream: 0.73%; formal DDPG: 0.66%), passes all three guardrails, and the
   frozen actor is indistinguishable from the final one. Graph-minus-flat is
   0.35 points, favorable in every seed. The online-RL null reproduces too.
2. **The learned residual survives on the coverage-correct anchor, without
   retraining.** Dropped zero-shot onto MDL-3, whose anchor-action features it
   never saw in training, the GCN residual still lowers cost by 0.19% (CI
   excludes zero, three of three seeds). The flat residual does the opposite:
   +0.71%, and it breaks two guardrails. The graph-versus-flat separation is
   therefore larger on the better anchor (0.9 points) than on MDL-2 (0.35).
3. **The GCN residual repairs the guardrail that MDL-3 alone breaches.**
   MDL-3 raises manufacturing ineligibility by 0.20 pp and fails the formal
   gate. MDL-3 plus the GCN residual lowers it by 0.19 pp relative to MDL-2
   and passes all three margins, while keeping essentially all of MDL-3's
   cost and completion gains and losing the fewest patients of any row
   (−31.7 per episode). Under decision B, which keeps the gate as written,
   this is the only row in the table that both captures the coverage gain
   and is clinically admissible.
4. **Mechanism.** Per episode, pooled: MDL-3 starts 79 more patients than
   MDL-2 and loses 13 more in manufacturing. The GCN residual on MDL-3 starts
   18 fewer than MDL-3, routes 162 more specimens, loses 14 fewer in
   manufacturing and 39 fewer while waiting. The residual reduces
   manufacturing-stage loss on both anchors (−0.47 pp on MDL-2, −0.39 pp on
   MDL-3), so it is a property of the routing policy, not of the anchor. It
   appears to move specimens away from starts that would fail.
5. **Per-scenario, the combined policy is not uniformly admissible.** Against
   MDL-2 it passes the pooled rule but breaches manufacturing ineligibility
   in the abrupt-shift and regional-drift scenarios and patients lost in
   regional drift; nominal and compound stress pass everything. The formal
   protocol uses the pooled rule (`strict_scenario_clinical_noninferiority`
   is false), so this matches how the formal learned policy was judged, but
   the scenario-level exceptions must be reported the same way the formal
   result reports seeds 10 and 14.

## What this does not establish

- These are the TD3 development checkpoints, three seeds, not the formal
  five-seed DDPG policies. The direction and magnitude match the formal DDPG
  on MDL-2, but the MDL-3 rows have no DDPG counterpart yet.
- The MDL-3 rows are zero-shot: the actor consumes anchor-action features it
  was trained to expect from MDL-2. That the result is favorable anyway is
  evidence of robustness, not of what a policy trained on MDL-3 would do.
- The finding was observed, not preregistered. It cannot enter the paper as a
  confirmatory result. It can enter as an exploratory result on fresh seeds
  with the label above, or it can motivate a preregistered confirmation:
  the collaborator's formal DDPG checkpoints evaluated zero-shot on MDL-3 on
  a new seed stream, with the gate and reading rule written before the run.
- The MDL-3 anchor is not adopted under decision B. What this table shows is
  a candidate *deployment configuration* (coverage-correct anchor plus
  graph residual) whose adoption would itself be a change-control amendment.

## Reconciliation with the headroom screen

The 2026-09-16 label screen found the routing channel's single-step
state-dependent value on MDL-3 to be 0.02% and its whole-channel value 0.07%.
The multi-step residual here captures 0.19% on MDL-3, about three times the
single-step probe, the same ratio observed on MDL-2 (0.68% versus 0.12%,
roughly five times). The screen's gates were built to decide whether to train
a new policy; they do not bound what an already-trained policy does when it
acts every epoch, and this table is the direct measurement of that.
