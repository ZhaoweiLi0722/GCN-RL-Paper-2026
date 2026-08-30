# Results

## Routing label budget study — 2026-08-29

### Decision

**The frozen reading rule returned `g1_negative_underpowered`. That
classification is mechanically correct and scientifically misleading, and it
is NOT the conclusion of this study.**

The rule's precondition — that this study measures the same estimator problem
Stage G1 measured — is violated by the data, and the violation is measurable
rather than a matter of opinion. The honest classification is
**`precondition_violated_cannot_adjudicate_g1`**.

Stage G1's negative therefore **stands unchanged**. Nothing about the closed
routing conclusion is reopened.

### The number that decides it

| Measurement at Stage G1's exact budget shape (3 vs 5 worlds) | Agreement |
| --- | ---: |
| This study | **0.821** |
| Stage G1 as reported | **0.545** |

At matched budget shape and matched procedure, this study's agreement is 0.28
higher than G1's. Budget cannot explain a gap measured *at the same budget*.
Whatever destabilised G1's labels is a property of its states and policy
context — 156 frozen-pretrain trajectory states drawn from Stage F1 training
runs — not of how many worlds were averaged.

### Agreement versus budget

27 states, 5 arms, 64 worlds, 8,640 paired-CRN rollouts.

| Worlds per group | Agreement | p10 | p90 |
| ---: | ---: | ---: | ---: |
| 1 | 0.785 | 0.741 | 0.852 |
| 2 | 0.796 | 0.741 | 0.852 |
| 4 | 0.822 | 0.778 | 0.889 |
| 8 | 0.855 | 0.778 | 0.926 |
| 16 | 0.912 | 0.852 | 0.963 |
| 32 | 0.952 | 0.926 | 0.963 |

Budget does help — agreement climbs 0.17 from one world to 32 — but the curve
never starts below the 0.70 gate. A study whose baseline already clears the
gate cannot test whether budget lifts a 0.545 baseline over it.

### Why the rule still fired, and why that is a lesson

The reading rule was written on the assumption that this study's low-budget
agreement would land near G1's 0.545, making "does it climb past 0.70?" the
decisive question. It did not: the setting is simply more stable than G1's.
The rule had no clause for "the baseline does not reproduce", so it read a
high plateau as evidence of successful budget lifting.

**A prospective rule protects against choosing a threshold after seeing the
data; it does not protect against the experiment failing to reproduce the
condition it was meant to probe.** That check has to be explicit. Any future
budget study of this kind should gate on reproducing the baseline first, and
declare itself uninformative if it cannot.

### Pre-registered prediction versus outcome

| Prediction | Outcome | |
| --- | --- | --- |
| Curve climbs but does not reach 0.70 by k = 32 | Reached 0.952; started at 0.785 | **wrong** |
| Likeliest classification: inconclusive | Inconclusive *for G1*, by a different route than predicted | partly right |

I predicted the wrong levels. The practical conclusion — that this study
cannot reopen G1 — matches the predicted classification, but for a reason the
prediction did not anticipate.

### Secondary result: sequential halving does not help here

| Simulator calls per arm | Uniform | Sequential halving |
| ---: | ---: | ---: |
| 2 | 0.793 | 0.784 |
| 4 | 0.826 | 0.804 |
| 8 | 0.869 | 0.835 |
| 16 | 0.907 | 0.880 |

**Sequential halving is worse than uniform allocation at every budget tested**,
by 0.01 to 0.03. This contradicts the synthetic validation, where at equal
budget it beat uniform (0.837/0.940/0.987 against 0.730/0.860/0.960).

The likely reason is arm count. The synthetic fixture had 11 arms and world
noise far exceeding the gaps between them, so discarding half the field early
was cheap. Here there are 5 arms, so halving runs about two rounds and can
eliminate a near-optimal arm before the near-ties are resolved — and near-ties
are precisely what the argmax depends on. The method's advantage does not
transfer to short ladders.

Reported as a negative for the labeller. It remains available for
many-armed screens, where the synthetic evidence supports it, and should not
be used for 5-arm families on this evidence.

### What this study did establish

1. In the MDL-2-anchored routing setting with centred-pressure specimen arms,
   best-action labels are **stable**: 0.785 agreement from a single world,
   0.952 from 32.
2. Label agreement is budget-sensitive here, but from a high base.
3. Sequential halving does not improve short-ladder labelling in this
   simulator.
4. G1's instability is **not** a budget artefact of the estimator class, since
   the same procedure at the same budget produces far higher agreement on
   different states.

### What would actually adjudicate Stage G1

Only a replication on G1's own states. Its 156 frozen-pretrain trajectory
states and their checkpoint artifacts live on the RTX 4090 host. A study there
could compare G1's 3+5-world labels against a larger pool on identical states,
which is the comparison this study attempted and could not make. That is a
separate specification and is not proposed here.

### Evidence

- Pool rows: `results/routing_label_budget_study/pool_rows.csv` (8,640 rows)
- Pool summary: `results/routing_label_budget_study/pool_summary.json`
- Analysis: `results/routing_label_budget_study/budget_analysis.json`
- Config: `experiments/configs/routing_label_budget_study.json`
