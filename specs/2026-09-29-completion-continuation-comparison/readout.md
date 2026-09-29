# S4: feedback-continuation result

## Decision

The bounded two-request feedback tree does not establish a stable conditional
gain over the booked rule in any of the four archived public contexts. It has
lower predicted cost than the reservation continuation, but that is the weaker
comparator in this diagnostic. Report both; do not claim an improvement by
dropping the stronger booked comparator.

This is not a proof of near-optimality or of reward-function failure. The tree
is a restricted controller fitted with finite Monte Carlo samples: 64/180
branch/first-request choices change when the inner budget increases from 32 to
128. It is neither an exact bound nor a validated strong MPC comparator.

Decision: do not promote this synthetic screen to DDPG training. Follow the
user's requested reward/objective diagnosis in `reward-diagnosis.md`, without
changing costs or learning from formal holdout outcomes.

## Verified execution

- Protocol/config: `e4e9fef`. Execution source:
  `6409afef49b667b7abef146ea117e59b7bb4cc5c`.
- One recorded matrix, exit 0, 35.638 seconds including its bounded audit.
  PID 25554 / parent 2465 was observed with the matching command while its
  selection/validation boundaries advanced; later process inspection found
  no remaining evaluator. No retry or enlarged budget.
- All six aliases/four unique contexts retained. Twelve reachable event
  branches across contexts produced 180 branch/first-action fits.
- 345,600 inner planning paths + 1,966,080 outer assessment paths = 2,311,680
  model forecasts; 49,204,865 scalar-equivalent transition queries.
- Zero new actual-world/patient episodes, optimizer updates or reward changes.
  All unfinished work and accepted costs settle under the common closure rule.
- Independent read-only audit passes all 436 raw arrays, 6 source + 315 input
  locks, frozen-tree/selection order and recomputed rankings/paired arithmetic.
  Scalar checks replay 9,240 bounded paths / 187,703 intervals, not the full
  matrix. No nonfinite metric, traceback or error artifact was found.
- 14 new tests, 154 combined relevant tests, full compileall and diff checks pass.

## Conditional results

Entries are frozen-policy cost differences in validation blocks A/B/C at 2048
paired paths. Negative favors the tree. Each comparator selects its own first
request in the independent selection block and then freezes it. The same
constant response model, information and action set apply to all methods.

| Context; downstream slots | Tree128 minus booked | Tree128 minus reservation | Stable signal |
| --- | --- | --- | --- |
| Unchanged; 1 | +.106827 / +.060226 / +.194611 | -.283844 / -.741516 / -.597351 | No |
| Unchanged; 4 | 0 / 0 / 0 | -1.009003 / -.972824 / -.840652 | No |
| Changed public context; 1 | +.102020 / +.071716 / +.255493 | -.728882 / -.511551 / -.786240 | No |
| Changed public context; 4 | +.123703 / +.242996 / +.215393 | -.765625 / -.551712 / -.589172 | No |

The changed public contexts each alias both persistent and fast-independent
S2 records. They are not two independent populations. Outer blocks quantify
fixed-model integration uncertainty, not generalization across operating worlds.
No population significance or clinical benefit threshold was evaluated.

Tree32 also fails to improve on booked in every validation context/block:
it exactly matches booked for unchanged/1-slot and has positive differences
elsewhere. Thus none passes the prespecified requirement of negative differences
against both comparator continuations in all blocks at both tree budgets.

The tree's 32-to-128 action changes are 9/30 and 14/30 in the unchanged 1/4-slot
contexts, and 15/60 and 26/60 in the changed 1/4-slot contexts. These are
dependent branch/action table entries, not experimental replicates.

Even against the unoptimized pure booked rule, Tree128 has zero or positive
validation mean differences in every context/block. The 4-slot changed case
has +.206772 / +.265274 / +.188171 versus that rule; the other rows coincide
with the table's booked comparison.

## What the cost components reveal

This decomposition is descriptive, not a causal diagnosis of training:

- In unchanged/1-slot, Tree128 saves .155-.214 labor units but adds .228-.309
  holding and .041-.046 switching units versus booked. Total cost worsens.
- In changed/4-slot, it saves .070-.129 labor units versus the selected booked
  continuation but adds .204-.265 holding and .047-.051 switching units.
- In changed/1-slot, small holding reductions in A/B do not cover increased
  labor/switching, and the holding sign reverses in C. It is not a replicated
  operational delay improvement.
- In unchanged/4-slot, all three component differences are exactly zero.

Changing weights after these results to reward one of these trades would
change the task. No such repricing was performed. This forecast-only study
also cannot measure neural credit assignment or reward-shaping learning speed.

## Evidence and remaining boundary

Root: `reports/2026-09-29-completion-continuation-comparison/run/`.
`trees/` preserves branch probabilities, both fitted tables and all inner
action comparisons; `selection/` preserves frozen first requests; `raw/`
contains all inner costs and outer component arrays. `raw.jsonl` records
seeds, uniform hashes, query counts and frozen-artifact hashes. `events.jsonl`
certifies the implementation's tree-before-assessment and selection-before-
validation sequence. `blocks.json` retains all nested budgets and comparisons.

| Artifact | SHA256 |
| --- | --- |
| summary.json | `3303cc459a2d2f2698d54053b15887eb18c520535dbe34b29fd047493b598672` |
| blocks.json | `34b41af6bfdd421df9cdb2313c05a304703b7b563e39a7bcc07bcb07fe114051` |
| audit.json | `055ba6ea5a98a3f20cb099da25c7e7ca87f110ace995635190d37f9baa17c78a` |
| inventory.json | `b03fb011642ff2a62bbcf4da9dc4cdf1764ff2a2e729349d9f3b5fdb3820b570` |

Use only the read-only audit command in `engineering.md` for verification.
Existing results, reward weights, formal paper evidence and Stage E remain
unchanged. Do not infer that other horizons, models or learned controllers
would have the same ranking. Further scientific execution needs new scope.
