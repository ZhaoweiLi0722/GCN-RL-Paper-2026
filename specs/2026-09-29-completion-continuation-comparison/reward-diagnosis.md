# Reward diagnosis after S4

Date: 2026-09-29. Read-only synthesis and algebra on archived evidence, not a
reward intervention, training result, cost sweep or authorization to retrain.

## Bottom line

It is reasonable to examine reward design, but **failure to beat a competent
rule does not establish that the reward is wrong**. S4 contains no neural
learning at all. It can expose conditional control trades and numerical
limitations; it cannot diagnose neural credit assignment or prove that a
shaping term would rescue online DDPG.

The highest-priority reward issue is already concrete: the historical training
data used inconsistent value targets. Resolve that before tuning operational
weights. The prospective N1-N7 code and N5 design already address parts of this
contract; do not describe those preparations as newly discovered fixes or as
validated performance improvements.

## Three different questions

| Question | Existing evidence | Consequence |
| --- | --- | --- |
| Is the implementation learning one well-defined return? | The formal replay audit found absolute-cost pretraining rewards mixed with anchor-relative online rewards; discontinuous cached multi-step windows; calibration using gamma rather than gamma^n for multi-step items. | A consistent reward/return/replay contract is necessary. This is an implementation concern, not a reason to change business priorities. |
| Does the objective represent the intended operation? | E1's `costs_and_delay_objective`, closure rules and meaningful-benefit inputs are still null. The support queue has hypothetical holding, labor and switching costs, not validated patient priorities. | Obtain explicit operational definitions/units before changing weights or claiming clinical relevance. |
| Does a correct objective provide useful learning signals? | S4 has per-step costs and full settlement, not only a terminal reward. Inner action estimation remains noisy; no actor/critic was trained. | Shaping, scaling, horizons and gradient quality remain hypotheses requiring isolated verification. Extra bonuses cannot be called a proven solution. |

Historical source: `specs/2026-09-29-formal-replay-contract/readout.md` and its
raw audit JSON. In that audit, 153/314 cached emitted windows cross an observed
trajectory discontinuity, and the offline/online reward definitions differ.
These facts do not establish the causal explanation of the historical null
final-vs-frozen result. Loss coefficients alone also do not establish gradient
domination; gradient norms/directions and causal comparisons were not recorded.

## Recommended priority

### 1. One objective, one target, one replay meaning

Retain the already proposed N5 contract for a finite fully settled task:
negative complete cost, gamma=1, one-step replay, one fixed positive scale for
all arms, and an absorbing terminal state only after liabilities settle.
One-step TD still bootstraps future cost; this is not a one-step planning task.

Use the same semantics in offline preparation, calibration, online data,
target construction and evaluation. Keep isolated counterfactual transitions
as one-step records, not fictitious consecutive trajectories. Do not insert old
mixed-semantic cached data into a corrected critic without an explicit audited
conversion. A different continuing-task objective would require a new design.

This is already represented in `src/rl/validated_returns.py`, the prospective
adapter's shared Bellman target and the N5 development-design protocol. The
current task did not edit those modules or migrate old checkpoints.

An anchor-relative immediate reward is not automatically a harmless baseline:
subtracting a state-dependent reference inside the accumulated objective can
change which visited states are preferred. Keep policy-evaluation cost fixed;
use an auxiliary counterfactual target only with documented semantics and a
separate attribution comparison. Do not equate reward centering with an
action-independent baseline in an actor-gradient calculation.

### 2. Separate operational weights from learning aids

Do not reduce labor cost or inflate waiting penalties merely because a tree
or RL policy takes a different trade. In S4, some tree actions save labor but
increase waiting and switching enough to lose on total cost; the opposite
trade against the reservation rule is also visible. Neither establishes the
correct operational weights.

The earlier, different finite-queue sensitivity also retained a matching
archived non-neural path in 66/72 dependent repriced cell/settings. It is not
evidence about every possible objective, but it weakens the case for another
unjustified cost sweep. Retain clinical/feasibility constraints as separately
audited conditions rather than trading them away via convenient reward weights.

### 3. Consider objective-preserving shaping only as a learning hypothesis

Potential-based shaping has the form `r' = r + gamma*Phi(next) - Phi(current)`.
Under its stated MDP and boundary assumptions it can preserve optimal policies;
it does not manufacture additional attainable performance. A finite episodic
use must handle terminal potentials and truncation consistently. This is a
theoretical option, not an assurance for a function-approximation implementation.
[Ng, Harada and Russell (1999), original paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf).

Use only causally available public observations/history for any potential;
never hidden productivity or future patient completion. A bounded potential
and correct terminal treatment require an explicit contract and tests. Do not
add ad-hoc positive bonuses for routing, resource use or repeated progress
events that can be collected without improving the external objective.

## Read-only algebra check, not a shaping experiment

As a post-hoc accounting illustration, use `Phi = -number of released unfinished
tasks`, computed from the archived public stage labels, with gamma=1. No new
potential was fitted and no trajectory, learner or reward file was modified.
On all 288 S2 archived episodes / 9,364 intervals:

- Initial potential is -2; fully settled terminal potential is zero.
- Sum of shaped rewards equals sum of raw negative costs plus exactly 2.
- Maximum telescoping residual is exactly zero. Every recorded return receives
  the same offset, so every comparison/ranking is unchanged.
- Zero new simulator queries or optimizer updates. This tests accounting only,
  not sample efficiency, policy generalization or learned performance.

Source steps SHA256:
`ed6167ca9cbbd384204a5ba35b00c3c784fe7b638f6c20ddd2ba762cdca94448`.
The check is reproducible without writing output or changing reward semantics:

```python
import json, math
from pathlib import Path
groups = {}
path = Path('reports/2026-09-29-completion-control-screen/run/steps.jsonl')
for line in path.read_text().splitlines():
    row = json.loads(line)
    groups.setdefault(row['case'], []).append(row)
phi = lambda s: -sum(x not in ('scheduled', 'done') for x in s['stages'])
assert len(groups) == 288
for rows in groups.values():
    raw = math.fsum(-r['cost']['total'] for r in rows)
    shaped = math.fsum(-r['cost']['total'] + phi(r['after']) - phi(r['before'])
                       for r in rows)
    assert phi(rows[-1]['after']) == 0
    assert shaped - raw == 2 == -phi(rows[0]['before'])
```

## What comes next, and what does not

Do not continue enlarging this synthetic action-ranking screen. Its prospective
gate did not pass, and changing rewards to reverse that result would answer a
different question. Preserve the negative result and current manuscript claims.

Before a corrected online-versus-frozen performance study, resolve the task's
remaining operational/model/headroom requirements. Then pre-register a clean
reward contract, identical information/initialization, competent frozen and
adaptive baselines, equal data/compute, and unchanged external outcome metrics.
To test shaping separately, compare shaped/unshaped learning under the same
external objective and charge exploration costs; do not change the objective
and the algorithm together and credit the difference to online RL.

No new training, reward revision or cost weights are approved by this readout.
No need to rebuild the already tested N1-N7 mechanics merely to show activity.
The next scientific decision is whether to pursue a domain-grounded corrected
development study, not which weights make the current synthetic tree win.
