# P2 reference-prior residual candidate: proposed next pilot

2026-10-01. DESIGN ONLY. Scientific execution NOT authorized. No P2 patient
simulation, fitting, fresh data or claimed seed namespace exists. Config:
`experiments/configs/candidate_reference_prior_design_20261001.json`.
The P1/P1-R1 protocols, failed results and their preservation remain immutable.

## Why this change, not another BC or reward search

P1-R1 stopped before PPO: block60 graph/self-only failed fixed-budget action
qualification. The saved-data readback also showed21.64%-26.34% mean reference
sampling probability across9 models, including greedily accurate initializers.
This does not establish adverse sampled outcomes or failure of PPO/DDPG.
The CE-only initialization did not use reward labels. Reward modification is
not motivated by this failed gate. See the P1-R1 terminal readout and raw audit.

The next question remains whether trajectory-return training improves cost
over the same initialized frozen controller and continued imitation. It is not
deployment adaptation, a new uncertain operating scenario, near-optimality or
PPO-versus-DDPG. This design removes the unnecessary requirement to approximate
an already available reference action with finite BC fitting.

## Policy definition

Let K be the number of unique legal requested-net classes and r the reference
class. For K>1 define q(r|s)=1-epsilon and q(a|s)=epsilon/(K-1) for a!=r.
For K=1, q=1. Proposed epsilon=.10, chosen prospectively as an explicit
conservative exploration allocation, not fitted or screened against outcomes.
This is a design choice, not a known optimum. The proposed configuration is
fixed across all blocks/representations and has no schedule or search.

Policy logits are log(q(a|s)) + f_theta(s,a). Set only the last affine scorer
layer to zero initially, so f=0 for every finite supported input. Initialize
the final value layer to zero too. All remaining encoder/head weights retain
the explicitly seeded initialization. No P1-R1 fitted weights, data, critic,
optimizer state or qualification labels initialize the new policy.

Consequences and limits:
- Initial greedy selection is exactly the original R4 request. Canonicalization
  preserves R4 as representative even when its class aliases the anchor.
- Initial stochastic collection has90% reference-class probability, NOT100%.
  Each distinct other class has10%/(K-1), regardless of aliases or option order.
  At six classes that is2% each. K=1 has no alternative to explore.
- This is per-decision, not per-trajectory fidelity or a safety guarantee.
  If every one of52 steps has alternatives, expected deviations are5.2. With
  independent private draws, probability of no deviation is.9**52, about.0042.
  Actual execution can merge requests or change future states, so these counts
  do not measure beneficial/damaging patient interventions.
- The fixed log prior remains in logits, but residual learning may override it.
 90% is an INITIAL probability, not a continuing exploration/safety bound.
- Zero output layers give nonzero first-backward output-layer gradients but
  zero initial encoder gradients. Encoders can receive gradients after output
  weights change; this is not proof they will learn useful network effects.
- PPO must record/re-evaluate the actual prior-plus-residual distribution,
  including entropy and old log probabilities. No teacher-probability surrogate.

New explicit model type/manifest: `ReferencePriorCandidatePolicy`,
`reference-prior-candidate-v1`. Legacy CandidatePolicy defaults and checkpoint
meaning stay unchanged. Prior mass, allocation and initialization bind to the
policy definition/receipt/checkpoint; mismatches fail rather than migrate.

## Information and controls

Retain the P1-R1 public561-dimensional raw input,80-dimensional original
requests,20 facility nodes/36 specimen message edges, unchanged physical
190-edge capacity dynamics and21-node hub-aware R4 inference. Reference/anchor
role flags are already public inputs for every representation; the prior adds
no outcome oracle. The existing52-step nominal-history environment, absolute
cost reward scaled once1e-9, candidate bank, terminal accounting and R4 hashes
are unchanged. No new scenario, reward/cost adjustment or DDPG fitting.

Same three R4 reference blocks60/61/62. Graph and self-only share initial tensor
values; flat is parameter-unmatched. Counts remain59,602/59,602/238,658 with
widths16/32. R4 itself contains graph processing; this tests an incremental
residual-message operator, not removal of all graph information.

Within each representation/block fork FROZEN/PPO/BC-CONTINUE from exactly the
same new initialized weights and declared sampling state. Both trainable arms
start with empty optimizer moments. No learned BC initialization or demonstration
phase. This is an explicit prospective protocol change, not a lower P1 gate.
Retain32 training episodes/continued arm, four-episode rollouts,128 Adam steps
per arm, and the original P1 PPO/continued-BC settings. BC learns reference
actions on its sampled states; PPO learns returns. No test-time updates.

The nine greedy frozen controllers initially equal R4 by construction. They
are not nine new independent baselines. For the first integration proposal,
retain the existing396-evaluation loop to audit full closed-loop equivalence
and simplify faithful reuse of the tested driver. Record duplicate comparator
lineage explicitly; do not multiply sample size or bootstrap them independently.
Same starting RNG states do not imply event-level identical CRNs after policies
diverge through action-dependent RNG consumption.

## New data and proposed limits

All P1-R1 demonstration/qualification data are now inspected development data.
Do not refit on them or call them independent confirmation. Proposed fresh
namespace: `P2-reference-prior-candidate-20261001-v1`; NOT yet collision-audited.
Proposal for environment ordinal groups: preflight0..11, prior qualification
12..17, training18..113 (32/block), test114..149 (12/block). Derive starts from
SHA256(namespace)[:12] big-endian shifted16 bits plus ordinal, then verify all
local historical manifests before any execution. Separate named neural,
sampling/shuffle and bootstrap streams must be locked; no seed selection.

Proposed new attempt budget, not authorization or transferred P1 allowance:

| Phase | Environment calls | Optimizer steps | Time limit |
| --- | ---: | ---: | --- |
| Real preflight including every clone |624 |0 |600s |
| Prior qualification,2 R4 episodes/block |312 |0 |600s |
|9 PPO continuations |14,976 |1,152 |600s/model |
|9 BC-CONTINUE controls |14,976 |1,152 |600s/model |
|396 final greedy evaluation episodes |20,592 |0 |120s/policy |
| Global including I/O/archive |51,480 |2,304 |21,600s |

No phases can borrow unused caps. Debit before every simulation/optimizer call,
including clones/preflight. One attempt; no automatic repair/relaunch or expanded
samples. Fresh output/archive/Dropbox subdirectories must be fixed in a new
effective config. Never route this through old P1/P1-R1 claim directories.

Prior qualification must check each representation for exact greedy R4 request,
normalized90%/remaining-class probabilities within8 float32 eps, positive
alternative support, finite zero residual/value and unchanged input parity.
In singleton states require probability1; report their frequency separately.
Use algebra plus observed request/probability checks, not a stochastic95% gate
that confuses finite-sample variation with a distribution mismatch. Any failure
closes the attempt. Equal initial greedy requests are not an outcome guarantee
for subsequently explored/trained policies.

## Analysis and launch gates

Retain P1 primary graph-PPO minus its own graph-FROZEN raw52-step cost; all
PPO/BC/R4/MDL-2 and self-only/flat contrasts, three block estimates,36 paired
world differences, descriptive df2 interval and within-block bootstrap. Retain
the prespecified all3-block gain/>=1% discussion threshold AND patient-loss,
completion and terminal-obligation checks. Do not change cost weights to pass.
Report raw components and sampling deviation/executed-flow coverage for each
training arm. No positive-result, clinical safety or publication guarantee.

Required before launch:
1. Explicit approval of the final bounded P2 scope, separate from this design.
2. New-profile serial runner, full phase/budget/checkpoint integration and
   independent raw verifier/aliases. Do not invoke the old initializer fitter.
3. Artificial PPO/BC optimizer-update/recovery/rollback acceptance for the new
   policy, including nonempty Adam state and whole-phase failure, plus compileall.
   Current design-only tests do not perform optimizer steps and do not satisfy
   this remaining update-boundary acceptance gate.
4. Complete fresh-stream collision audit, immutable prior evidence review,
   reference/source/runtime locks and local implementation/effective commits.
5. No duplicate process; then one budgeted real preflight and conditional pilot.

Readiness is an evidence gate, not a promised calendar time. The immediate next
work is integration/validation, not further reward or algorithm exploration.
After these gates and execution approval, the experiment can start in that same
work session; the6h ceiling is not a runtime prediction. Do not start a process
or recurring automation merely to satisfy a calendar target.
