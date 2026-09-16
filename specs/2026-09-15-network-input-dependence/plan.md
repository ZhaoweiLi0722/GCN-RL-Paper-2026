# Network demand and supply dependence: staged follow-up plan

Status: revised protocol candidate for review; not frozen or authorized to run.
Date: 2026-09-15.

**This belongs to the follow-up paper. Experimental work starts only after the
current paper's reporting audit and reproducibility freeze; it is not a
submission dependency.** This revision implements the review of the discussion
draft. Numerical choices below are proposed prospective rules, not thresholds
validated by results or amendments to any completed study.

## Scientific question

Does spatial or temporal dependence make specimen-routing counterfactual labels
more stable and predictable from the existing observations, and does that
support a graph-representation advantage? The sequence is Environment first,
then Algorithm, then Architecture. This study first tests the environment and
then, conditionally, fixed graph/flat rankers; it does not search architectures.

**Online RL attribution is deferred.** Altering demand does not remove the
observed mismatch between small continuous actor movements and integer-lot
execution. The evidence supports attenuation, not a theorem that every DDPG
gradient is zero. Nevertheless, another DDPG final-minus-frozen null would not
isolate the effect of input dependence. No online DDPG arm is included here.
An action-aligned online algorithm and its transfer tests would need a later,
separately approved protocol after the ranking gate.

This complements the [optimality-gap proposal](../2026-09-15-optimality-gap-measurement/plan.md):
the gap study asks how much opportunity exists; this study asks how opportunity
and policy performance depend on the structure of network inputs.

## Existing evidence and what would be new

The simulator already supports Poisson demand with changing rates, regional
demand shifts, persistent clustered demand shocks, and regional supplier
disruptions. The routing-primary scenarios include abrupt shift, gradual
regional drift, and compound regional stress.

An earlier, differently configured no-routing regional study reported 1.1337%
lower cost than MDL-2 and 1.1166% lower cost than matched flat control. Its
benefit was primarily from pretraining and it did not generalize reliably to
other regimes. **That policy used reagent and capacity edge-flow heads with
specimen transfer fixed at zero.** The present proposal concerns specimen-only
residuals, a different control mechanism. The 1.13% is not an expected effect
size or a power assumption for this study.

Sources: [regional results](../2026-07-25-regional-regime-network-afr/results.md),
`src/env/capacity_planning.py`, and
`experiments/configs/20_clinic_patient_condition_geo_compound_regional_stress.json`.

The new contribution would be a controlled dependence experiment, not simply
another higher-demand or higher-disruption setting.

## Distinguish three changes

| Change | Example | What it tests |
| --- | --- | --- |
| Marginal distribution | Poisson versus mean-matched negative-binomial demand | Sensitivity to burstiness; does not isolate graph value |
| Spatial dependence | Equally severe disruptions dispersed across clinics versus concentrated in connected regions | Whether location and connectivity change coordination value |
| Temporal dependence | Short isolated events versus persistent outages or moving demand hotspots | Whether history and delayed consequences improve decisions |

Negative-binomial demand is a candidate count model, not an empirically
validated PRM demand distribution. Regional dependence should be justified by
the application or labeled a synthetic mechanism study. More variability is
not inherently favorable to learning.

## Prior-oracle screen result (2026-09-15)

A no-training screen measured how much MDL-2 loses from its demand prior by
running the same decision rule with the simulator's true rate
([results](../2026-09-15-prior-oracle-gap-screen/results.md)). In the four
routing-primary scenarios the ceiling on "better distribution knowledge" is at
most 0.08%, and under the abrupt regime shift a clairvoyant rate is 2.4% worse
than the stale prior because it removes an accidental capacity hedge. Level
misspecification that persists for the whole horizon yields about 0.8% at 40%
misspecification. Consequences for this plan: the treatment must create value
through anticipation and cross-tier hedging, not through mean tracking; a
regularized adaptive heuristic and a hedging-aware heuristic belong in the
comparator list; and an anticipation-oracle arm (future rate at lead-time
horizon) should be added to Stage 1.

## Implementation prerequisites: new code, not configuration-only changes

1. **Add a topology-blind placement control.** Existing samplers choose the k
   geographically nearest clinics to a random start; k equal to clinic count
   shocks everybody. They do not implement uniform sampling of k distinct
   clinics. Add an explicit sampling mode under `src/env/`, shared by the base
   and patient environments. Call this control *random placement*, not strictly
   dispersed placement: a random draw can still contain adjacent clinics.
   Report the realized clustering and do not discard inconvenient draws.
2. **Decouple dependence from physical feasibility.** Add an explicit demand
   dependence graph/placement parameter. Keep specimen, reagent, and capacity
   feasibility edges unchanged. Construction and alignment diagnostics belong
   under `src/graph/`. For any alignment claim, compare against the benchmark's
   explicit `specimen_edges` list, not the default geographic graph. A future
   misalignment control changes only the dependence graph; it is not one of
   the first two cells.
3. **Implement matched exogenous input templates.** Generate input paths before
   policy execution, using dedicated RNG streams. Separate arrival templates
   from patient-attribute and supplier streams. Verify identical inputs across
   policies within a cell, aggregate matching across placement cells, patient
   identity invariants, and exact flag-off regression against existing behavior.
   Assert the realized scenario in the live patient environment, avoiding the
   earlier base-environment-only wiring and scenario-reconstruction defects.

These prerequisites require implementation and validation after approval; no
sampler, graph parameter, or environment code is implemented by this document.

## Proposed dependence cells and fixed information contract

Start with one input channel and a small prespecified spatial-by-temporal
design, rather than changing demand, supply, and costs together:

| | Short-lived pressure | Persistent pressure |
| --- | --- | --- |
| Random placement | Reference (R1) | Temporal-dependence comparison (R8) |
| Spatially clustered | Spatial-dependence comparison (C1) | Joint-dependence comparison (C8) |

The suffix denotes a one-epoch versus eight-epoch placement persistence. Start
with R1 and C1 only. R8/C8 remain conditional on the gate below. This is one
demand-placement factor; supplier generation is identical across cells.

For a routing-only controller, demand redistribution is the most direct first
candidate: patients accumulate in some regions while feasible destination
clinics retain capacity and reagents. Supplier dependence can be a separate
factor later; it may be uninformative if reagents are rarely binding.

Use the nominal 20-clinic routing contract, with a 52-epoch horizon, original
costs and physical resources, and the explicit routing edge list. Do not layer
the experiment onto a simultaneous abrupt demand-regime shift. Disable the
legacy demand-shock generator when applying the new templates.

**Multiplicative shocks cannot be moved indiscriminately across clinic tiers.**
The nominal rate tiers are clinics 0–4: 7.5; 5–9: 3.2; 10–14: 6.0; and
15–19: 3.5. A 2.6x multiplier therefore adds different absolute demand in each
tier. The matched-input design must use either tier-stratified assignments or
absolute-count templates; it may not assert that moving a multiplier preserves
the network load.

For this candidate choose **absolute-count redistribution templates**. Draw the
baseline count vector once per epoch and redistribute up to 20 existing
arrivals toward four designated clinics, at most five added per recipient.
Withdraw those same arrivals from the other clinics, without negative counts,
using a fixed deterministic proportional allocation and clinic-index tie-break.
If donors lack 20 arrivals, move only the available count and record the dose.
Apply the same donor-limited total dose to both placement treatments. Total
arrivals are unchanged exactly; the intervention changes their location, not
the mean workload. This is a synthetic relocation mechanism, not a fitted PRM
arrival law. No negative-binomial factor is included in this first study.
The 20-arrival dose is a single proposed screen setting, not a calibrated
clinical parameter; there is no outcome-driven amplitude sweep.

R1/C1 resample the four recipients every epoch; R8/C8 retain them for eight
epochs. Precompute placement schedules independently of outcomes. Balance
recipient exposure within the four stated tiers over the schedule pool and
report residual per-clinic imbalance. If exact balancing and clustered
placement cannot both be achieved, resolve and document the design before
policy rollouts; do not silently relax matching after seeing costs. The final
sampler, schedule construction, donor rule, and input hashes must pass the
input-only lock listed below.

Matching clinic marginals alone does not match aggregate variance: covariance
changes the distribution of total network load. Explicitly distinguish a
placement-only experiment with matched aggregate paths from a broader
dependence experiment where that aggregate change is part of the treatment.

Do not assume geographic adjacency causes demand or supplier shocks. Separate
the graph governing feasible transfers from any proposed referral/shared-supply
dependence graph. Justify their relationship and include a misalignment control
if alignment is part of the hypothesis.

## Required mechanism

The hypothesis is most plausible when observations contain predictive signals,
alternative clinics have accessible slack, and decisions have effects over the
relevant lead-time and deterioration horizon. For example, two candidate
destinations may have similar capacity now, but one faces rising neighboring
queues while the other is likely to remain available.

This is a hypothesis, not a prediction of a positive outcome. A system-wide
outage can remove every useful destination; extremely severe shocks can make
all policies fail similarly. Independent unobservable shocks can add noise
without adding exploitable information. Long transfers can make available slack
irrelevant to the patient in question.

The routing contract already includes a **12-epoch demand-history window and
a two-epoch forecast**. `src/models/gcn.py` also contains a temporal demand node
encoder. This study tests use of existing history, not the effect of newly
adding memory. Preserve identical observable history features for graph/flat
and adaptive comparators; do not silently enable a sequence encoder in one arm.
Freeze and audit the forecast source explicitly as `prior_estimate`. Historical
base configs can inherit `effective_rate`, so a two-epoch horizon alone does
not establish that the signal is a prior forecast. Hash the fully resolved
observation contract rather than relying on shorthand descriptions.

Eight-epoch persistence is a conservative proposed duration, not a necessary
condition for actionability. Formal specimen transit is one epoch, production
three, and return zero; reagent/capacity transfers can take up to three epochs.
Eight exceeds both the four-epoch specimen route-plus-production interval and
the six-epoch resource-transfer-plus-production interval. Verify these values
in the live environment. A fixed policy reacting to history still does not
establish online parameter adaptation.

## Comparisons and attribution

- MDL-2 plus an appropriately tuned adaptive/history-aware heuristic or
  information-matched rollout planner. Do not give the learned policy new
  predictive signals while leaving comparators deliberately uninformed.
- Matched GCN and flat controllers with the same observations, teacher,
  action authority, training distribution, compute budget, and selection rule.
- Frozen-pretrained graph and flat policies, if a later policy-comparison stage
  is approved. No final-versus-frozen online arm in the present screen.
- No-message-passing or shuffled-message-edge control to test topology use.
  Keep physical feasibility edges unchanged: changing those changes the
  operational problem, not just the representation.

Report graph-minus-flat differences and their changes across dependence
treatments as representation findings. Do not label a passing critic ranking
gate as an executed policy gain, or representation value as online-RL value.
A heuristic matching any eventual policy benefit is also informative.

Keep zero-shot robustness, retraining per regime, and genuine online adaptation
as separate experiments with separate claims. Generalization must use unused
shock locations/trajectories or regimes, not only fresh seeds for one fixed
hotspot schedule. Report all prespecified cells rather than choosing the most
favorable distribution after observing results.

## Staging, budget, and proposed prospective gates

### Stage 0: input-only implementation gate

Complete the prerequisites and freeze config/code hashes before scientific
rollouts. Audit 100 generated 52-epoch input schedules per initial cell, with
no policy execution: 10,400 input epochs total. Require exact network-arrival
matching within each paired epoch, nonnegative counts, the correct four
recipient clinics and persistence, and unchanged physical edges/observation
width. Report tier exposure, per-clinic moments, donor-limited doses, and
spatial clustering against the actual specimen-edge list.

Require C1's mean recipient-pair dependence-graph distance to be strictly below
R1's on this fixed input pool. If it is not, the manipulation failed; stop
before outcome collection. Report physical-edge alignment separately and make
no alignment claim without measured separation. No retries with different
seeds or graph construction to obtain favorable policy performance.

Before release, commit a manifest allocating mutually disjoint trajectory,
discovery, validation, fitting, and future confirmation RNG streams. Verify
against all earlier inventories, including formal seed 91100000. Seed numbers
are not assigned here without that collision audit. Record sampler details,
resolved forecast/history contract, and the source of any frozen checkpoints.
Pair exogenous worlds intentionally across placement treatments within a
stage; disjointness applies across stream roles and against earlier studies,
not between the paired policies or treatments.

### Stage 1: label stability first; no model training

Run R1 and C1 on six fresh MDL-2 trajectories per cell. Sample exactly eight
zero-indexed decision epochs per trajectory: `[4, 8, 12, 16, 24, 32, 40, 44]`,
giving **48 states per cell**. This is a new MDL-2 state distribution, not a
replication or reopening of G1's frozen-pretrain states.

At each state evaluate the five G1 legal specimen options: anchor and
`−0.10, −0.05, +0.05, +0.10`, with eight discovery and eight disjoint validation
continuations to the original episode end, then MDL-2 continuation. Never
sum these overlapping suffix improvements into an episode policy gain.

Budget: **48 × 5 × 16 = 3,840 continuation rollouts per cell; 7,680 for R1/C1**,
plus 12 state-generation episodes. Fixed allocation, no adaptive replication
extension and no additional horizon sweep. This uses more worlds per stream
than G1's 3/5 split and cannot attribute a difference from G1 solely to the
input process.

First audit execution: retain all sampled states, deduplicate identical
executed actions for ranking, and require at least two distinct actions in
at least 80% of states (39/48), with at least four actionable states in every
trajectory. Otherwise classify the cell as insufficient action coverage.
Report non-actionable states separately, not as perfect
agreement. On actionable states, require **discovery/validation best executed
action agreement ≥70% pooled and ≥60% within every trajectory**. Resolve exact
cost ties with a fixed action ordering; report tie and best-versus-second gaps.
No actor, critic, planner, or policy training precedes this gate.

The 70% primary threshold inherits G1's prospective criterion. The per-
trajectory floor and coverage requirement are new safeguards proposed here.
Report trajectory-clustered uncertainty; passing these screening rules does
not establish statistical certainty or a universal stable target.

From the same rows, next require **prospective state-dependence value ≥0.5%**
of mean anchor remaining-horizon cost: compare discovery-selected per-state
options with the discovery-selected best constant option, both scored on
validation. Include the anchor among constant options. Require positive value
in at least four of six trajectory groups. Compute this value across all 48
sampled states using the original five option identities; preserve options
that execute identically as equal-cost choices, rather than changing the
constant's meaning from state to state. Require that validation-mean completion
does not fall and patients lost and manufacturing ineligibility do not rise
versus the anchor for the selected per-state choices. Report these as local
screening guardrails, not demonstrated whole-policy clinical noninferiority.
This is a bounded five-option
screen, not a whole-policy optimality bound. The 0.5% is a proposed resource-
allocation threshold, not a clinical threshold or a multiple of a claimed
noise floor. Generic improvement over MDL-2 alone cannot pass this gate.

Labels obtained by cloning latent patient microstate are **conditional simulator
diagnostics**, even when future CRNs are disjoint. They must not be described as
information-matched achievable gains. The ranker below uses only observable
features; any deployable planner must pass the separate information audit.

### Stage 2: held-out-trajectory critic ranking

Only cells passing both Stage 1 gates enter ranking. Use the already collected
rows, with no additional simulator calls. Fit one fixed GCN and one parameter-
matched flat ranker, six leave-one-trajectory-out folds each. Fit on discovery
labels from the other five trajectories and evaluate solely against validation
labels on the unseen trajectory. Fit normalization on training folds only.

Use the G1 remaining-horizon normalized smooth-L1 plus pairwise-margin loss,
learning rate `3e-4`, 200 epochs, margin 0.10, material pair gap 250,000 cost
units, and normalization floor 1,000,000. Fix fresh model initialization seeds
and architecture configs before labels are opened. No HPO or checkpoint
selection; this is supervised critic fitting, not actor or online-RL training.

The fixed GCN feasibility gate requires pooled top-1 ≥40%, each fold top-1
≥30%, material-pair accuracy ≥65%, and top-1 gain ≥5 percentage points over a
fold-training-only best constant in at least four of six folds. Select that
constant by mean discovery cost in the five training trajectories, never by
the held-out labels. Evaluate top-1 on actionable states using the same
execution-equivalence classes and tie rule as Stage 1. Discovery-to-
validation top-1 loss must be ≤10 percentage points. Report material-pair
counts; absence of evaluable pairs is inconclusive, never a pass. The flat
ranker is a matched representation comparator, not a candidate selected to
replace the GCN after viewing results. These are prospective feasibility rules,
not a claim that failing models prove unlearnability.

Budget: **12 fits per passing cell, at most 24 for the initial pair**. The
constant must never be selected using held-out outcomes, correcting the
information advantage identified in the earlier ranking utility.

### Stage 3: decision on the persistent cells

Proceed to R8/C8 only if C1 passes coverage, label stability, state-dependence,
and ranking. R1 is always reported, regardless of whether it passes. Its
failure alone is not evidence that C1 is significantly superior. If C1 fails
any gate, stop this bounded study; do not move to persistence as an unplanned
rescue. This deliberately conservative ordering does **not** test or reject the
persistence hypothesis when the spatial-only arm fails.

If released, R8/C8 use fresh state-generation and continuation streams and the
same per-cell budgets and gates. A persistence claim requires explicit paired
R8-versus-C8 contrasts and short-versus-persistent contrasts whose uncertainty
respects independent stage samples. The latter are exploratory because the
persistent stage is released conditionally on C1. Crossing a threshold in one
cell alone is insufficient. All four completed cells remain in the report.

Maximum proposed scope, if all stages are later approved: **24 state-
generation episodes, 15,360 continuation rollouts, and 48 fixed ranker fits**.
There are **zero actor training runs, zero online RL runs, and zero full-episode
planner evaluations** in this budget. At a conservative 52 steps per
continuation, the rollout ceiling is 798,720 simulator steps plus 1,248
state-generation steps. Wall time is unknown until an execution-only smoke;
do not replace an elapsed-time overrun with fewer seeds or selective cells.

### Stage 4: reuse the optimality-gap work; no duplicate planner campaign

A successful screen permits a design review, not automatic policy training.
Reuse **Measurement A** of the
[optimality-gap proposal](../2026-09-15-optimality-gap-measurement/plan.md)
for any information-matched full-episode planner and its state-dependence
comparison against a tuned constant strategy. Inherit its warning that existing
lookahead utilities deepcopy latent patient state; reseeding future draws does
not remove privileged information. No new planner implementation or duplicate
rollout budget is commissioned here.

Any actor/graph-ablation comparison, new confirmation dataset, misalignment
cell, or online algorithm gets a separate specification and budget. The screens
can establish promising label/representation conditions, not an RL endpoint
gain. A stopped or inconclusive screen is a valid terminal result and does not
justify changing thresholds or input distributions after inspection.

## Release boundary

No existing scientific artifacts or simulator settings are changed by this
document. The [locked execution plan](../../docs/patient_indexed_specimen_routing_locked_execution_plan.md)
requires explicit approval, appended change control, and a committed plan update
before a changed experiment launches. This direction is not an authorization
to reopen the closed routing-primary campaign.

The motivation for testing relational structure follows the graph-network
inductive-bias literature; it does not guarantee superiority over flat models:
[Battaglia et al. (2018)](https://arxiv.org/abs/1806.01261).
