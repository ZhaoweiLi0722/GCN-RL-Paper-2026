# Graph representation and online adaptation: research roadmap

Date: 2026-09-28. Status: proposed research plan, not a locked execution spec.

## 1. Decision and scope

Preserve the existing graph-aware controller paper while testing one focused
extension: can online DDPG updates learn a persistent change in the response to
resource allocation that a strong frozen controller cannot already handle?

The intended explanation is spatial representation plus temporal adaptation:
GCN represents dependencies between sites; online policy updates may adapt
decisions when the operating response changes. Both are hypotheses to isolate,
not an assumed additive gain or a guarantee of acceptance at EAAI.

Zhaowei requested this plan. It does not record Howard's approval, reopen any
closed formal campaign, or authorize an unspecified training search. Keep this
work on the local integration branch; no push, PR, or main merge. Before a new
campaign, resolve the operational choices below and record the execution
authorization and any required collaborator sign-offs truthfully.

**Immediate deliverable:** a mechanism specification and small, independently
auditable feasibility study, not another large DDPG run.

## 2. What we know, and what remains open

Use the [integrated evidence ledger](../../docs/team_updates/2026-09-28-integrated-evidence-and-next-study.md)
as the numerical source of record. Do not combine percentages across studies.

- Existing matched comparisons support a GCN encoder benefit in the tested
  setting. They do not establish universal graph superiority or isolate use of
  the correct edges without topology ablations.
- Offline distillation and online DDPG updates are different sources of value.
  Existing final-versus-frozen evidence does not establish a stable online gain.
- Our [local overtime screen](../2026-09-08-user-authorized-residual-screen/results.md)
  failed its material-benefit gate. It changed one action before a fixed
  continuation, so it is not a certificate that sequential control is optimal.
- The [P0 fixture](../2026-09-09-online-adaptation-mechanics/readout.md) shows that
  feedback can matter without any policy-parameter update. It is a software
  fixture, not evidence that online RL helps the manufacturing task.
- Howard's planner results motivate repeated-decision headroom analysis, but
  raw artifacts and observation parity still need auditing. Planner benefit is
  not evidence of online DDPG benefit.
- The [September 28 pilot](../2026-09-28-disruption-feasibility/readout.md) tested
  equipment removal and procurement-delay changes with fixed rules. It did not
  test unknown resource effectiveness, adaptive identification, MPC, or online
  neural updates. Its small rule gains do not eliminate the proposed scenario.

The bottleneck may lie in the task, measurement, information, controller, or
learning algorithm. Do not diagnose every negative result as a DDPG failure.

## 3. First priority: unknown persistent resource effectiveness

### Operational story

Several facilities share a limited pool of qualified flexible staff-hours.
Facility A develops a persistent staffing or support-service efficiency loss.
More effort at A may protect its urgent backlog, but the same effort may be
more productive at B. Moving all work to B congests B; using C introduces
transport and downstream delays. Resource commitments take time and changing
them incurs an operational cost.

The controller knows its commitments and observes the resulting public service
and queue measurements. It does not receive the latent efficiency factor or
the future time of a regime change. The change persists long enough for
observations to potentially improve decisions.

This combines four testable ingredients: actionable uncertainty, persistent
response changes, network spillovers, and delayed consequences. Adding random
noise alone is not the proposed contribution.

### Physical mechanism must be decided before implementation

The preferred candidate is allocation of qualified labor to a documented
controllable preparation or support-service bottleneck. Staff-hours can be
continuous and unused work can carry between decision intervals; individual
patients and completed products remain indivisible. Qualified routes, required
testing, release rules, biological processing times, and patient identity stay
fixed. Overtime cannot magically accelerate cell growth or relax quality rules.

This service mechanism is a proposal, not a verified fact about the current
simulator or PRM operations. Confirm which operation is labor-limited, its work
units, measurements, constraints, and calibration with Howard/domain input.
Do not invent a fluid production stage merely to make gradients convenient.

If the defensible mechanism is instead staffing-dependent availability of
discrete production slots, retain that discreteness and audit cumulative action
sensitivity. If useful action differences remain unidentifiable at the data
budget, close this DDPG channel rather than pretending it is continuous.

### Minimal action and observation contract

- Start with one resource-allocation channel. Hold the other routing and
  procurement rules common across arms; joint learning is a later ablation.
- Allocate a bounded shared effort budget across eligible sites, including an
  option not to purchase additional effort. Charge committed effort and any
  mobilization/switching cost consistently to every controller.
- Freeze the delay from commitment to usable service, site caps, conservation
  rules, and contractual limits before performance screening.
- Observe committed effort, eligible backlog, available resources, measured
  service, queues, receipts, and public history at stated times. Throughput
  alone is censored by demand, reagents, and downstream queues; an estimator
  must not interpret every idle interval as low efficiency.
- Expose no hidden response coefficient, future disruption calendar, future
  random tape, or patient detail unavailable to the other methods.
- Estimate the shared public belief from each controller's own observations.
  Same information rules do not mean identical realized trajectories after
  actions diverge. Use a shared estimator implementation, not oracle traces.

If a convex resource cost is used, justify its units and range operationally;
do not select its curvature or weight after observing which method wins. Report
cost components and a prespecified sensitivity range, retaining clinical weights.

## 4. Scenario portfolio and priority

| Priority | Scenario | Why study it | Main competing explanation | Scope decision |
| --- | --- | --- | --- | --- |
| 1 | Persistent unknown resource-to-service response | Experience can change the useful allocation rule | System identification plus MPC, or frozen history feedback, already solves it | Main new mechanism; first small study |
| 2 | Persistent supplier or transport delay-distribution change | Ordering/transfer decisions have delayed shortage consequences | Updated lead-time estimates plus an inventory rule suffice | Backup mechanism or later transfer test; not concurrent expansion |
| 3 | Reagent perishability and capacity reservation | Current actions consume future options | Sequential planning helps, but online weights add nothing | Add only if existing semantics and calibration support it |
| 4 | Coupled preparation, production and QC queues | Local utilization can conflict with network delivery | A joint scheduling heuristic or MPC is sufficient | Likely follow-up; too much new process modeling for the first iteration |

Inventory in this plan means shareable reagents/consumables, never
interchangeable patient-specific therapies. QC remains mandatory. Do not add
supplier choice or expediting to scenario 2 without explicitly implementing and
validating those new decisions.

Across the main mechanism, predeclare a small boundary map:

1. No change: a negative control for harmful or needless updates.
2. Announced persistent change: tests control/representation with no detection
   advantage; every method receives the same announcement.
3. Unannounced persistent change: the primary adaptation question.
4. Fast, uncorrelated fluctuation: a negative control for chasing noise.

Vary persistence and network coupling on fixed development ranges, not an
unbounded severity search. Reserve new trajectories and joint configurations
for locked testing. First compare interpolation within the offline training
range; label extrapolation separately. Do not train the frozen arm only on an
easy regime while giving the online arm a broader offline curriculum.

A sufficiently capable history-conditioned frozen policy can itself implement
an adaptive decision rule. This study therefore asks about the empirical value
of parameter updates at a realistic finite data/compute budget, not whether
online weight changes are mathematically necessary for adaptation.

## 5. Method: keep DDPG, reduce the adaptation burden

First test ordinary DDPG on the validated action interface. Do not switch
algorithms whenever a result is negative. Before adding a new loss, establish
that legal actions create repeatable outcomes and the critic can rank them.

A bounded candidate is a graph-conditioned residual policy with a small online
adaptation head. Preserve the useful offline GCN representation initially;
update the action head and critic from actual subsequent transitions. Compare
head-only updates with full-network updates as one prespecified development
ablation. This may reduce the data burden, but it is not an established fix.

Give the frozen fork exactly the same architecture, initialization, action
channel, history, belief input, and offline budget. Its parameters remain fixed;
its recurrent state and public belief may still change. If new controller
initialization requires additional offline training, give both forks that same
training and audit its cost. The old actor cannot simply be called trained for
a new action space that it never saw.

Use the original operational objective, with explicit horizon/discount and
terminal accounting. A short auxiliary target must not silently replace the
long-horizon question. New reward shaping, replay changes, and change detectors
are not all bundled into the first candidate. Select at most one alternative
update design on development data and freeze it before independent evaluation.

Do not use fresh simulator counterfactuals as if they were free observations
from operating the plant. A simulator-assisted variant is a separately labeled
method with query and model-access budgets, not ordinary interaction-only RL.

## 6. Required comparisons and attribution

### Belief adaptation versus policy updates

Run this core 2x2 with identical public information access and architecture:

| | Policy weights frozen | Policy weights updated online |
| --- | --- | --- |
| Fixed explicit response estimate | Fixed-estimate frozen policy | Fixed-estimate online DDPG |
| Public-history response estimate updated | Adaptive-estimate frozen policy | Adaptive-estimate online DDPG |

All cells retain the same permitted history; a recurrent frozen policy can
implicitly infer a change even in the fixed-explicit-estimate row. This design
isolates the explicit estimator and weight-update interventions, not every
possible form of inference. Use matched forks within each row, and common
initialization across rows where the information contract permits it.

The primary contrast is the right minus left cell in the second row. Also
include a strong domain-randomized, frozen history-aware policy, an adaptive
rule, and online system identification plus MPC. MDL-2 and MDL-3 remain useful
reference anchors, not the only challengers the proposed method must face.

Frozen versus online total-policy value includes exploration and its cost. For
diagnosis, a frozen arm with the same exploration mechanism helps distinguish
weight updates from merely trying different actions; do not claim identical
state/action trajectories once feedback diverges. Preserve deterministic
readout curves as secondary endpoints, not substitutes for deployment cost.

### Graph representation versus online updates

If the main adaptation gate passes, evaluate graph/flat x frozen/updated with
matched inputs, parameter budgets, offline data, online budgets, and seeds.
Measure graph benefit and online benefit separately. A synergy claim additionally
requires the graph-minus-flat online increment to improve, not just a best
GCN+online cell. Include edge-ablation/retraining controls that preserve feature
access and tensor shape; a test-time graph corruption alone is not sufficient.

### Budget ledger

Report offline data generation, optimization, online interactions, simulator
queries, planning computation, update time, inference latency, and hardware
separately. ID-MPC learns from its own public experience, with any nominal model
access declared. Show performance at comparable operating/compute budgets.
A privileged oracle is a diagnostic and is never an equal-information baseline.

Reset policy, optimizer, replay, recurrent state, and estimator at each
independent deployment world using the locked initial state. Allow adaptation
within that world. Reusing learned parameters across test worlds is a different
continual-learning protocol and would need its own ordering and reset rules.

## 7. Feasibility gates before a large training run

| Gate | Evidence needed | Failure response |
| --- | --- | --- |
| G0: physical and accounting validity | Conserved resources, legal actions, settled terminal obligations, defensible observation timing | Repair a demonstrated implementation defect in a new version; preserve old evidence |
| G1: consequential sequential decisions | Repeatable, meaningful improvement from a feasible repeated-decision comparator on a small faithful case; quantify what it beats | Do not train when even an optimistic diagnostic cannot find actionable room; a heuristic failure alone is inconclusive |
| G2: information and learnability | Public histories identify useful response changes within the available operating duration; independent action-ranking replication and uncertainty estimates | Close or revise the measurement mechanism once; no hidden-state shortcut |
| G3: incremental value beyond adaptation baselines | Adaptive heuristic, strong frozen history policy and ID-MPC leave a plausible unresolved decision problem | If they solve it, report the cheaper solution and bound the RL claim |
| G4: bounded DDPG development | Noncollapsed behavior, finite losses, critic ranking, useful updates, full cumulative cost including learning, paired multi-seed readout | One declared remediation for a diagnosed failure; no open-ended tuning |
| G5: independent confirmation | Frozen protocol, fresh worlds/regimes, adequate precision, constraint and cost results, auditable raw rows | Report positive, negative or inconclusive outcome without reopening the test set |

For small cases, use an exact solver or certified bound only if it solves the
stated restricted problem with the appropriate information. A feasible planner
shows achievable value; it does not certify optimality. A hindsight bound is
explicitly privileged. Report the distinction and any discretization gap.

Do not gate on an arbitrary percentage of positive states alone. Freeze a
minimum worthwhile cumulative benefit in operational units and a precision
target before screening; choose both from operating costs and the scientific
question, not a desired journal-sized improvement. The earlier 0.5% screen
threshold remains historical and is not retuned retroactively.

### Known prerequisites from the existing pilot

- Resolve idle-equipment overflow semantics. The first smoke exposed stock
  deletion through clipping; the completed synthetic pilot explicitly relaxed
  per-site idle storage, which is not an innocuous fix to the original model.
  Choose physically coherent retention, rejection, or queued arrival behavior
  for the new model, test it, and scope any historical sensitivity separately.
- Price and settle waiting patients, active production, outstanding orders,
  and staff commitments at the horizon. Do not gain by pushing liabilities
  beyond the last recorded step.
- Verify executed actions, not just real-valued actor outputs. Float work
  accumulation can coexist with discrete completions, but it does not guarantee
  a useful smooth value landscape or an accurate critic gradient.
- Keep the formal benchmark implementation/results unchanged. New process
  mechanics and their consequences belong to a versioned synthetic extension.

## 8. What counts as a stable online benefit?

Primary endpoint: paired cumulative operating cost from deployment through the
locked endpoint and settlement, including the transient cost of exploration
and adaptation. Report absolute units and a prespecified relative denominator.
Do not exclude an inconvenient early period or select the best checkpoint after
viewing independent test outcomes.

Report separately:

- Online-versus-matched-frozen benefit with uncertainty across training seeds
  and deployment worlds, plus results by prespecified regime.
- Patient loss, eligibility/completion outcomes, waiting tails, and failures;
  clinical noninferiority needs a written, justified margin, not a convenient
  numerical tolerance or a nonsignificant difference.
- Tail cost, time to recover, cumulative break-even time, and performance when
  there is no change or the change is too fast to learn.
- Benefit versus adaptive rules and ID-MPC, including total computation and
  decision latency. Do not change the primary goal to speed only after a loss.
- Graph-versus-flat differences and graph sensitivity, without treating all
  Monte Carlo replications as independent training seeds.

A positive headline needs an uncertainty interval excluding no benefit on the
locked primary contrast, an effect meeting the previously chosen practical
criterion, justified guardrails, and support across held-out conditions within
the claim's scope. Use a crossed paired analysis when worlds are shared across
training seeds. Predeclare handling of multiple scenarios and secondary claims.

Three development seeds can diagnose a large failure; they cannot establish
robustness. Start planning independent evaluation with at least five training
seeds and determine worlds/seeds from a development-based precision calculation.
Five is a floor for planning, not a guarantee of adequate power. If the budget
cannot resolve the target effect, the answer is inconclusive, not positive.

Hold back new seed families and process combinations before selection. Keep
historical formal seed 91100000 and all closed confirmation evidence untouched.
Disclose the development scenario search and retain every prespecified cell,
including negative ones. Selecting a promising mechanism is development, not
independent confirmation of that mechanism.

## 9. Two-week decision schedule

This is a conditional work plan, not a promise to finish training or submit in
fourteen days. Estimate wall time after the small benchmark; extend only for
an explicit scientific reason, not because the observed sign is disappointing.

| Window | Work and deliverable | Decision |
| --- | --- | --- |
| Days 1-2 | A short operational mechanism/observation contract; storage and horizon accounting design; evidence inventory and raw-data request for Howard | Agree the controllable stage, units, costs, measurements, calibration status and permissions |
| Days 3-4 | Small conservation/timing fixtures; repeated-decision headroom and public-information tests; first adaptive-rule and ID-MPC comparisons | Can the new channel produce and identify useful decisions? Stop early if clearly no |
| Days 5-7 | Critic/action-ranking diagnosis; freeze one DDPG candidate and one allowed development ablation; benchmark compute and predeclare endpoint | Decide whether neural development is justified; do not silently launch a large campaign |
| Days 8-10 | Bounded paired three-seed development, frozen/adaptive comparators, cumulative learning curves and failure analysis | Select at most one configuration or close the mechanism; never select on final test worlds |
| Days 11-14 | If justified and authorized, locked independent evaluation and graph attribution; otherwise complete a negative/inconclusive readout and publication revisions | Produce an evidence-backed paper decision and list remaining precision/calibration gaps |

Deliver a go/no-go memo by the end of the first small feasibility stage. If
the capacity mechanism fails, scenario 2 receives its own frozen specification
and bounded screen; it is not automatically compressed into the remaining days.
After two scientifically distinct failed mechanisms, prioritize the existing
paper rather than adding a third simulator redesign to chase significance.

## 10. Publication work that proceeds in parallel

The manuscript should not depend entirely on discovering an online gain.

1. Recover and audit Howard's raw rows, logs, configs, model hashes, and planner
   information access. Preserve his reports and label unverified claims.
2. Reconcile the executed objective, reward/discount, graph inputs, teacher
   generation, offline training and online update equations. Label each source
   of improvement accurately; imitation of planner actions is not online RL.
3. Correct the paired/crossed uncertainty calculation. Report cost components,
   seed-level variability, clinical tradeoffs, and effect sizes in useful units.
4. Complete the graph attribution evidence: matched flat, topology controls,
   size/topology generalization and transport-sensitivity tests where justified.
5. Document calibration, operating assumptions, and limitations of the
   synthetic environment. A reproducible synthetic benchmark is valuable but
   not equivalent to clinical or real-world manufacturing validation.
6. Package a traceable reproducibility bundle and a claim-to-evidence table.
   Keep old formal results, new development, and any new confirmation separate.

Target figures are an attribution matrix (graph x updates), a post-change
cumulative-cost curve including learning costs, a prespecified persistence x
coupling boundary map, and a performance/computation comparison. These figures
answer mechanisms and tradeoffs, not just show a favorable bar chart.

### Honest paper outcomes

| Observed outcome | Supported paper direction |
| --- | --- |
| GCN helps and online DDPG reliably improves matched frozen control | Graph-aware control with separately demonstrated online adaptation under specified operating changes |
| GCN helps; belief adaptation or frozen history feedback captures the new gain | Graph-aware learned control with explicit limits on the need for online weight updates |
| ID-MPC is best; learned policy offers a measured deployment-cost advantage | Evidence-backed quality/computation tradeoff, if operationally worthwhile; not an RL-dominance claim |
| No meaningful headroom or unresolved identifiability | Retain the GCN/offline story and report the tested boundary; do not claim global near-optimality |
| Calibration, safety-margin justification or inference remains inadequate | Further validation is needed; adding a positive RL mean does not make the manuscript submission-ready |

Changing the paper's framing must follow the evidence. Neither GCN benefit alone
nor a positive new RL experiment automatically supplies novelty, adequate
validation, or acceptance at EAAI.

## 11. Literature anchors and limits

These sources inform the design, not a claim that DDPG will work here. The
publisher/arXiv abstracts and available article preview were inspected; this
is a focused design scan, not a systematic literature review.

- [Wu et al., EAAI 2024: dynamic job-shop scheduling with uncertain processing time](https://doi.org/10.1016/j.engappai.2023.107790).
  The paper connects a specified engineering uncertainty to state/action/reward
  design and benchmark comparisons. It motivates concrete process modeling;
  its results do not establish online-weight value in our setting.
- [Kumar et al., RMA, RSS 2021](https://arxiv.org/abs/2107.04034).
  A base policy plus adaptation module is deployed without fine-tuning. This
  illustrates why adaptation and test-time policy-weight learning must not be
  used interchangeably; it is a robotics example, not PRM evidence.
- [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper/2021/hash/f514cec81cb148559cf475e7426eed5e-Abstract.html).
  Limited training runs create substantial uncertainty; interval estimates and
  performance profiles are preferable to relying on a single mean. Our exact
  crossed design still needs its own justified analysis.
- [Elsevier's EAAI description](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976).
  The journal emphasizes novel AI aspects in real engineering applications and
  replicable validation using public datasets. Public synthetic artifacts and
  operational calibration strengthen this study, but are not substitutes for
  a demonstrated engineering contribution or a guarantee of journal fit.

## 12. First implementation packet, after protocol approval

Keep the next implementation narrowly bounded: one documented resource-response
mechanism, one observation adapter, conservation/measurement tests, and a small
non-neural repeated-decision feasibility script. Add a separate config and
output root; do not modify historical evidence or launch formal confirmation.

The first readout must answer: what is controllable, what changed, what was
observable, which feasible actions mattered, what a capable adaptive baseline
already achieved, and how much operating experience was required. Only then
decide whether spending a larger budget on online DDPG is justified.

## 13. September 29 evidence checkpoint

The bounded software packets now answer part of that question. The static
service-effort fixture had no feedback value beyond its best open-loop
sequence. The coupled-queue fixture has feedback value, independently verified
from logged trees, but a prespecified non-neural baseline attains the restricted
optimum in every cell. Its arrival factor is redundant. Removing exact progress
still leaves completion-feedback value under the diagnostic's known finite
law; operational measurement availability and matched practical reduced-
information baselines remain open. See
`../2026-09-29-service-queue-boundary/readout.md`.

These are completed synthetic software checks, not completed clinical G0-G3
gates or authorization to train on a solved toy. The next learning study needs
a defensible, unresolved decision problem after accounting for a strong frozen
history policy and adaptive control. Do not add arbitrary noise or weaken
comparators merely to create an online-RL result. A computational advantage of
a fast learned controller is a separate possible claim that needs measured
latency, query budgets and matched performance, not just planner query counts.

In parallel, the original formal final/pretrain cost rows have been recovered
and audited. The crossed-inference sensitivity preserves favorable GCN cost
contrasts and null online attribution. See
`../2026-09-29-formal-crossed-audit/readout.md`. This advances the reproducibility
and statistical-reporting workstream without any new training or holdout
simulation, while the operational extension remains explicitly uncalibrated.
