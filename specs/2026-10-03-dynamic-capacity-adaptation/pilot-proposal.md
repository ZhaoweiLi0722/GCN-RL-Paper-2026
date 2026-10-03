# Dynamic Capacity Adaptation: Single Pilot Decision

**DRAFT / PROPOSAL ONLY. `scientific_execution_authorized: false`.**
Prepared 2026-10-03 UTC on `codex/september-research-integration` from source
snapshot `aac47b090e2cb8ba6eba6b8b7bba9f5b19bcc2f3` and the coordinator's
current support-subclass implementation decision. This is not an executable
protocol, implemented campaign, fitted controller, calibration, or result.
The companion [pilot-proposal.json](pilot-proposal.json) carries the numerical
contract. Only these two files belong to this assignment.

## Decision and Evidence

Propose **one local, serial, development pilot**, not another qualification
campaign: initialize one capacity-policy family, then compare all six arms
across all three conditions. The question is whether online weight updates add
settled cost/patient value beyond competent public-history feedback, including
the price of exploration. Neither a positive result nor adequate learning at
this small budget is promised. No objective, severity, architecture, algorithm,
checkpoint, horizon or sample search is included.

[Prior evidence](prior-evidence.md) reports that continuous overtime and shared
commitments alone did not establish worthwhile held-out value; history feedback
can help without weight updates; and the latest conservative comparison had
zero incremental benefit over its frozen/BC controls. Those results motivate
this comparison, not a prediction of gain. The two-site fixture and closed
attempts stay closed. This does not replace a failed historical gate with a
claim that it passed, reopen Stage E, or reuse any formal holdout.

## Frozen Synthetic Operating Assumptions

All numbers below are **proposed synthetic assumptions, not calibrated data**.
Cost units are modeled units, not dollars. E1 operational evidence is absent.

| Item | Proposed value |
| --- | --- |
| Network | Four sites `S0..S3`; undirected transport/resource ring 01, 12, 23, 30 |
| Flexible staff eligibility | One separately declared pool qualified for all four sites; not a transport edge |
| Time | One epoch = one modeled week; control/enrollment epochs 0..47; settlement 48..63; host horizon 64 |
| Arrivals | Independent Poisson(1.5) per site per control epoch; zero during settlement; no demand change |
| Initial state | Zero patients/pipelines; reagents 8 and idle reactors 8 per site |
| Bounds | Specimens 100, reagents 100, idle reactors 40, purchases 6 per site; transfer limits 4 specimens, 4 reactors, 6 reagents |
| Common operations | Public-input MDL-2, lookahead 2, multiplier 1, sharing on; no learned routing/procurement; production starts most urgent ready eligible patients |
| Logistics | Specimen and resource transfer lead 1; one specimen transfer per patient; product return lead 0; fixed procurement lead 1, categorical probabilities [0,1] |
| Biology | Existing 5-epoch production; waiting specimen shelf life 6, finished shelf life 2; no culture/QC acceleration or new yield model |
| Support work | Uniform 4 work units per patient; preserve the host's urgency queue order and existing deterministic ties at the current material location; partial work follows the patient |
| Ordinary staff | 4 booked hours/site/epoch throughout all 64 epochs, including idle time; separate expense at 100 units/hour |
| Flexible staff | Site cap 4 hours; shared budget 8 hours/epoch; zero legal; radial projection; bookings at t mature at t+2 |
| Response | Private eta in {0.5,0.75,1,1.25,1.5}; usable work = eta * (ordinary hours + matured flexible hours); unused capacity expires |
| Incremental charge | 100*sum(h) + 5*sum(h^2) + 20*sum(abs(h-h_previous)); booked, not productive, hours charged |

The response applies to both staffing sources in this proposal. Ordinary
completions must not be attributed solely to flexible effort. This formula
requires explicit binding in the coordinator's producer, not inference from
receipt shape. No staff serves twice or moves to an unqualified site. In-transit
patients receive no support. Support work completes immediately before starts,
after arrivals/routing, without a second aging step; unfinished patients remain
in host queues and age/expire normally. `_start_production` must update its
production array **in place** to the actual integer starts before the parent's
reagent/reactor accounting. Reject every historical overtime flag.

Keep the existing patient dynamics: healthy/frail decay .01/.06, Weibull
shape/scale 1.5/8, post-shock multiplier 2, eligibility .75, one risk class with
probability/multiplier 1, no extra waiting-decay term. Keep patient-loss/expiry/
urgency weights 50000/40000/5000 and urgency margin .1. Existing base cost
coefficients remain 42174 purchase, 113.5 reagent holding, 86504.5 reagent
shortage, 14.4 reactor holding, 50273.6 reactor shortage, 600 specimen transfer,
500 reactor transfer and 200 reagent transfer. New support charges explicitly
extend this objective; there is no simultaneous reward search.

The common operations adapter may read only the public operational view. It
must not pass the private support environment to a heuristic that expects an
`env` object. Preserve resources, in-transit stock and identities; detect and
abort on unaccounted clipping rather than silently expanding storage bounds.

## Conditions and Information

Use the same three condition families for offline initialization and deployment:

1. **No change:** draw each site's response uniformly from the five-value set
   at reset and hold it for all 64 epochs. The coefficient is still unknown.
2. **Persistent unannounced change:** initial responses are all 1; draw change
   epoch uniformly from integers 16..28 inclusive, then randomly permute
   [0.5,0.75,1.25,1.5] across sites and retain it through settlement.
3. **Fast independent fluctuation:** each site's response is an independent
   uniform draw from the same five-value set every epoch, including settlement.

Controllers know these distributions and their equal offline mixture, but not
the condition label, realized coefficients, actual change epoch, future tape,
latent health/shock variables or true remaining support work. The time-to-cohort
closure is public; no change notification or exact scheduled-change feature is
given. Randomized change timing reduces, but does not eliminate, a learned
hazard cue. That limitation applies to every controller.

The proposed matched public view, still requiring implementation, gives all
arms current operational counts, inventory/pipelines, current survival
and material ages, public job entry/exit/completion identities, their own raw and
executed requests, ordinary/matured flexible hours, and end-of-epoch completion
receipts known at t+1. The current collector supplies only the base observation
plus capacity history, pending hours and ready counts; it does **not** supply
this additional patient-ID/survival/material-age API. Neither the estimator nor
MPC below can be claimed integrated until that public view is implemented.
Use an 8-receipt masked feature window and the same causal
response estimator. Histories can differ after different actions. Full receipts
are retained for restoration, never future observations. If work-progress
measurement is actually public in operations, a completion-only application
claim is invalid; a later progress-observed study needs a separate decision.

**One proposed estimator, not yet implemented:** a five-response-value interval
filter with **per-patient-ID residual-work intervals**, not a FIFO-head state.
Only new jobs start at [4,4]. Carry each ID's interval through priority changes;
transfer its interval with the patient, conservatively taking the source
hypotheses' convex hull into each destination hypothesis. A move is not a new
four-unit job. Propagate 25 old/new response pairs per site using transition
probabilities .95*I + .05/5, walking the actual public host urgency order and
retaining intervals for every partially served ID. Condition on public integer
completions, completed IDs and exits, merge compatible per-ID intervals by
convex hull, and normalize compatibility weights. An unchanged feasible point
interval has compatibility weight one, not a zero-width division. This loses
cross-patient correlations and is an approximate conservative filter, not an
exact posterior. Retain zero-completion inequalities and
backlog-limited inequalities; never divide completions by flexible hours or
interpret zero completions as zero efficiency. If all hypotheses become
incompatible, flag the event and restore the fixed uniform prior and [0,4]
intervals for affected unresolved IDs, without latent access or parameter
changes. Completed/retired IDs must not regain work. Export the five weights,
per-ID interval view, node summaries and count of such resets to every arm.
No added sensor
noise is used to manufacture an identification problem.

## Six Controllers and One Learner

| Arm | Fixed prospective implementation |
| --- | --- |
| `frozen_history` | End-of-offline GCN-DDPG actor, no deployment optimizer, updated public history/filter, deterministic action |
| `frozen_matched_exploration` | Tensor-identical frozen fork plus the online arm's exploration rule and paired noise innovations |
| `online_matched_fork` | Same actor/critic/targets/replay/optimizer initial state; same exploration rule; 40 actor and 40 critic steps per world |
| `adaptive_rule` | Delay-aware public backlog/urgency allocation using the shared filter, no neural fitting or model rollout |
| `id_mpc` | Same filter plus bounded 8-step receding-horizon approximate public-state planner, no private simulator clone |
| `fixed_allocation_reference` | Commit [2,2,2,2] every control epoch; same legacy operations; context only |

Choose a **fresh capacity GCN-DDPG**, with ID-MPC behavior-cloning warm start and
fixed offline actor/critic training, not the old specimen selector. DDPG fits
continuous priced hours and reuses replay/update-boundary concepts; distillation
gives a nonrandom starting policy, while subsequent learning is explicitly RL.
BC improvement itself is not RL gain. Two 32-wide ReLU graph layers use the
public resource ring with self loops and symmetric normalization; concatenate
the per-node 8-receipt history, filter features and operational view. Actor head
32-ReLU-1 emits raw hours `clip(2 + z, 0, 4)`, followed by radial projection.
Critic concatenates executed commitments to node features, uses the same width,
mean pooling and 32-ReLU-1 scalar head. Actor/critic encoders are separate.
No recurrent hidden state, batch normalization, residual specimen gate, or
representation search. Graph attribution is **not** tested by these six arms.

Use fixed feature divisors: counts 100, hours 4, work 4, ages 64, remaining
control time 48; survival, masks and filter weights unchanged. No statistics
from evaluation are fitted. Float32 neural arithmetic, float64 physical work.
Adam (.9,.999), epsilon 1e-8, no weight decay, gradient norm cap 10; actor LR
1e-4 and critic LR 3e-4 offline; both 1e-4 online. Discount 1, target Polyak
tau .005, batch 64, reward = negative full cost / 100000, without reward clipping
or shaping. The last control transition receives **all** settlement cost and
`done=true`; intermediate transitions bootstrap normally. No optimizer on tail
transitions; the already-budgeted last control update executes after settlement.

`adaptive_rule`: predict eligible workload at t+2 from public queue entries,
expected arrivals, the estimator's mean response and booked pipeline; subtract
ordinary and already committed expected service. Let positive remaining work
be d_i, urgency weight v_i = 1 + 2*(fraction of eligible jobs below survival .85).
Approximately minimize the four-variable constrained convex surrogate
`sum(v_i*(max(0,d_i-eta_hat_i*h_i)/4)^2 + .01*h_i + .0005*h_i^2
+ .002*abs(h_i-h_previous_i))`, with the same site/shared limits, using exactly
20 proximal-gradient iterations, step .1, initialized at previous commitments.
For gradient iterate y, compute the joint constrained proximal map
`h_i(lambda)=clip(previous_i + soft(y_i-lambda-previous_i, .0002),0,4)`;
use lambda=0 if feasible, otherwise 32 fixed bisection iterations on the shared
nonnegative multiplier until the sum is at most 8 (take the feasible endpoint).
This internal solver is not the native radial action transform; its feasible
output is a fixed point of that transform. Twenty outer iterations are not an optimality guarantee.
Those coefficients are fixed allocation-rule assumptions, not
substitute reporting rewards. No simulator calls, fitting or tuning.

`id_mpc`: at each control epoch evaluate 16 sequences: eight constant vectors
(zero, equal, adaptive-rule, previous, and four site-focused vectors with
4 hours at one site and 4/3 at each other site), plus the eight variants that
switch after two predicted epochs to the feedback adaptive rule. All forecasts
use only public mean arrivals and three estimator summaries (10th/50th/90th
response quantiles, weights .25/.5/.25), not true latent states or future CRNs.
Use predicted host-priority support with per-ID carried intervals, ready queues,
five-stage production and the common
operations, including patient loss/expiry, labor and inventory accounting.
The predictive patient decay approximation uses current survival and the fixed
population mean rate .035, not the patient's private health/shock draw.
At planning-horizon end value each still-live patient at fixed 50000*(1-current
predicted survival), add committed labor already payable, and give zero inventory
salvage. This approximate continuation value is for planning only, never the
evaluation settlement. Choose minimum predicted cost, tie by candidate order;
execute only the first commitment. All candidates obey known cohort closure.
Finite candidates and an approximate model do not imply MPC optimality.

Exploration is independent Gaussian N(0,.25^2) hours/site each control decision,
clipped to nonnegative/site caps before radial scaling. Frozen-exploratory and
online share the same innovations in each paired world, not necessarily the
same executed actions. No annealing, surprise trigger or extra probe episodes.

## Initialization, Isolation and Evaluation

Three independent model seeds: **520260301, 520260302, 520260303**. Each block:

1. Collect 12 ID-MPC teacher trajectories: 4 worlds per condition, 48 control
   decisions plus 16 settlement steps each. Train actor BC for 256 updates on
   its 576 public decision/action rows. Targets are executed legal hours.
2. Fit the fresh critic for 256 TD updates on the same settled transition set;
   hold actor fixed and soft-update critic target. Then collect 12 fresh learner
   trajectories (again 4/condition) with the fixed exploration rule and perform
   576 DDPG update pairs, one per newly available control transition. Defer each
   trajectory's last update until its settlement return is attached.
3. Seal the **fixed final** state and the 1152 teacher+learner transitions. No
   best-checkpoint selection, validation selection, further teacher query or
   offline retraining. Retain all imitation and training diagnostics, including
   failure to match the teacher. Competence is a design aim, not verified fact.
4. Evaluate 3 conditions x 4 new worlds x 6 arms per block. Reset patient state,
   actor/critic/target tensors, optimizer moments, replay and filter/history at
   every independent world. Frozen arms never call an optimizer. Online replay
   mixes 32 sealed offline rows with 32 rows sampled with replacement from that
   world's observed control transitions; it updates after receipt 9 through 48
   inclusive (40 pairs). Last update waits for settlement and cannot improve
   already observed performance. No cross-world accumulation or test selection.

For block b=0..2, condition c=0..2 and replicate j=0..3, teacher world seed is
62600000+1000*b+10*c+j; learner world seed is 62604000+1000*b+10*c+j;
evaluation world seed is 62610000+1000*b+100*c+j. All are native integers, not
decimal strings. Each phase cycles condition c=episode_index mod 3 and
j=floor(episode_index/3). Separate keyed RNG substreams for demand, patient
identity attributes, response/change, exploration and replay are derived from
SHA256(canonical JSON of namespace, phase, b,c,j,purpose); first 8 digest bytes,
big endian. Namespace `dynamic-capacity-pilot-v1`; arrays of immutable arrival
and identity-indexed patient draws are generated once per world and reused
across arms. Action-dependent RNG draw order must not destroy CRN pairing.
Training/prediction never accesses evaluation tapes or historical checkpoints.

There is no separate scientific validation or selection phase. The first teacher
trajectory doubles as the native schema/accounting preflight **inside** the
stated budget and is retained, not rerun. Reuse fake-record tests; no extra
toy-fitting campaign. This pilot is development evidence, not confirmation.

## Closed-Cohort Settlement and Liability

Host horizon is 64, not 48. Enroll only at t=0..47, with the last new cohort
appearing at the end of epoch 47. Run all 16 tail epochs for every arm: charge
ordinary staff and normal operating expenses even when idle. New flexible
bookings are zero from t=48; charge the switch to zero once, and honor bookings
from t=46/47 when they mature at 48/49. No refund for idle prepaid hours.
No tail neural action, tail-transition optimizer or planner query is allowed.
The final control-transition update is deferred until all tail costs are known;
it remains in the stated 40 online/576 offline pairs and buys no extra update.

Use a common deterministic drain policy: no new specimen/resource transfers;
allow outstanding transfers, support, production and returns to finish; retain
normal survival and expiry. Through t=60 purchase only the capped positive
shortfall between the site's still-live support/ready patients and on-hand plus
on-order reagents; no safety stock. Zero purchases thereafter. Purchases are
charged once at booking and received normally. No new arrivals, horizon reset,
patient deletion, age reset, free outsourcing or numerical terminal bonus.

Sum costs over all 64 epochs, including all losses/expired material, support,
procurement and holding expenses. Support work consumed then abandoned on
death/expiry is recorded separately, not refunded or silently dropped. Remaining
work conservation is initial plus newly required work = all performed work +
retired unperformed work + active remaining work; performed work on lost patients
is a reported subset of performed work, not counted twice. Remaining
reagents and returned reactors are conserved in a closing inventory ledger with
zero salvage, no assumed resale and no continuing holding cost after declared
closure. This is finite closed-cohort accounting, not an infinite-horizon cost.
Do not count support or manufacturing starts as treatments.

At t=64 require every enrolled ID to be terminal delivered or lost, every
pending staff/order/transfer/production/return obligation empty, and identity,
work and resource conservation reconciled. The 16-epoch tail is a proposed
allowance motivated by shelf life 6, production 5 and logistics lead 1, **not
a verified drainage bound**. Do not forcibly label survivors lost to make it pass.
For **any** live support/ready/in-transit/in-production/finished patient, pending
hours/order/transfer/return, unbooked expense, lost identity, orphan work or stock
discrepancy: save individual liability records and already-incurred costs, mark
`settled=false`, end the attempt, and withhold the full-cost comparison claim.
No extra tail, imputed zero liability, result-dependent liquidation penalty,
automatic retry or selective exclusion. An engineering failure is not an RL null.

## One Finite Budget

These are proposed caps for later specific approval; **current actual scientific
allowance is zero**. Construction/reset counts are separately charged. One
native step advances the support wrapper and its host exactly once, not two
independent transitions. Approximate planner epochs are not native env steps.

| Phase | Trajectories | Control steps | Tail steps | Native steps | Actor optimizer | Critic optimizer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Teacher collection + BC/critic warmup, 3 blocks | 36 | 1728 | 576 | 2304 | 768 | 768 |
| Offline DDPG, 3 blocks | 36 | 1728 | 576 | 2304 | 1728 | 1728 |
| Deployment, 3x3x4x6 | 216 | 10368 | 3456 | 13824 | 1440 | 1440 |
| **Total** | **288** | **13824** | **4608** | **18432** | **3936** | **3936** |

Total optimizer calls 7872, consisting of 768 BC actor calls, 768 critic-warmup
calls and 3168 DDPG pairs of exactly one actor plus one critic call each:
768 + 768 + 2*3168 = 7872. Charge the actual `optimizer.step()` invocation before
dispatch, including a failed/interrupted invocation; target copies/Polyak
updates are not optimizer calls. No extra critic steps or optimizer-backed
planner are allowed. Batch-example presentations are 503808 across
those optimizer calls. No optimizer is hidden in the estimator or planner.
Neural forward cap is **25824 module calls**: 6912 behavior actor calls,
768 BC actor calls, 3x768 critic-warmup calls, and 5x3168 DDPG-pair calls
(target actor, target critic, online critic, actor, actor-loss critic).
Batch size is at most 64; count each dispatch before execution, including failed
calls. There are no uncounted qualification or checkpoint-selection forwards.
At most 288 native constructions and 288 construction-triggered resets; no
additional explicit reset or simulator clone: 19008 charged native operations
if constructions, resets and steps are counted together. No historical model
loads; exactly three new initial model pairs, 3x3x12=108 learned evaluation arm
restorations from the corresponding new seal (including frozen arms).

ID-MPC is queried for 1728 teacher and 1728 evaluation decisions, total 3456;
16 candidates x 3 summaries x 8 predictive epochs = 384 queries/decision,
**1327104 approximate model epochs** maximum (165888 candidate rollouts).
These queries have their own counter and time charge. No patient-env rollouts,
Monte Carlo futures, privileged scenario oracle or uncharged planner calls.
One estimator update per native receipt: 18432 updates x 4 sites x 25 response
pairs = **1843200** site-hypothesis transitions at most; each transition includes
the full per-ID queue traversal, not just one head. Per-ID arithmetic and
transfer bookkeeping consume the same phase time/RSS budgets, not native steps
or uncounted planner rollouts. Tail estimator updates may
be logged but cannot produce control or optimizer work. Unused budgets expire.

| Wall-clock owner, includes its normal recording/I/O | Hard seconds |
| --- | ---: |
| Admission, frozen-lock checks, schema binding | 300 |
| All initialization/training and seals | 2100 |
| All six-arm evaluation and settlement | 2100 |
| Raw analysis, summaries, local archive and verification | 600 |
| Reserved failure closure/flush/process shutdown | 300 |
| **Global wall clock** | **5400 (90 minutes)** |

One CPU worker, four compute threads, no concurrent scientific jobs, no MPS
fallback or remote execution. No narrow per-world watchdog; phase/global caps
include serialization and recording, not just stepping. Timings are conservative
planning assumptions, **not measured throughput**. Stop launching scientific
work on phase/global/query/storage/memory exhaustion. RSS cap 4 GiB, raw artifacts
1 GiB, one local archive up to 1 GiB, combined 2 GiB, at most 2000 files. Stream
per-trajectory records, not per-step files; flush every 8 epochs and at boundaries.
Save initial/final seals, actual actions, receipts, costs, liabilities, seed/tape
identifiers and counters. Preserve failure artifacts, stderr and consumed calls.
One new persistent local run directory only; no Dropbox, remote or automation.

The single attempt is consumed on admission to scientific work, including a
failed first constructor. No restart, resume, replacement seed, omitted arm,
reduced sample, top-up budget or follow-on experiment is implied. If failure
prevents archival completion, retain originals and report the precise unverified
archive state. No cap is silently relaxed to obtain a comparison.

## Endpoints, Paired Unit and Decision

Primary deployment endpoint: total settled 64-epoch cost per world. For both
frozen contrasts define saving = C_frozen - C_online; report absolute units and
100*saving/C_frozen, including exploration and tail costs. Show all three
conditions separately, plus the prespecified equal-condition mean. Online vs
adaptive rule and ID-MPC are separate stronger comparisons; fixed allocation
alone cannot establish online value. Offline training costs/computation are
reported separately, not disguised as free deployment initialization.

Report losses and delivered patients/counts/rates; mean and p90 support wait,
total wait and turnaround; patient-time until loss/delivery for **all** enrolled
patients alongside survivor-conditional metrics; ready/support backlog and
unfinished-work/liability counts at t=48 and t=64; ordinary/flexible booked,
matured, productive and idle effort; labor/switching/base/patient costs; requested
vs executed action distances; filter resets; exploration cost (frozen-exploratory
minus deterministic frozen); and per-phase wall, model queries and updates.

Descriptive recovery lag in the persistent cell: first post-change boundary at
which the trailing four-epoch online-minus-deterministic-frozen incurred cost
sum is <=0 and online support backlog is no larger for four consecutive
boundaries. Use no terminal-lumped return in this diagnostic; if absent by t=48,
right-censor at 48-change_epoch. No recovery claim for no-change/fast cells and
no privileged recovery-triggered action. Full settled cost remains primary.

The paired unit is one (training block, condition, exogenous world) with all six
reset arms: **36 units total, 12 per condition, nested in 3 independent training
seeds**. Decisions, patients, arms and planner queries are not independent
replicates. Average four paired differences within block/condition first; report
all three block means and all twelve paired differences per condition. Use 2000
hierarchical bootstrap resamples (seed 62690001), resampling the three training
blocks and then four paired worlds within each selected block/condition;
preserve all arm contrasts. Percentile 2.5/97.5 bounds are descriptive with only
three training seeds, not confirmatory CIs, p-values or clinical noninferiority.

One integrated readout, no intermediate efficacy gate: classify as a **candidate
online adaptation signal** only if persistent-cell mean cost is lower versus
both frozen forks in every training block, the descriptive saving intervals for
both exclude zero, and mean extra patient loss is <=0 versus both. Always show
no-change/fast harms, adverse tails and adaptive/MPC contrasts; failure of this
screen is no reliable online benefit **at this budget**, not equivalence or a
universal RL verdict. Lower cost with extra losses is an explicit trade-off,
not a successful clinical result. If adaptive control matches/beats online,
feedback may suffice here. Weak initializer imitation or offline learning is
reported as limited competence, never used to justify more fitting. These are
interpretation rules, not authority to launch another study or guarantee gain.

## Exact Remaining Integration and Single Approval Boundary

The existing interface provides legal effort and typed receipt/history checks;
the old service fixture exposes work and explicitly has an unsettled terminal
ledger. Its `PublicServiceResponseEstimator` consumes delivered-work receipts
and **cannot** be used unchanged under completion-only measurement. The
coordinator is implementing an unregistered patient-support subclass, ledger
and native adapter; this proposal does not claim they or the following are done:

1. Bind the four-site synthetic config, eta*(ordinary+flexible) formula,
   in-place start counts, ordinary expense, host-priority support with per-ID
   transfer carry, and closed-cohort
   arrivals to the native producer. Verify old overtime is rejected and all
   resource/patient aging rules remain once-per-epoch with fake records.
2. Implement the missing public individual identity/survival/material-age view
   beyond the delivered base observation/history/pending/ready counts, and
   whitelist common MDL-2
   inputs; connect ring logistics and separate staff eligibility; implement the
   shared per-ID interval filter with variable urgency order and transfers,
   without actual remaining-work or latent leakage.
3. Implement capacity-specific GCN actor/critic, BC/TD/DDPG collector and typed
   replay; use legal executed hours, sealed terminal cost and optimizer/noise
   restoration. Historical specimen request transforms are not reusable here.
4. Implement all six arms, the exact adaptive-rule solver and the 16-sequence
   ID-MPC approximation; the existing work-receipt EWMA is not this controller.
5. Implement fixed 48+16 collection/settlement, zero-arrival boundary, drain
   purchases, patient/work/resource/expense reconciliation and failure cases.
   A liability ledger alone is not automatic settlement.
6. Implement one serial executor, counters for every construction/reset/step,
   query/optimizer and I/O, CRN identity tapes, world-local restores, watchdog,
   raw comparator/bootstrap and local preservation. Reuse existing fake tests;
   include actual persisted schemas with fake loaders. No extra fitting gate.
7. Reconcile source/config against this packet, freeze exact source/runtime/
   input/seed locks, and record one specific prospective execution approval
   and committed change-control entry **outside this draft**. This task does
   not authorize that commit or any scientific admission.

Unsettled assumptions requiring that one decision are physical support-work
validity; staff eligibility and eta acting on ordinary as well as flexible
hours; completion-only observability; synthetic demand, workload, costs and
response persistence; unchanged common-controller competence; sufficient offline
learning and 16-epoch drainage; and achievable local runtime/storage. Engineering
mismatch must be resolved transparently before locking, not silently substituted.
No repeated approval chain for routine phases once the complete later packet is
approved. Until then: zero scientific loads/forwards, constructors, simulation,
training and planner queries. No commits, remote actions or automation here.
