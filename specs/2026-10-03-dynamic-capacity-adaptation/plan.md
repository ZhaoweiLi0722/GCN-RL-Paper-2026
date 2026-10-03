# Dynamic Capacity Adaptation: A Different Question

Prepared 2026-10-03 UTC, following Zhaowei's instruction to proceed if this
direction could help establish RL value. Direction and local implementation
preparation are authorized; a complete numerical proposal now exists but no
frozen executable scientific packet or numerical execution approval exists.
No Howard approval, site calibration, positive result or publication guarantee
is inferred. The closed two-round attempt stays closed.

## Why This Is Worth Testing, Not Already Proven

The completed two-round comparison has no incremental action, cost or patient
benefit over its own frozen policy or BC. It does not prove global optimality,
universal RL failure or an incorrect reward. Its inherited MDL-2 advantage must
not be relabeled as RL training benefit.

The proposed new uncertainty is the persistent response of a controllable
support operation to allocated qualified staff-hours. The same investment can
produce a different amount of usable support work after a staffing/productivity
change. Scarce effort, commitment delay and downstream congestion create a
possible sequential allocation problem. Experience might help, but fixed
history-conditioned feedback or ID-MPC may already capture the entire gain.

Dynamic capacity alone is not novel evidence here. Prior overtime, shared
commitment, disruption and service-queue attempts already exist. Their actual
results and limitations are in [prior-evidence.md](prior-evidence.md). In
particular, do not rerun the solved two-site queue as a new positive study.
The later completion-only S2/S3 results also failed to establish reproducible
identification benefit or stable remaining headroom:
[S2](../2026-09-29-completion-control-screen/readout.md),
[S3](../2026-09-29-completion-action-ranking/readout.md).

The hypothesis is useful because it can distinguish three paper outcomes:
RL training adds value; feedback/system identification suffices; or this channel
has no worthwhile remaining value at the declared budget. All are reportable.

## Single Proposed Mechanism

- Use a new versioned patient-coupled **preparation/support-work layer**, not a
  change to cell growth, manufacturing duration, QC, release or patient identity.
  Its physical validity and coefficient calibration are not established. Any
  initial study is explicitly a synthetic extension, separate from Stage E.
- Carry fractional unfinished support work across intervals. Release a patient
  to the existing production queue only when its own work is complete. Patients,
  batches and treatment completions remain indivisible. An interface gradient
  is not a guarantee of smooth clinical value or useful DDPG gradients.
- Allocate a bounded, priced shared staff-hour pool with per-site limits and
  the option not to spend. Commitments arrive after a declared delay. Preserve
  the existing radial shared-budget convention and charge booked hours even
  if subsequently idle; record raw requests and executed commitments separately.
- Let site-specific work-per-hour effectiveness change persistently, with its
  realized value and future change time private to the evaluator. Do not add
  simultaneous supplier, demand, reward or architecture searches.
- Completion-only receipts are the conservative proposed measurement model:
  own commitments, applied hours, eligible jobs and completion events. Zero
  completions are censored evidence, not zero productivity. If actual work
  progress is operationally public, it must be available to **all** comparators;
  hiding it to manufacture a learning task is not acceptable.
- Other routing/procurement/production rules remain common across arms. Shared
  staff eligibility is not the specimen-transport adjacency. Declare that
  relation separately; do not silently reinterpret the old graph or claim a
  graph contribution without a matched representation comparison.
- Freeze the workload, normal staffing, response range/persistence, all labor
  charges and terminal settlement rule before results. New support charges
  extend the accounting model; do not pretend the objective is literally the
  unchanged historical reward. Existing patient-cost weights are not tuned.

## Comparison That Must Be Delivered

Use one newly initialized capacity policy; the old specimen selector was never
trained for this action space and is not a competent same-start initializer.
Offline initialization covers the declared operating range for all learned
forks, including changed response values. Reset weights, optimizer, replay and
public-history state between independent deployment worlds.

| Controller | Public feedback changes? | Policy weights change in deployment? | Purpose |
| --- | --- | --- | --- |
| Strong frozen history-aware policy | Yes | No | Primary competent reference |
| Its frozen fork with the same exploration mechanism | Yes | No | Separate exploration from weight updates |
| Tensor-identical online-updated fork | Yes | Yes | Test incremental online updates |
| Adaptive allocation rule | Yes | No neural fitting | Test whether simple feedback suffices |
| Online identification plus receding-horizon control | Yes | No neural fitting | Strong conventional challenger |
| Fixed allocation / common legacy controller | As available | No | Context only, not sole winning benchmark |

DDPG remains a reasonable *candidate* for genuine continuous effort, not a
required winner. Reuse the typed replay and update-boundary machinery, but do
not directly reuse a historical residual/gate request transform that encodes a
different action. Choose and freeze one clean learner before a single pilot;
no concurrent PPO/SAC/TD3 search. Public history and the same estimator are
available to frozen and updated forks. Compare cumulative deployment cost
including exploration, not just the best checkpoint or late post-change slice.

The required condition set is no change, persistent unannounced change and
fast independent fluctuation. Do not weaken the frozen offline distribution or
give only RL a change notification. Pair exogenous worlds and reset each arm;
after actions diverge, information rules remain matched, not realized histories.
Predeclare change timing without making it a perfectly predictable clock cue.

Report full settled operating cost, patient loss/completions, waiting/turnaround,
tail liabilities, action changes, adaptation cost and recovery time. Report
offline training, online interaction, planner model queries and wall time
separately. Do not call planner queries free or present privileged foresight
as an equal-information comparator. Test outcomes cannot select cost weights,
stopping checkpoints, horizons or additional samples.

An online claim needs stable improvement over both frozen forks, not merely
over MDL-2 or no spending. Superiority over adaptive control is a separate
stronger comparison. Graph x updates and topology controls follow only if a
first adaptation effect warrants their separately budgeted study. No clinical
noninferiority claim without a justified prospective margin; no equivalence
claim from a null or a few seeds. Development and confirmation remain distinct.

## Finite Delivery Chain

1. **Delivered:** versioned legal-effort and causal public-history interface in
   `src/rl/capacity_adaptation_interface.py`. It reuses `ServiceEffortConfig`,
   preserves float requests, supports zero purchase, validates commitment delay
   and booked costs, and restores receipt history at a boundary. Artificial
   gradient/receipt tests use no optimizer or environment. No production
   collector or patient-layer integration is claimed.
2. **Delivered in source/fake integration:** the additive patient support
   adapter and matched baseline interface, using fake patient records first.
   Preserve survival aging, reagent/reactor conservation, job identity and
   complete terminal obligations. Reuse old service-work machinery where its
   semantics fit; keep old sources/results unchanged. Receipts separately expose
   ordinary staffing and matured flexible hours. Do not attribute base-staff
   completions to flexible effort or drop ordinary staffing from the accounting.
   The producer must bind both to actual resource availability; receipt-shape
   validation alone is not physical verification.
3. **Decision-ready proposal, not an executable campaign:** prospectively specify
   synthetic operating assumptions, one initializer/learner and six comparators. Specify all
   seeds, condition cells, initialization/update/interaction/model-query caps,
   preflight and settlement calls, wall-clock/I/O limits, endpoints and one-
   attempt failure rule in one complete numeric package. Include the planned
   direct training comparison; do not insert repeated toy-fitting gates or
   a succession of action-ranking campaigns.
   `pilot-proposal.{md,json}` and the fake-only serial campaign now provide this
   decision. Complete real controller/estimator/settlement/recording bindings to
   the approved scope before freezing; fake dispatch tokens are not algorithms.
4. **Only after specific execution approval, real integration and source/input/runtime locks:**
   run the one end-to-end pilot, analyze all prespecified worlds and preserve
   its outputs. Routine stages need no additional permission. Failure or a
   null result does not authorize a retry or a new scenario/severity/weight.

The02:35heartbeat ends this finite chain at fake integration and the numerical
decision. Do not delay that decision with another indefinite preparation chain.
The current permission covers steps1-3, not step4. Missing E1 operating data
remain missing; synthetic parameters must be labeled as assumptions. New
modeling assumptions and the full finite execution budget go into one user
decision, not repeated routine-step questions. No actual trial is running.

## Reward Decision

Do not change patient-loss weights now: the observed null does not diagnose a
reward bug. First implement physically meaningful new effort and its mandatory
accounting. A potential-shaping experiment is not automatically new information
or a changed optimum. If a later pilot produces a cost/patient trade-off, report
it; do not retune the objective after looking at test outcomes. A genuinely
different service-versus-cost objective is its own prospective decision.

## Research Anchors

- [RMA, RSS 2021](https://www.roboticsproceedings.org/rss17/p011.pdf) demonstrates
  deployment adaptation without policy fine-tuning in robotics. It motivates
  the strong frozen-history comparator, not a PRM or online-weight benefit.
- [Agarwal et al., NeurIPS 2021](https://proceedings.neurips.cc/paper_files/paper/2021/file/f514cec81cb148559cf475e7426eed5e-Paper.pdf)
  emphasizes uncertainty with limited RL runs. Independent training seeds and
  paired operating worlds must not be pooled into an inflated sample size.

No remote publication, messaging, Dropbox export, holdout access, Stage E
reopening, historical rerun or Howard sign-off is authorized by this plan.
