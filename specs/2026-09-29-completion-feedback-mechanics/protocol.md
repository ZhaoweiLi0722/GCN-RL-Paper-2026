# S1: completion-feedback support queues, bounded mechanics only

## Authorization and question

After the cost-sensitivity readout, Zhaowei requested continuation. Implement
the next design as an isolated, explicitly hypothetical service mechanism and
bounded engineering check. Do not interpret this as an answer supplying site
data, Howard sign-off, approval for a patient experiment or a neural campaign.
The twelve E1 operational fields remain unfilled. Source, config and protocol
are committed before the recorded check; all previous evidence is immutable.

The old fixture allowed exact progress to identify a deterministic response.
Here the controller must infer response from noisy, interval-observed completion
events. This changes the observation/problem definition; it is not a repaired
version of the old result and cannot be pooled with it.

## Hypothetical physical and observation contract

Two sites have site-local divisible support availability. The sum is constrained
by a central purchasing budget, not by teleporting a shared employee. At most
one FIFO support task per site receives effort in one interval. A request h_i
in [0,1] is purchased now and delivered next interval; sum h_i <= 1. All requests
are accepted and delivered in this engineering fixture. Missing attendance,
integer crews, qualification classes and real booking granularity are not modeled.

If an eligible task receives h_i > 0 during interval t, it completes at the end
of that interval with probability p_i = 1 - exp(-theta_i(t) * h_i). Otherwise it
remains unfinished. A completed task does not free time for a second support
task in that same interval. This is an interval-observed, memoryless service
hazard in effort units, **not** fractional job completion or a calibrated staffing
response. There is no exact continuous work/remaining-work observation.

The exponential survival/CDF identity is standard; see the
[NIST handbook](https://www.itl.nist.gov/div898/handbook/apr/section1/apr161.htm).
Its use for support service, parameter values, discretization and interpretation
are our hypothetical assumptions, not evidence supplied by that source. The
sampled completion is discrete; only its expectation is smooth. This does not
guarantee that DDPG can learn a useful critic or improve total cost.

Use uniforms indexed by (engineering seed, interval, site), drawn in advance
even for intervals without eligible work. Actual outcomes may diverge across
policies; action-dependent RNG consumption must not shift the exogenous tape.
Response tapes and uniforms remain evaluator-only, outside policy inputs.

Every job then requires an indivisible two-interval downstream operation. Support
effort cannot shorten or bypass it. Start waiting jobs FIFO at interval start;
new support completions can start downstream no earlier than next interval.
Arrivals are booked, known at time zero, at [0,8,16,24] for each site. There is no
claim of hidden or stochastic demand. This deliberately permits idle intervals
before later work arrives; verify it changes transitions, not just holding cost.

Public observations contain time, named job stages, past ready times, known
downstream remaining duration, pending prepaid requests and previous request.
Public receipts contain committed/delivered effort, eligible task identity,
exposure, completion events and cost. They become available at t+1. No latent
theta, event probability, uniform, change flag or future completion is exposed.
Observed delivered time on an eligible task is exposure; purchased time with
no eligible task is idle spending, not a zero-productivity sample.

## Shared inference prototype

Implement one fixed Bayesian grid filter, with rates [0.5,1,1.5], uniform prior
and a 0.05-per-interval prior-refresh mixture. Predict before updating each
receipt. For an exposed task, completion likelihood is 1-exp(-rate*h); an
unfinished interval contributes exp(-rate*h), not a pseudo-label of zero rate.
No exposure contributes likelihood one. Use log-space normalization.

The mixture is a fixed engineering setting, not a tuned change detector or the
true law of each scenario. It is deliberately not called a sufficient belief
state or optimal estimator. Its data interface can later be shared by frozen
and online policies. Updating this belief is system identification, **not** an
online actor/critic update. One noisy event must not identify the rate exactly.

## Finite recorded check

Config: `experiments/configs/completion_feedback_mechanics_20260929.json`.
Cross unchanged / persistent-left-slow / persistent-right-slow / rapid-alternating
responses, downstream slots 1 / 4, and engineering uniform seeds 0 / 1: 16 cases.
All responses start (1,1). After interval 8 use (0.5,1.5), its reverse, or alternate
these each interval. The rapid case is a negative-control design, not an assumed
negative result. Only a public booked-backlog availability rule is executed:
buy 0.5 at a site when it has support work or a booking for next interval.
The filter is observed/logged but does not affect this rule's actions.

Run 32 decision intervals; then issue one zero request to drain the pipeline,
and use the same public rule until every booked job, downstream task and prepaid
request is settled. Charge holding on every released unfinished job at interval
end, labor 2*h+0.5*h^2 at purchase, and 0.25*|h-previous| for switching. These are
fixed hypothetical units, unchanged from the preceding mechanism checks, not
clinical loss or currency. No free terminal disposal or salvage is allowed.
Closure is capped at 256 intervals and fails visibly with the unresolved ledger;
failure is never a successful truncated episode. Maximum 4,608 steps per matrix.

Also evaluate the one-step completion function on a deterministic 1,000-point
midpoint uniform grid at rates {0.5,1,1.5}, effort {0,.25,.5,.75,1}. The midpoint
frequency must approximate the declared probability within 1/1000. These 15,000
function evaluations are quadrature, not independent Monte Carlo replications,
rollout returns or headroom. Record all values and derivatives of the declared
mean function; do not use them as online training labels.

## Verification and stops

Test invalid inputs, action delay, FIFO identity conservation, no future/support
state leakage, no-effort/idle/censored distinctions, exact interval likelihoods,
finite normalized posterior, atomic rejection of invalid/out-of-order receipts,
downstream occupancy and duration, recurrent arrival effects, purchased-hour
conservation, full cost closure and explicit closure failure. Audit persisted
receipt likelihoods and costs independently of the transition implementation.
Repeat the same matrix once; require identical scientific output and inventories.
Run relevant old regressions plus full Python compilation. Never overwrite a run.

This check establishes software/mechanism feasibility only. It does not compare
policies, estimate safe headroom, validate a clinical model, establish parameter
identification accuracy or prove online learning. There are no scientific
evaluation seeds, historical checkpoints, teachers or formal-holdout accesses.
Next, preregister a small matched-information conventional-control/headroom
screen and an independent ranking-replication check before training. A competent
frozen history policy and completion-data identification + MPC remain mandatory
for a later online-versus-frozen comparison; do not weaken either to favor RL.
