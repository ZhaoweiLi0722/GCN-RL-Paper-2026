# S1 readout: completion-feedback mechanics pass; learning benefit untested

Date: 2026-09-29. Local synthetic engineering only. No patient-model changes,
training, formal-holdout reuse, remote actions or changes to manuscript results.

## What is implemented

`src/env/completion_feedback_queue.py` provides a separate two-site support
queue. Delivered support availability influences an interval completion
probability. Tasks remain indivisible; the mandatory downstream operation is
unchanged by support effort. Requests arrive one interval later, and all paid
availability and switching/holding charges are accounted for.

The controller sees completed/unfinished task events, exposures, bookings and
queues, not exact remaining support work, true productivity or the random tape.
Repeated future bookings can create actual idle periods and later renewed work;
a paired unit test verifies changed completion events, not only cost offsets.

`src/baselines/completion_response_filter.py` updates a fixed grid belief using
only these receipts. An unfinished exposed interval contributes a survival
likelihood, not a zero-productivity label. An idle interval supplies no response
measurement. Prediction-time prior refresh is separate from measurement update.
The same interface is available to a future frozen and online controller.
This is **system identification, not an actor/critic update**.

The model's exponential completion probability is a declared hypothetical use
of the [standard exponential CDF](https://www.itl.nist.gov/div898/handbook/apr/section1/apr161.htm).
That mathematical reference supplies neither manufacturing calibration nor
evidence of clinical staffing effects. Constant memoryless service, divisible
effort, immediate acceptance, perfect attendance and known bookings all remain
assumptions. No site measurements were supplied or imputed into E1.

## Verified results

| Check | Recorded result |
|---|---|
| Mechanics cases | 16/16 pass |
| Intervals in primary matrix | 512 |
| Support completions | 128, all followed by mandatory downstream completion |
| Exposed but unfinished site-intervals | 168 |
| Outstanding jobs/prepaid availability at end | 0 in every case |
| Independent persisted-row and likelihood audits | 16/16 pass |
| Midpoint probability checks | 15 settings, 1,000 points each, within 0.001 |
| Repeat | All 7 output files byte-identical |
| Tests | 29 new; 97 total related tests pass |

Both recorded matrices pass on their first attempt. The repeat adds another
512 intervals; unit tests execute additional small fixtures separately. The
15,000 midpoint evaluations per matrix are deterministic quadrature, not 15,000
independent runs. The two engineering seeds and shared tapes do not support
statistical effect estimates. No policy is optimized or compared here.

All main-matrix jobs finish before the 32-interval decision boundary, so those
cases alone do **not** exercise a positive-length terminal continuation. Two
additional unit tests explicitly force early cutoffs, preserve future bookings,
charge the tail and reject reporting prefix-only costs. The one-decision fixture
needs 30 closure intervals: decision cost 4.5, closure cost 64.5, all obligations
settled. These are accounting assertions, not additional scientific scenarios.
The bounded failure test leaves an unresolved ledger and fails the audit rather
than calling unpaid truncation a completed episode.

Under the single public booked-backlog rule, each case has 3.5-4 paid availability
units without an eligible task. They remain charged. This is **not** an avoidable
cost bound: purchasing ahead can rationally trade some idle spending for reduced
delay. A frozen controller can make that trade-off too.

Changing downstream slots changes downstream event timing in all eight paired
family/seed cases, over 3-13 intervals each. It does not change support completion
events under this rule. The intermediate queue is unbounded: downstream waiting
affects completion cost but does not physically block upstream support. Do not
claim finite-buffer backpressure or network-wide learned coordination.

## Decision and remaining research gates

The new mechanism has noisy, completion-only information and nonzero expected
action leverage. One event does not reveal the rate exactly. It is therefore
a better-defined vehicle for the proposed question than the old exact-progress
fixture, but **neither useful headroom nor an online-RL contribution is established**.
The fixed-rule cases have only eight jobs and 13-26 exposed site-intervals each;
they do not measure sustained adaptation or identification accuracy.

Do not launch DDPG yet. The next bounded scientific preparation is:

1. Lock a conventional-control/headroom screen with identical causal inputs,
   actions, costs, complete liabilities and private random tapes. Compare a
   sensible state-feedback rule, fixed-model MPC and completion-data
   identification + MPC. Document planning convergence/query budgets rather
   than using a deliberately weak short-horizon planner.
2. Rank candidate decisions by settled downstream total cost, not only immediate
   completion. A newly purchased action cannot affect service until next interval;
   immediate reward labels alone would miss its benefit. Replicate selected
   rankings on an independent development split without changing weights.
3. Before a neural comparison, establish a competent pretrained history-aware
   frozen policy with the same estimator interface, and verify feature/reward
   parity for this new environment. The old patient checkpoint and N7 engineering
   feature dimensions are not automatically compatible. Then compare tensor-
   equal frozen/online forks including adaptation/exploration costs.

The rapid-alternating engineering tape is deterministic and potentially
predictable from history/time. It is **not** a valid stand-in for independent,
unpredictable fast noise or proof that a negative-control world is unlearnable.
A scientific persistence comparison needs a separately specified stochastic
change law, dwell-time/marginal matching and replication design. Do not silently
relabel or pool these engineering cases with that future study.

Future budgets, scientific seeds, practical effect margins and a statistical
analysis are not assigned by this mechanics check. Full clinical calibration,
many-site graph claims, finite buffers and uncertain attendance also remain
unvalidated. Stop if ordinary adaptive controls solve the justified problem;
do not alter it post hoc merely to make DDPG win. Stage E remains closed.

## Provenance and reproduction

- Protocol/config commit: `8e46e50`.
- Recorded execution commit: `3f9db7ffbc6d2e209a73a93bcfb5a58bfbcd3dd8`.
- Reports: `reports/2026-09-29-completion-feedback-mechanics/{run,repeat}/`.
- Summary SHA256: `8aacaf8604103ac0e43c76fa6f5bd264b1ca033a5e0606e633f03a07a5c966b1`.
- Trace SHA256: `75efda57b0cd8334648d1e87510dd731000d29b385f270fd3ad38495f06e746e`.
- Python 3.9.6, NumPy 2.0.2; no Torch/GPU or optimizer needed by the new modules.

Nine source/config/protocol locks and eight historical locks verify unchanged.
Both inventories and persisted row audits reverify; full Python compilation
and diff checks pass. The added positive-closure tests were written after the
recorded matrix revealed zero tail intervals; they do not change the locked
implementation or the recorded outcomes.

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m evaluation.check_completion_feedback_mechanics --output /private/tmp/completion-feedback-new-check
"$PY" -m unittest tests.test_completion_feedback_queue tests.test_completion_feedback_check tests.test_completion_feedback_closure
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Use a new destination. Source files must equal their committed versions;
recorded source hashes allow replay of the original version. A different HEAD
changes the claim/inventory metadata even if scientific JSON/trace bytes match.
