# N7 readout: complete bounded collection/learning session recovery

## Outcome

The live collector, OU explorer and DDPG kernel now pass the declared single-
episode CPU closed-loop mechanics check, including exact recovery in a new Python
process. Historical source, N4/N5/N6 modules, old defaults/results and the formal
holdout were not modified. There was no performance comparison or online-gain claim.
The fixture retains N4's gamma .9, absolute cost reward scale 1e-5 and three-step
returns for mechanics coverage, not N5's proposed settled-task scientific objective.

| Software coverage case | Primary path length | Online updates | Frozen updates |
| --- | ---: | ---: | ---: |
| Graph64 / physical / collector cutoff | 6 | 3 | 0 |
| Graph64 / self-only / environment truncation | 6 | 3 | 0 |
| Flat68 / physical / terminal-mask mapping | 6 | 3 | 0 |

Each of these six mode-specific paths is run continuously and independently as
a four-step prefix plus a two-step fresh-process continuation. At the interruption,
the state contains two pending records, two replay windows and one online update
(zero in frozen mode). Closing the path publishes lengths `[3,3,3,3,2,1]` exactly
once, leaves no pending records and wraps the capacity-4 ring to position 2.

All six restarted paths match their own continuous counterpart exactly, including:

- Requested actions, pre-gate proposals, gates, deterministic requests and OU draws.
- Raw state/next-state records, rewards, execution receipts and patient identities.
- Environment arrays/dtypes, queues/transits, histories, counters and RNG state.
- Collector cursor/closure/lineage, unflushed records and replay provenance/order.
- Replay samples, update losses, all model/target tensors, Adam and sampling RNG.

Online weights change; frozen weights remain identical. The frozen branch follows
the same collection and replay scheduling code without optimizer updates. No claim
is made that its untrained engineering initialization is a competent baseline.
Each case routes two specimens, verifying that the physical channel was exercised;
this neither establishes behaviorally distinct routing from online updates nor
constitutes a routing-performance improvement.

## Execution and verification

One matrix executes 72 primary environment steps, 72 clone-verification steps,
18 kernel updates and 36 Adam steps. The repeated matrix adds the same amounts:
144 primary plus 144 clone steps, 36 kernel updates and 72 Adam steps in total.
Unit tests add bounded mechanics steps separately. These are not scientific seeds
or independent performance replications.

- Protocol/config commit: `6ea2116a94c2ec428bcf3ccc11411c8a97538d9e`.
- Recorded execution commit: `8203350384f5e1bb7ba2ec515eac7ffd6d6c6342`.
- Report: `reports/2026-09-29-prospective-closed-loop-engineering/check.json`.
- SHA256: `b7031b9e4a1ff18d090bbde58db503ed6a25f6da29e81bc653035a29d7aa0965`.
- Thirteen diagnostic JSON files are byte-identical on repetition. Both recorded
  matrices passed on their first execution; all workers exit 0 with empty stderr.
- Eighteen checkpoint payloads match tensor for tensor across runs. The six final
  continuous/restarted payload pairs also match by direct recursive comparison,
  not only by comparing stored checksum strings. Both file inventories verify.
- All 34 loaded local source hashes and the locked N4-config/N6-report hashes match.
- Fourteen new tests and the combined 144-test regression suite pass; full Python
  compilation and diff checks pass.
- Python 3.9.6, torch 2.8.0, NumPy 2.0.2, CPU float32, one torch thread. This is an
  explicit CPU engineering fixture, not a fallback from a required MPS experiment.

The tiny wrapper is `src/rl/prospective_patient_session.py`; the finite runner is
`evaluation/check_prospective_session.py`. It reuses the original collector,
environment snapshot API, OU process and N6 learner rather than changing those
components or registering another algorithm. Each session refuses a second episode
and cannot exceed six decisions. One successful update is scheduled per eligible
decision; no hidden tail-only extra optimization occurs at closure.

The checkpoint encodes NumPy arrays as dtype-tagged tensors and loads with
`weights_only=True`. It validates manifests, canonical environment round trips,
observed endpoints, lineage, pending/replay reconstruction, OU draw history and
update counts before replacing live state. Malformed loads leave state untouched.
Injected failures after the environment has already advanced or during a kernel
update roll back the entire session. Save paths cannot overwrite existing evidence.

The clone is a scratch verification environment: each step reloads the actual
pre-step snapshot into it before comparing the step. Its previous state is not an
input to policy selection and need not be in the session checkpoint.

The 18 `.pt` files remain local under the persistent report directory, about
6.2 MB including JSON. Existing `.gitignore` rules remain unchanged; binaries
are not force-added. Their sizes, binary hashes and semantic hashes are in the
tracked inventory. Binary container bytes differ across saves, so that inventory
is verified separately rather than required to be byte-identical between runs.
The read-only follow-up verification is recorded in `repeat_verification.json`.
Nothing was pushed, merged, sent to Howard or scheduled as an automation.

## Reproduction

Use the persistent worktree and a new output destination:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_prospective_session tests.test_patient_replay_collector tests.test_prospective_learner
"$PY" -m evaluation.check_prospective_session \
  --config experiments/configs/prospective_closed_loop_engineering_20260929.json \
  --output /private/tmp/prospective-session-new-check
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

Whole-report byte equality needs the same source commit/runtime because execution
metadata is embedded. Checkpoint payload comparisons do not require the temporary
archive filenames to match. Existing outputs must never be overwritten.

## Next research gate

The N1-N7 software chain now covers typed returns, common inputs, bounded actual
collection, matched counts, actual updates and single-episode CPU recovery. This
is sufficient to close this engineering packet; additional infrastructure alone
will not create a useful online-learning signal.

The central missing evidence is still a defensible decision with safe, replicable
headroom beyond a competent frozen history policy and adaptive controls. For the
proposed qualified setup-support channel, E1 still requires a specific bookable
task/crew/resource model, time-valid event availability, and units/costs/closure.
These cannot be filled by copying the convenient routing fixture. A documented
synthetic study is possible only with an explicit hypothetical scope and a new
approved protocol; it cannot be represented as calibrated manufacturing evidence.

Next scientific work should resolve those assumptions and measure action leverage
and state-dependent value before selecting online DDPG hyperparameters or
launching a performance comparison. Use equal information and include adaptive
rules and identification-plus-MPC. Preserve unfavorable results. Neither positive
online benefit nor a publication outcome can be promised in advance.

Current scope excludes GPU/asynchronous/multi-episode recovery and clinical
terminal settlement. The terminal mapping above only tests TD masks; it does not
settle outstanding patients. The small fixture does not validate changing-capacity
scenarios, stochastic world generalization, or a policy's ability to learn them.
N5's scientific launch dependencies remain unresolved, and Stage E remains closed.
