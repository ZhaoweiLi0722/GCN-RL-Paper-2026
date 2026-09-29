# N4 readout: real collection and neural forward path

## Decision

The bounded engineering check passes. The real patient environment now has a
separate explicitly enabled path through lossless observation construction,
frozen neural actor/gate selection, actual environment steps, lineage-checked
replay returns and target actor/critic forwards. It does not replace the old
agents and has no optimizer, update method, training registration or checkpoint
migration. This is software verification, not new evidence of online RL benefit.

## Recorded evidence

- Protocol/config commit: `f2efa1f`.
- Initial implementation: `96e0d9a`; source-inventory correction: `dad186a`.
- Successful execution commit: `dad186ab4e21e1a238e19590ecd1711aebafc9f1`.
- Raw report: `reports/2026-09-29-prospective-collector-engineering/check.json`.
- Report SHA256: `67c35b480a7ca647f791e6a401660eb4ced8d413ef7c3421ac68b7fab1ec3c61`.
- Python 3.9.6, NumPy 2.0.2, PyTorch 2.8.0; explicitly CPU/float32.
- Twenty-eight loaded local source files are hashed; all reconcile to disk.
- Two successful executions produce byte-identical reports at this source commit.

The first wrapper invocation at `96e0d9a` completed its checks but failed while
hashing PyTorch's relative virtual `_ops.py` module name as a file. It wrote no
report. The correction excludes relative pseudo filenames, retains absolute
local source checks and adds a regression test. No scientific result was
discarded, checkpoint resumed, objective tuned or historical artifact replaced.

## Checks and limits

| Check | Observed evidence |
| --- | --- |
| Actual execution | 9 cases, 36 decisions; graph/flat, physical/self-only messages and explicit cutoff/horizon mappings |
| Snapshot continuity | All 36 real steps reproduced exactly in a cloned environment: observation, reward, done, complete info and post-step snapshot/RNG digest |
| Physical routing | The fixture executes two specimen routes per case; identity conservation is checked |
| Observation information | Every raw base/patient-summary/time slot reconstructs exactly; both views receive physical links, state-specific MDL-2 anchors and declared gate proposals |
| Hidden state separation | Snapshot/RNG tokens are metadata, not neural inputs; changing RNG alone changes the token but not observation-derived inputs |
| Action coordinates | Pre-gate proposal, gate, submitted normalized request and actual integer execution are distinct; replay uses the submitted request |
| Returns | Windows of lengths 3, 3, 2, 1, including every tail; one reward scale and gamma^n; independent scalar-return and NumPy-target references agree |
| Horizon | Collector cuts and declared environment truncation bootstrap; declared finite-terminal fixtures do not bootstrap beyond the terminal record |
| Target path | Endpoint state/anchor, separate copied actor/gate/critic targets and post-gate next request reach the actual target-Q forward |
| No learning | Zero optimizer updates; all network and target weights unchanged; no gradients accumulated |
| Validation | 17 new tests, combined 171-test suite, full compileall and diff checks pass |

Current engineering head counts (excluding separate target copies): graph
actor/critic/gate = 1,192/1,189/1,288, total 3,669; flat = 772/769/868, total
2,409. Both have zero trainable parameters in this frozen check. They are
**not parameter matched**. Equal raw inputs do not establish isolated graph
attribution or reproduce historical trained controllers. These deliberately
small random-weight heads are for tensor/dataflow verification only; no policy
quality comparison is made.

The gate is an explicit soft sigmoid with zero-logit initialization, not a
claim to reproduce the historical learned gate. Only specimen coordinates
receive the declared residual scale. Rewards are absolute negative total cost,
not an implicit conversion of historical anchor-relative/counterfactual labels.
The time-horizon tests validate software flags, not clinical liability closure.
Unsupported overtime/procurement/hub or multiple relation layouts fail closed.
Resume, production learner integration and GPU/device equivalence are untested.

## Reproduction

From the persistent integration worktree, with the implementation committed,
use a fresh output path (the script refuses existing files):

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_patient_replay_collector
"$PY" experiments/scripts/check_prospective_collector.py \
  --config experiments/configs/prospective_collector_engineering_20260929.json \
  --output /private/tmp/prospective-collector-new-check.json
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
```

The report includes the current execution commit. Whole-file byte equality
requires the same commit/runtime; later documentation commits legitimately
change that metadata. No evidence files should be overwritten to reproduce it.

## Next scientific boundary

This finishes the approved N4 packet. Before any learning study: choose and
freeze the actual training reward, deployment gate, architecture/parameter
controls, optimizer/replay/resume contract, action leverage and terminal
accounting; approve a development-only protocol with frozen-policy, online
policy, adaptive-rule and identification-MPC comparisons. Domain inputs for
the proposed qualified-support scenario remain unresolved. No training launch,
scenario calibration, positive gain, publication readiness, Howard sign-off or
remote operation follows automatically from this software result.
