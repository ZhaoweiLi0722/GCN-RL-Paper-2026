# Family Selection Integration Readout

2026-10-06. Complete-package approval received; source/config freeze and launch
are the next routine actions. No scientific execution is claimed by this note.

Five candidates are integrated on the same31-feature support-hour task with
fixed H8/H16 and the predeclared historical-reference model set. Four actor-
critic implementations reuse the existing graph modules; the value arm reuses
observed TD8 and the repaired planner. All historical scientific files and
consumed attempts remain unchanged.

Necessary checks:23tests passed (16learner synthetic tensor checks and7runner/
selection/resource checks), plus full `python -m compileall -q .`. Runner tests
forbid real native environments and optimizers; learner tests use only artificial
tensors. These checks are engineering evidence, not scientific performance.

| Family | Forwards per training world | Optimizer dispatches | Trainable parameters |
| --- | ---: | ---: | ---: |
| DDPG | 208 | 64 | 6370 |
| TD3 | 240 | 80 | 9571 |
| SAC | 304 | 96 | 9575 |
| PPO | 160 | 64 | 6342 |
| observed TD8 value-MPC | 81 | 32 | 3169 |

Each training world has48actions and32minibatch iterations. Target forwards,
twin critics and PPO rollout values are counted; targets are excluded only from
the trainable parameter column. Evaluation uses48forwards per learned role,
zero updates. Exact selected-family totals are checked against real receipts at
completion; the approval envelope is not mislabeled as consumed compute.

Two finite delegates completed and closed: advancement implemented the four
learners and tests; read-only efficiency review reused saved timing receipts,
recommended48h/32GiB and warned against duplicated replay snapshots. No extra
toy campaign, history re-audit or additional scientific gate is required.

The single package includes development selection, fresh finalist retraining,
graph/self-only and initial controls, all raw-data readout, single archive,
local manuscript conclusions and existing additive Dropbox handoff. No result
or unique winner is promised. No automatic scientific follow-on or retry.
