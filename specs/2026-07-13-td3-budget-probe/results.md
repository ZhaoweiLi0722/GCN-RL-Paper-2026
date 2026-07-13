# Supplemental GCN-TD3 budget probe results

Date: 2026-07-13

This is a supplemental local diagnostic run from Zhaowei's correct
`/Users/lizhaowei/GCN-RL Paper 2026` workspace. It should not replace Howard's
Stage 0b result in `specs/2026-07-12-robustness-experiments/results.md`, which
is the stronger 500k / 3-seed result and should remain the primary basis for the
manuscript decision. The purpose of this note is to record an independent 300k
cross-check using the same `gcn_td3` + `facility_action` runner.

## Run Summary

The smoke test passed in the correct workspace:

```bash
PYTHONPATH=. .venv/bin/python -m evaluation.td3_budget_probe --smoke
```

The default long run was then started:

```bash
PYTHONPATH=. caffeinate -i .venv/bin/python -m evaluation.td3_budget_probe
```

The run completed `gcn_td3` at 300k steps for seeds 0 and 1. Seed 2 was still
training after a long wait, so the run was interrupted before the 500k stage.
The two completed TD3 seed CSVs were retained under:

```text
results/td3_budget_probe/steps_300000/
```

To obtain a same-seed comparison against the reference heuristics, the runner
was then rerun for the completed seeds only:

```bash
PYTHONPATH=. .venv/bin/python -m evaluation.td3_budget_probe --steps 300000 --seeds 0 1
```

This reused the existing `gcn_td3_seed0.csv` and `gcn_td3_seed1.csv`, evaluated
the reference heuristics for seeds 0 and 1, and wrote:

```text
results/td3_budget_probe/steps_300000/summary.csv
```

The 300k result below is therefore a 2-seed diagnostic, not the planned full
3-seed result.

## 300k Step Result

IQM summary over seeds 0 and 1, with 20 evaluation replications per seed:

| Algorithm | Cost IQM | Eligibility IQM | Patients Lost IQM | Service IQM |
| --- | ---: | ---: | ---: | ---: |
| MDL-2 | 0.978B | 0.8240 | 1158.8 | 0.6799 |
| ISO | 0.985B | 0.8245 | 1156.0 | 0.6804 |
| MDL-1 | 0.996B | 0.7966 | 1316.2 | 0.6471 |
| MYO | 1.008B | 0.8051 | 1280.4 | 0.6538 |
| UMYO | 1.022B | 0.8053 | 1281.7 | 0.6542 |
| GCN-TD3 | 1.538B | 0.6903 | 2153.1 | 0.4720 |

Best heuristic: `mdl2`, cost IQM = 0.978B.

GCN-TD3 gap versus best heuristic:

```text
1.538B / 0.978B - 1 = 57.4%
```

## Seed-Level Diagnostic

GCN-TD3 showed strong seed sensitivity at 300k:

| Seed | Mean Cost | Eligibility Mean | Patients Lost | Service |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 1.091B | 0.8129 | 1231.9 | 0.6647 |
| 1 | 1.986B | 0.5677 | 3074.3 | 0.2793 |

Seed 0 was closer to the heuristic family but still above MDL-2. Seed 1
collapsed on both service and eligibility. This is the main signal from the
budget probe: the facility-action GCN-TD3 readout improves plausibility relative
to short smoke runs, but 300k steps does not produce stable heuristic-level
performance.

## Interpretation

Under the decision gate in `plan.md`, the observed 300k gap is above the
`gap >= 0.35` threshold. Eligibility and service also remain clearly below the
strongest heuristics in the 2-seed diagnostic.

This agrees qualitatively with Howard's Stage 0b 500k result: TD3 is more
promising than the unstable DDPG ablation, but tuned heuristics still win on
cost and the learned policy remains seed-sensitive. Howard's 500k seed 0 reaches
heuristic-level eligibility but at higher cost; this local 300k run shows the
same pattern one step earlier, with one relatively good seed and one collapsed
seed.

Recommendation:

- Do not merge the PR as a "graph RL beats tuned heuristics" result.
- Follow Howard's Stage 0b interpretation as the main result: use the honest
  negative framing, with TD3-vs-DDPG as a useful backbone-stability finding.
- The manuscript story should emphasize:
  - the patient-condition simulation testbed,
  - graph RL improving learned control over flat RL,
  - fair tuned heuristics remaining stronger under the tested regimes,
  - the need for stronger structure, imitation, or hybrid heuristic-RL policies
    before claiming learned superiority.

## Next Decision

A full 500k, 3-seed run would be expensive on the current local machine. Given
the 57.4% 300k gap and seed-1 collapse, the next experiment should be a targeted
single-seed or two-seed 500k spot-check only if we specifically want to test
whether more training rescues the unstable seed. Otherwise, the paper should
pivot now to the honest negative narrative.
