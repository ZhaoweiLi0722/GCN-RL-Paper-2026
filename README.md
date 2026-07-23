# Graph-Aware Deep Reinforcement Learning for Adaptive Capacity Planning

Research codebase for graph-aware capacity planning in distributed personalized
regenerative medicine (PRM) manufacturing networks. The problem is modeled as a
constrained graph MDP in which the product is perishable, identity-bound to a
single patient, and demanded only while that patient remains clinically eligible.
The manuscript lives under `paper/`.

## Where things are

Everything implemented is under `src/`, `evaluation/`, and `experiments/configs/`.
Written-up results are under `specs/` (raw output CSVs are git-ignored — see below).

### Simulation environment — `src/env/`

| File | What it is |
|------|-----------|
| `capacity_planning.py` | Base distributed capacity-planning MDP: facilities, reagent inventory, bioreactor capacity, transshipment/relocation, Poisson demand, Bernoulli supplier disruption, demand shocks, optional in-state demand forecast, and the per-episode train-time domain-randomization hook (`enable_train_randomization`). |
| `patient_capacity_planning.py` | The patient-condition environment used in the paper. Extends the base env with per-clinic patient queues, survival-index deterioration, eligibility loss, and the condition summary in the state/graph. |
| `patient_condition.py` | Patient survival / deterioration dynamics (health index, Weibull deterioration epoch, accelerated decline). |
| `aging_inventory.py` | Age-bucketed, expiry-aware material/product inventory. |

Environment configs are JSON under `experiments/configs/` (e.g.
`20_clinic_patient_condition.json` is the nominal 20-clinic setting;
`*_disruption_*`, `*_forecast.json`, `*_stress.json`, and the 2-clinic
curriculum config are variants). Build an env from a config with
`build_env(config)` in `src/rl/experiment.py`.

### Geographic network

The geography-aware scenarios use the 20 point-of-care locations in
`data/bb_20_clinic_locations.json`. Address-based coordinates define the clinic
graph, distance-related transfer cost and delay, continuous transfer time, and
regional demand/disruption clusters. See `docs/geographic_network.md` for the
location provenance and modeling assumptions. The joint patient-condition,
geography, and demand-prior-drift scenario is
`experiments/configs/20_clinic_patient_condition_geo_demand_drift.json`.

### Learning algorithms

| File | What it is |
|------|-----------|
| `src/models/gcn.py`, `graph_features.py` | Shared GCN state encoder, size-invariant `facility_action` readout, graph-feature reconstruction (patient- and forecast-aware). |
| `src/models/gcn_ddpg.py`, `gcn_td3.py`, `gcn_sac.py`, `gcn_ppo.py` | Graph-aware actor–critic backbones (GCN encoder then backbone). GCN-DDPG is the ablation; TD3/SAC/PPO are the family. |
| `src/baselines/flat_ddpg.py`, `td3.py`, `sac.py`, `ppo.py` | Flat-state (non-graph) RL, to isolate the value of the graph encoder. |
| `src/rl/agents.py` | Agent registry. `get_agent_class(name)` returns any of the above by key (`gcn_ddpg`, `flat_ddpg`, `td3`, ...). |
| `src/rl/experiment.py` | `build_env`, `build_agent_config`, `train_off_policy_agent`, shared training plumbing. |

### Heuristic baselines — `src/baselines/heuristics.py`

`myo` (myopic), `iso` (isolated-facility), `mdl1` / `mdl2` (look-ahead), plus the
fair information-aware baselines `umyo` (urgency/condition-aware) and `fmyo`
(forecast-aware). All use the true demand rates and the same forecast signal the
learned agents see.

### Experiment runners — `evaluation/`

| Runner | Experiment |
|--------|-----------|
| `campaign_runner.py` | Nominal 20-clinic campaign (graph vs flat vs heuristics). |
| `disruption_sweep.py` | Experiment A — supply-disruption severity. |
| `condition_stress.py` | Experiment B — patient-condition stress. |
| `forecast_robustness.py` | Experiment C — forecast-error robustness (train-on-range, test-OOD). |
| `budget_curve.py` | Budget / learning-curve diagnostic (undertraining vs real gap). |
| `evaluate_formal.py`, `aggregate_stats.py` | Monte Carlo evaluation and IQM / bootstrap-CI aggregation. |
| `run_patient_pilot.py` | Two-stage lean pilot. |

### Results and findings

Raw per-run CSVs are written to `results/` (e.g. `results/campaign/`,
`results/disruption_sweep/`, `results/condition_stress/`,
`results/forecast_robustness/`). **`results/` is git-ignored**, so those CSVs stay
local. The committed, written-up findings — the ones to read — are:

- `specs/2026-07-11-pilot-experiments/pilot-findings.md` — pilot outcomes.
- `specs/2026-07-11-pilot-experiments/campaign-results.md` — nominal campaign (graph beats flat by a wide margin; heuristics win at nominal).
- `specs/2026-07-12-robustness-experiments/results.md` — robustness experiments A/B/C, in-distribution vs OOD, honest-negative analysis, and the budget-escalation trigger.
- `specs/2026-07-17-patient-condition-geo-residual/results.md` — joint
  patient-condition/geography residual-policy development.
- `specs/2026-07-20-demand-drift-robustness/results.md` — advantage-filtered
  residual GCN-DDPG, five-seed paired evaluation, and the matched flat-state
  ablation.

### Plan of record — `specs/`

`specs/mission.md`, `tech-stack.md`, and `roadmap.md` hold the research argument,
architecture decisions, and sequenced phase plan. Each dated
`specs/YYYY-MM-DD-<phase>/` folder is a feature spec (`requirements.md`,
`plan.md`, `validation.md`) for one phase.

## Setup

```bash
conda activate gcn-rl          # project environment
export PYTHONPATH=$(pwd)        # runners import src / evaluation from repo root
```

## Running

Tests (no pytest; use unittest):

```bash
python -m unittest discover -s tests
```

A quick pilot, and the robustness runners (long; wrap in `caffeinate -i` on macOS,
all resumable per-CSV):

```bash
python -m evaluation.run_patient_pilot
CAMPAIGN_STEPS=150000 python -m evaluation.forecast_robustness
```

Override the training budget for any campaign / robustness runner with the
`CAMPAIGN_STEPS` environment variable.

The patient-forecast benchmark manifest is
`experiments/configs/patient_forecast_benchmark.json`, which evaluates the
graph-dynamic patient-forecast scenario (residual GCN-DDPG with MYO-anchored
local-search distillation) against flat DDPG, TD3, SAC, PPO, and the MYO/ISO/MDL
heuristics. Dry-run and smoke it, then run the 500-replication formal comparison:

```bash
python -m evaluation.run_full_benchmark --plan experiments/configs/patient_forecast_benchmark.json --phase dry-run --budget smoke
python -m evaluation.run_full_benchmark --plan experiments/configs/patient_forecast_benchmark.json --phase all --budget smoke
python -m evaluation.run_full_benchmark --plan experiments/configs/patient_forecast_benchmark.json --phase all --budget formal
```

## Reproducibility notes

- Keep experiment parameters in `experiments/configs/*.json`, not hard-coded.
- Large outputs (`results/`, `runs/`, `checkpoints/`) are git-ignored; commit the
  written-up summaries in `specs/` instead.
- Every reported number must trace to a logged run — no projected numbers as if
  measured.
