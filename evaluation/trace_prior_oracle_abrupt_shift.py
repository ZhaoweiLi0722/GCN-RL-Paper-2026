"""Per-epoch, per-tier trace of prior vs oracle MDL-2 on the abrupt regime shift."""
import sys, copy, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from evaluation.prior_oracle_gap_screen import compose_scenarios, make_policy
from evaluation.run_full_benchmark import load_benchmark_plan
from src.rl.experiment import build_env

plan = load_benchmark_plan('experiments/configs/patient_indexed_specimen_routing_benchmark.json')
sc = compose_scenarios(plan, ['routing_abrupt_regime_shift'])['routing_abrupt_regime_shift']['env']
TIERS = {'T0-4 (1.25→0.75)': slice(0, 5), 'T5-9 (0.80→1.40)': slice(5, 10), 'T10-14 (1.15→0.80)': slice(10, 15), 'T15-19 (0.85→1.30)': slice(15, 20)}
CHANGE = int(sc['demand_regime_change_step'])
N_WORLDS = 30
SEED = 99_910_000

def run(arm):
    env = build_env({'env': copy.deepcopy(sc)}, seed=SEED)
    pol = make_policy(arm)
    T = int(sc['episode_horizon'])
    acc = {k: np.zeros((T, 20)) for k in ['lost', 'under_bio', 'under_reag', 'reagents', 'idle_bio', 'purchase', 'waiting']}
    for w in range(N_WORLDS):
        state = env.reset(seed=SEED + w); pol.reset()
        for t in range(T):
            acc['reagents'][t] += env.reagents
            acc['idle_bio'][t] += env.bioreactors[:, 0]
            acc['waiting'][t] += env.specimens
            a = pol.select_action(state, explore=False, env=env)
            state, r, done, info = env.step(a)
            acc['lost'][t] += np.asarray(info['patients_lost'], dtype=float)
            acc['under_bio'][t] += np.asarray(info['under_bioreactors'], dtype=float)
            acc['under_reag'][t] += np.asarray(info['under_reagents'], dtype=float)
            if done: break
    for k in acc: acc[k] /= N_WORLDS
    return acc

res = {arm: run(arm) for arm in ['mdl2_prior', 'mdl2_oracle_rate']}
print(f'{N_WORLDS} worlds; regime change at epoch {CHANGE}. Values are per-world sums over the tier and window.')
for metric in ['lost', 'under_bio', 'waiting', 'idle_bio', 'reagents']:
    print(f'\n== {metric}')
    print(f"{'tier':22s} {'window':10s} {'prior':>10s} {'oracle':>10s} {'oracle-prior':>13s}")
    for tier, sl in TIERS.items():
        for wname, wsl in [('pre  <26', slice(0, CHANGE)), ('post >=26', slice(CHANGE, None))]:
            p = res['mdl2_prior'][metric][wsl, sl].sum(); o = res['mdl2_oracle_rate'][metric][wsl, sl].sum()
            if metric in ('idle_bio', 'reagents', 'waiting'):  # stock levels: report mean per epoch per tier
                n = res['mdl2_prior'][metric][wsl, sl].shape[0]; p /= n; o /= n
            print(f'{tier:22s} {wname:10s} {p:10.1f} {o:10.1f} {o-p:+13.1f}')
# epoch profile of network patient loss around the change
print('\n== network patients lost per epoch, epochs 20..40 (prior | oracle)')
for t in range(20, 41):
    print(t, f"{res['mdl2_prior']['lost'][t].sum():6.1f} | {res['mdl2_oracle_rate']['lost'][t].sum():6.1f}")
