# Robustness / stress experiments — Results

IQM over seeds; fair baselines (`umyo`/`fmyo`) included per invariant I1. Learned agents
`gcn_ddpg` (facility_action) + `flat_ddpg` contrast, 150k steps. Cost is total episode
cost (lower better); elig = mean eligibility rate; lost = patients lost; service = level.

## Experiment A — supply-disruption severity (per-regime, A1) ✅ 2026-07-12

Question: does graph-DRL's advantage grow with disruption? 3 seeds; 0.3 midpoint is the
main campaign. `include_demand_forecast_state=false` here, so `fmyo` has no signal and
degenerates to `mdl1` (expected — `fmyo` is meaningful only in Experiment C).

| disruption | strongest method | cost | gcn_ddpg cost | gap (gcn/best) | elig gap |
|---|---|---|---|---|---|
| 0.05 | myo (blind) | 9.72e8 | 1.396e9 | **1.44×** | 0.094 |
| 0.6  | **umyo (fair, cond-aware)** | 1.017e9 | 1.38e9 | **1.36×** | 0.057 |

Full table (IQM/3 seeds):

- **0.05:** myo 9.72e8 · iso 9.78e8 · mdl2 9.82e8 · fmyo=mdl1 9.87e8 · umyo 1.065e9 ·
  **gcn_ddpg 1.396e9** · flat_ddpg 3.087e9
- **0.6:** **umyo 1.017e9** · mdl2 1.031e9 · iso 1.076e9 · fmyo=mdl1 1.076e9 · myo 1.136e9 ·
  **gcn_ddpg 1.38e9** · flat_ddpg 2.846e9

**Findings (honest):**
1. **Graph ≫ flat, robustly** — `gcn_ddpg` ~2× better than `flat_ddpg` at both endpoints
   (RQ1 holds under disruption stress).
2. **Gap narrows with disruption** (1.44× → 1.36× on cost; elig gap 0.094 → 0.057) —
   directionally supportive, but…
3. **…disruption alone does not flip the ranking.** Even the fair, condition-aware `umyo`
   (strongest at 0.6) beats graph-DRL by 1.36×. Heuristics are strikingly robust
   (mdl2 cost barely moves 0.05→0.6). **Supply disruption is not the regime where DRL wins.**

Implication: the "DRL wins under stress" story rests on **forecast drift (Exp C)** — where
DRL consumes a shared signal heuristics use naively — and possibly learned condition
anticipation (Exp B). The strong non-crippled baseline here makes a future C win credible.

**Caveat:** 3 seeds, 150k steps (kill-prone env budget); consistent with the nominal
campaign's undertraining note but `gcn_ddpg` is not divergent (elig ~0.71–0.75, vs
`flat_ddpg` collapse ~0.26).

## Experiment B — patient-condition stress ✅ 2026-07-12

Hardened deterioration (frail_decay 0.06→0.12, weibull_scale 8→4, post_shock 2→3,
eligibility 0.75→0.80). 3 seeds, 150k. Headline claim was `gcn_ddpg` beats `umyo`.

Full table (IQM/3 seeds), cost ascending:
- mdl2 6.643e8 (elig .771) · fmyo=mdl1 6.693e8 (.747) · iso 6.71e8 (.772) · myo 6.787e8 (.751)
  · **umyo 8.046e8 (.749)** · **gcn_ddpg 9.644e8 (.715)** · flat_ddpg 2.261e9 (.234)

**Findings (honest negative):**
1. **Claim fails.** `gcn_ddpg` (9.64e8) loses to `umyo` (8.05e8) by 1.20× on cost and on
   eligibility (.715 vs .749). Learning the condition response did **not** beat the
   hand-crafted urgency rule at this budget.
2. Blind look-ahead (`mdl2`/`iso`) is actually the strongest here — even `umyo` (condition-
   aware) trails it on cost. Condition stress does not reward condition-*awareness* per se.
3. **Graph ≫ flat holds** (9.64e8 vs 2.26e9). RQ1 survives; the DRL-beats-heuristics claim
   does not.

## ⚠️ Strategic read after A + B

Two of three robustness experiments are honest negatives: neither supply disruption nor
patient-condition stress flips the heuristics-beat-DRL ranking. `gcn_ddpg` sits at a
strikingly consistent ~1.4× cost gap and ~0.71–0.75 eligibility across nominal, A, and B —
the same signature the pilot flagged as possible **undertraining** at 20 clinics
(150k steps, ~80-d action). Either (a) a real capability gap at this budget, or (b)
systematic undertraining. The entire "when does DRL win" narrative now rests on **Exp C**
(forecast drift, true OOD) — the one regime where DRL consumes a signal (in-state forecast)
that heuristics use only naively. If C is also negative, the honest paper story becomes
"graph ≫ flat, but tuned heuristics beat learned control across all tested regimes at this
budget" — which would make a **budget escalation** (300k–500k, the flagship tier) the next
lever before any DRL-wins claim.

## Experiment C — forecast error, redeemed (flagship) — ⏳ queued after B

Train error ~ U[0,0.4]; test {0,0.2,0.4} in-dist, {0.6,0.8} OOD; fair `fmyo` baseline
(forecast ON here, so `fmyo` is live). Pending.
