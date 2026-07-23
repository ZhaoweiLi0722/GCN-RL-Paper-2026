# Budget-escalation plan — is the DRL negative undertraining or a real gap?

Pre-drafted so it can fire the moment Experiment C lands. **Do not blindly re-run the
whole campaign at 500k** — that is ~15 h of compute on a bet. Run a *cheap decisive probe*
first; escalate the full campaign only if the probe shows the gap actually closing.

## Trigger

Evaluate after C completes:
- **C is also a negative** (DRL does not beat `fmyo`, incl. OOD): execute this plan.
- **C is positive** (DRL degrades more gracefully OOD, CIs separate): the "when does DRL
  win" narrative is already saved. Escalation becomes *optional* — run only Stage 0 to
  strengthen the flagship numbers, skip the full re-campaign.

## The question

`gcn_ddpg` shows a strikingly consistent **~1.4× cost gap / ~0.72 eligibility (vs ~0.80)**
across nominal, A (both disruption endpoints), and B (condition stress), all at 150k steps.
Two mutually exclusive explanations, with opposite implications:

- **(U) Undertraining** — the 20-clinic action space (~80-dim) and kill-prone dynamics need
  more than 150k steps to converge. Fixable by budget. The pilot already showed 30k→150k
  was decisive once; 150k→500k may be the next step.
- **(G) Real capability gap** — tuned look-ahead heuristics genuinely dominate learned
  control here. Not fixable by budget; needs reward/architecture work or an honest negative.

A single learning curve discriminates (U) from (G). Everything below is built to answer it
with the least compute.

## Stage 0 — Learning-curve probe (decisive, cheap; do this first)

**Runner:** `evaluation/budget_curve.py` (pre-built, smoke-validated). `gcn_ddpg`
(facility_action) on the **nominal** 20-clinic env (`CAMPAIGN_ENV`), where the gap is
cleanest and heuristics are already banked from the campaign. Resumable per (algo, seed,
steps) CSV. `flat_ddpg` included at the top budget only, to check whether it is the encoder
or the RL that is starved.

**Ladder (cheapest-decisive-point first):**
1. **0a — 500k, `gcn_ddpg`, 3 seeds** (~4.4 h). The single most informative point: if 500k
   doesn't move the gap, budget is not the answer.
2. **0b — 300k, `gcn_ddpg`, 3 seeds** (~2.6 h) — fill the midpoint only if 0a shows partial
   closure, to establish the *trend* (still descending vs plateaued).
3. **0c — 500k, `flat_ddpg`, 3 seeds** (~2.2 h) — only if 0a closes the gap, to confirm the
   win is the graph encoder scaling, not flat RL catching up.

150k is already banked (campaign + A/B), giving the low anchor for free.

**Decision gate** (relative cost gap `G(s) = cost_gcn(s)/cost_bestHeur − 1`; at 150k,
`G ≈ 0.40–0.45`):

| G(500k) | Read | Action |
|---|---|---|
| `< 0.20` (gap more than halved) **and** still descending 300k→500k | **Undertraining** | → Stage 1: escalate the flagship campaign |
| `0.20–0.35` | **Ambiguous** | fill 0b/0c; extrapolate; one 1M spot-check on the best seed before committing |
| `≥ 0.35` (barely moved) | **Real gap** | → Stage 2: do not escalate; pivot narrative |

Apply the same test to the eligibility gap (0.72→? vs 0.80) as a cross-check; cost is the
headline.

## Stage 1 — Escalated flagship campaign (only if Stage 0 ⇒ undertraining)

Re-run the experiments most likely to convert, at the winning budget (500k, or the knee of
the curve), 5 seeds, IQM/CI, same referee invariants (fair `umyo`/`fmyo`, true OOD):
1. **C first** (forecast, flagship) — the OOD generalization claim benefits most from
   convergence, since a stronger policy exploits the in-state forecast heuristics use naively.
2. **Then B** (condition) and **A2** (disruption robustness, single policy) if C flips.
Reuse the resumable chain pattern (`run_robustness_chain.sh`), `caffeinate`, monitor.
Wallclock: C at 500k, 5 seeds × 2 learned ≈ 15 h; stage sequentially overnight.

## Stage 2 — If it's a real gap (Stage 0 ⇒ plateau)

The honest paper is still publishable and referee-defensible:
- **RQ1 stands** — graph ≫ flat robustly (2–2.3× across every regime), the encoder is the
  contribution; the benchmark + fair strong-baseline suite is a reusable asset.
- **Honest negative** — tuned heuristics dominate learned control at all tested budgets;
  report it as a finding, not a failure (strong non-crippled baseline is the point).
- **Levers to list as future work (do not commit compute now):** patient-outcome reward
  shaping; scale-curriculum warm-start (2→20 already wired); temporal/condition-aware
  encoder (Phase 8, deferred); longer decision horizon. Each is a hypothesis for *why* the
  gap persists, not a promised win.

## Compute & staging

Per-seed wallclock (measured: 150k gcn_ddpg ≈ 25 min at 20 clinics; linear in steps):
150k ≈ 25 min · 300k ≈ 52 min · 500k ≈ 87 min · 1M ≈ 175 min.

- Stage 0a: 3 × 87 min ≈ **4.4 h** (the whole decision often rests here).
- Full Stage 0 (0a+0b+0c): ≈ **9 h**.
- Launch **after C** (CPU contention). Gate the runner on C's completion sentinel, or fire
  manually. All resumable; a sleep-kill loses only the in-flight seed.

## Invariants (unchanged)

Fair baselines (`umyo`/`fmyo`, true demand/forecast), true OOD where a generalization claim
is made, IQM + bootstrap CIs; a "DRL wins" claim requires separated CIs. Higher budget does
not relax any of these — it only tests explanation (U).
