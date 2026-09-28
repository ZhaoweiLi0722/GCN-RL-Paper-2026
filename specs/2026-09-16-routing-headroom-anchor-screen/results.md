# Routing headroom and label-stability screen on MDL-2 versus MDL-3 anchors

Date: 2026-09-16 (run started 2026-09-15). Branch `e2b-run`. Diagnostic; no
training, no checkpoint, no protected seed stream, no existing artifact changed.
Tool: `evaluation/routing_headroom_anchor_screen.py`. Row file (30,720 rows)
under gitignored `results/routing_headroom_anchor_screen/`, SHA-256 in
`artifacts/summary.json`. Summaries and metadata are versioned in `artifacts/`.

## Question

Does specimen routing still offer state-dependent, replicable headroom when the
anchor is coverage-correct (MDL-3) instead of the under-covering MDL-2 used in
the formal study? This is the Stage 1 design of
`specs/2026-09-15-network-input-dependence/plan.md`, run on both anchors so the
answer is a paired contrast.

## Protocol

- Anchors: `mdl2` (lookahead 2, the formal anchor) and `mdl3` (`mdl2` with
  `lookahead_periods: 3`). Verified bit-identical to the stock class.
- Scenarios: the four routing-primary scenarios, composed as the formal
  benchmark composes them for heuristics.
- States: 6 anchor trajectories per scenario, snapshots at epochs
  4, 8, 12, 16, 24, 32, 40, 44 → 48 states per scenario, 192 per anchor.
- Options: anchor plus shifts −0.10, −0.05, +0.05, +0.10 on the mean-centred
  specimen pressure pattern (the G1 and label-budget ladder). States kept only
  if all five arms execute distinct specimen flows. Coverage was 100%: no
  state was dropped for either anchor.
- Continuations: 8 discovery and 8 validation worlds per arm, disjoint seed
  families (99.96M, 99.97M), anchor continues to episode end.
- Gates, written before the run: coverage ≥ 80%; best-action agreement
  ≥ 0.70 pooled and ≥ 0.60 in every trajectory; pairwise sign agreement
  ≥ 0.65 over pairs separated by ≥ 250,000 in both streams; prospective
  state-dependence value ≥ 0.5% of anchor remaining cost, positive in ≥ 4 of
  6 trajectories. Gate functions are the pre-existing
  `evaluation/label_stability.py` and `evaluation/state_dependence_value.py`,
  unchanged.

## Result

| | MDL-2 anchor | MDL-3 anchor | gate |
| --- | ---: | ---: | --- |
| best-action agreement, 8 vs 8 worlds (95% CI) | 0.698 [0.63, 0.76] | 0.615 [0.55, 0.68] | ≥ 0.70 |
| minimum per-trajectory agreement | 0.38 | 0.38 | ≥ 0.60 |
| pairwise sign agreement (material pairs) | 0.836 (1,050) | 0.841 (1,063) | ≥ 0.65 |
| prospective state-dependence value, % of anchor cost (95% CI) | +0.059 [+0.034, +0.083] | +0.019 [+0.001, +0.043] | ≥ 0.5 |
| trajectories with positive value | 20/24 | 16/24 | ≥ 4/6 per scenario |
| best constant option over anchor | +0.065% (shift −0.10) | +0.052% (shift −0.10) | |
| prospective per-state choice over anchor (95% CI) | −0.124% [−0.154, −0.095] | −0.071% [−0.099, −0.045] | |
| clinical screen (completion, lost, ineligibility) | pass | pass | |
| **verdict** | **fails: labels below floor** | **fails: labels below floor** | |

Per scenario, MDL-2 passes pooled agreement in nominal (0.77) and abrupt shift
(0.75) but fails the per-trajectory floor everywhere; MDL-3 fails pooled
agreement in every scenario (0.52 to 0.69). Both anchors fail the
state-dependence gate by an order of magnitude in every scenario.

Agreement as a function of world budget k (disjoint k-vs-k splits of the 16
worlds, 100 splits):

| k | 1 | 2 | 4 | 8 |
| --- | ---: | ---: | ---: | ---: |
| MDL-2 | 0.51 | 0.56 | 0.61 | 0.68 |
| MDL-3 | 0.46 | 0.52 | 0.59 | 0.65 |

Stage G1's 0.545 at a 3-versus-5 budget sits exactly on the MDL-2 curve. It
was budget-limited, but the curve has not reached 0.70 at k = 8 and is
flattening; the per-trajectory floor fails regardless of budget.

## Reading

1. **On the formal anchor, the routing channel's state-dependent headroom
   under this probe is about 0.06% of cost, and the whole five-option channel
   is about 0.12%.** The labels replicate at roughly 70% with 16 worlds per
   arm, not 54.5%, but not above the floor in every trajectory. This is
   consistent with the online-RL null: a bounded residual chasing a 0.06%
   state-dependent signal with labels that flip in a third of states.
2. **On the coverage-correct anchor the signal is roughly half as large and
   less replicable.** State-dependent value falls from 0.059% to 0.019%, the
   per-state choice's gain over the anchor from 0.124% to 0.071%, and
   agreement from 0.70 to 0.62. Fixing the anchor's coverage removes part of
   what routing was compensating for.
3. **Under the protocol's own gates, no learned policy should be trained on
   this channel on either anchor.** The Environment-first rule in
   `docs/online_rl_attribution_postmortem_and_followup_brief.md` §4 says the
   environment is redesigned, not the agent.

## What this does not show

- The probe is a single-step deviation followed by the anchor. A policy that
  corrects every epoch can exceed the sum of one-step gains; the formal GCN's
  −0.66% on MDL-2 is about five times this probe's whole-channel value on the
  same anchor. Scaling MDL-3's 0.07% by the same factor would suggest a few
  tenths of a percent, but that is an extrapolation, not a measurement. Only
  retraining answers whether the routing gain survives on MDL-3.
- Continuations restore the full patient registry, so labels are conditional
  simulator diagnostics with privileged latent state, not information-matched
  achievable gains. Real labels would be noisier, not cleaner.
- The ladder is coarse (four magnitudes on one pressure pattern). A richer
  option set could show more state dependence; the plan's Stage 2 ranking
  gate was not reached because Stage 1 failed.

## Erratum discovered on the way

`specs/2026-08-29-routing-label-budget-study/` sampled decision epochs 30, 55,
and 80 in a 52-epoch episode. Epochs 55 and 80 lie past the horizon, so 18 of
its 27 states had one-step continuations (mean remaining cost 67 to 74M versus
1,411M at epoch 30). Its 0.821 agreement figure is inflated by those
degenerate states and should not be cited. See
`specs/2026-08-29-routing-label-budget-study/ERRATUM-2026-09-16.md`. The
present screen's epochs are all inside the horizon, and its MDL-2 curve
(0.51 at k = 1 to 0.68 at k = 8) is the corrected reading.

## Decision this feeds

This is the fork in the plan of 2026-09-15: routing headroom on the
coverage-correct anchor is **not established**. The protocol answer is not to
retrain. If a definitive answer to "does the routing gain survive on MDL-3" is
wanted anyway, it must be a separately approved, bounded development
experiment (three seeds, MDL-3 anchor, same residual authority), knowing that
its own pre-training gate has already failed.
