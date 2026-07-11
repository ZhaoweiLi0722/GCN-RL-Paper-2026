# Full campaign scaling plan (post-pilot)

Date: 2026-07-11. Owner: Howard. Status: draft, pre-campaign.

## 1. What the pilot found

Lean two-stage pilot, 30k training steps/seed, 5 seeds (`results/patient_pilot/`).

- **Stage A (2-clinic screen):** eligibility saturates ~0.917 across all stable
  methods; `gcn_td3` promoted as flagship (best graph backbone, 0 divergent seeds).
- **Stage B (20-clinic confirm):** the heuristic **`mdl2` wins decisively** — elig
  IQM 0.823 / cost 9.84e8 — while every learned policy collapses:
  `gcn_td3` 0.391 / 3.10e9, `gcn_ddpg` 0.430 / 2.99e9, `flat_ddpg` 0.308 / 3.04e9.

**Diagnosis: undertraining, not instability.** `gcn_td3` had **zero divergent
seeds** (stably underfit) and was competitive at 2-clinic under the same budget.
The 20-clinic problem has ~10x the state/action dimension (action ≈ 4 × 20 = 80),
so 30k steps is far too few.

**Impact:** as-is, RQ5 ("do learned policies beat strong look-ahead heuristics?")
answers *no* at scale. The pilot did its job — it caught this before the full
campaign spend.

## 2. Two levers

### Lever 1 — Budget (verified-safe, encoded now)
`evaluation/campaign_manifest.py` scales the training budget and fixes the roster.

| Budget | min/seed | Serial (5 learned × 5 seeds) |
|-------:|---------:|-----------------------------:|
| 150k   | 22.5     | ~9.4 h  |
| **300k (floor)** | **45** | **~18.8 h** |
| 500k (flagship confirm) | 75 | ~31.2 h |
| 1M     | 150      | ~62.5 h |

Rates measured on pilot Stage B (single CPU, ~9 s/1k steps/seed). Replay buffer
(1M) and LRs already default-sane; **budget is the only change in Lever 1.**
Recommended: **300k floor for the family**, **500k for the flagship confirm**.

### Lever 2 — Representation + curriculum (VERIFIED, wired)
The graph actor defaults to `readout_mode="global_flat"`: head input is
`num_facilities × encoder_dim` (~1280-dim at 20 clinics), i.e. **not size-invariant
and parameter-heavy** — a plausible co-cause of the collapse. The `facility_action`
mode applies a **shared per-facility head** (fewer params, permutation-equivariant,
transferable across clinic counts).

**Layout gate — PASSED.** `facility_action`'s `transpose(1,2).flatten` emits a
type-major vector `[comp0×N, comp1×N, ...]`, which matches the env's decoding of the
normalized action as `[w(0:N), e(N:2N), q(2N:3N), p(3N:4N)]`. Certified in
`tests/test_gcn_facility_action.py` (layout, size-invariance, transfer) — 5 tests.

**Curriculum warm-start — VERIFIED end-to-end.** `transfer_matching_parameters`
(in `src/models/gcn.py`) copies every shape-matching weight; the deterministic
backbones expose `warm_start_actor` (`gcn_td3`, `gcn_ddpg`). A 2-clinic
`facility_action` policy transfers the **full** policy into a 20-clinic agent —
*only* the non-learnable `encoder.adjacency` buffer is skipped (rebuilt for the
target graph) — **provided the 2-clinic env has matching per-node features**. The
pilot's 2-clinic screen env does not (it uses `production_lead_time=2`, no hub), so
use the matched **`experiments/configs/2_clinic_patient_condition_curriculum.json`**
(identical per-node structure to the 20-clinic env). With it, the input layer also
transfers. Wired in `evaluation/campaign_manifest.py`
(`curriculum_pretrain_config`, `GRAPH_READOUT="facility_action"`).

**Scope / follow-up:** `facility_action` currently covers the deterministic GCNActor
backbones (`gcn_td3` flagship, `gcn_ddpg` ablation). `gcn_sac`/`gcn_ppo` keep
`global_flat` until their stochastic actor classes gain the readout.

**Still to confirm empirically:** correctness is proven, but that `facility_action`
*trains better* than `global_flat` at scale needs an actual run — do a short 2-clinic
`facility_action` vs `global_flat` comparison before committing the campaign budget.

## 3. Recommended campaign sequence

1. **Re-confirm at scale (Lever 1 only):** full family at 300k, `gcn_td3` flagship
   at 500k, vs `mdl2` and the other heuristics. If a learned method beats `mdl2`,
   RQ5 stands and we proceed to the stress/forecast scenarios.
2. **If still trailing `mdl2`:** land Lever 2 (verify + adopt `facility_action`,
   add the 2→20 curriculum warm-start), re-run the flagship.
3. **If learned methods still can't beat `mdl2` after 1+2:** report honestly and
   pivot the empirical spine to **graph-vs-flat** (does the GCN encoder help RL?)
   plus the **testbed contribution**. The literature (`Stranieri2025PharmaDRL`,
   already cited) supports that RL does not uniformly dominate tuned heuristics, so
   an honest result is defensible, not fatal.

## 4. Do NOT launch the big run at 30k steps.
The scaled budget is the whole point of this note.
