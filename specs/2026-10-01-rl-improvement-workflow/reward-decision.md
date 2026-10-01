# Reward audit and decision

2026-10-01. Source and saved-report review only, with no new patient episode,
model fit or counterfactual outcome. Keep the current reward unchanged while
testing elementary sampled learning. Its arithmetic is consistent with the
declared finite-horizon objective; that is not validation of its economic or
clinical interpretation. Before a new patient pilot, explicitly settle whether
the target remains a 52-step objective or becomes a complete-lifecycle target.

## Verified definition

The actual P2 effective execution config, not source defaults, declares:

- Negative absolute environment step cost; one positive scale of 1e-9.
- Gamma 1, horizon 52, terminal boundary, zero terminal bootstrap and zero added
  terminal cost. `full_lifecycle_claim=false` and `cost_weights_changed=false`.
- Original 20-facility scenario and 80-dimensional submitted requests.
- R4 input configs for all three blocks 60/61/62 use patient-loss coefficient
  500000, expiry coefficient 100000 and urgency coefficient 25000. These are
  simulator objective coefficients, not newly validated clinical valuations.

`src/env/patient_capacity_planning.py` forms operating cost plus patient-loss,
expiry and urgency terms, then returns negative total cost. The collector
checks negative reward against total cost and the component sum. Scaling occurs
once in `src/rl/candidate_rollout.py`; terminal flags suppress bootstrap.
All four inspected source hashes match P2's execution locks.

The existing independent training diagnostic covers 9 PPO models, 288 training
episodes and 14,976 step records. It reconciled saved return arithmetic with a
maximum error of 1.2662e-6 in scaled return units. This review re-read that saved
diagnostic and source; it did not newly re-evaluate raw trajectories.

## Objective composition and unresolved meaning

The following are ranges across the nine saved training-model aggregates,
not causal action effects or uncertainty intervals:

| Component | Share of recorded total cost |
| --- | --- |
| Patient loss | 43.98% to 44.38% |
| Bioreactor shortage | 42.73% to 43.30% |
| Material expiry | 8.78% to 8.86% |
| Urgency | 0.319% to 0.330% |

Two substantive questions remain.

1. **Shortage proxy versus actual loss.** The shortage term uses unmet
   specimen/capacity load, while the patient-loss term uses realized losses.
   They are not duplicate dictionary entries or a demonstrated arithmetic bug.
   They could represent distinct operational and outcome costs, or partially
   overlapping penalties. Independent rationale is missing. Their large shares
   do not establish which term drives useful action gradients; neither term
   should be removed or downweighted merely to make RL improve.
2. **Unfinished obligations.** Remaining active patients are recorded, but
   not valued after the 52-step terminal boundary. This is consistent with the
   frozen finite-horizon protocol. It cannot support a complete-lifecycle claim.
   The previous PPO policy did not change greedy decisions, so this is not
   evidence that PPO learned to defer patients or exploit the boundary.

The cost/loss/completion/terminal-active trade-off in the P2 terminal audit must
remain visible. No clinical noninferiority or acceptable trade-off threshold
has been established by these numerical coefficients alone.

## Decision before each next stage

| Stage or finding | Decision |
| --- | --- |
| Artificial sampled-return control | No reward change; isolate basic sampling and learning capability |
| Dynamic request interface | No reward change; qualify action identity, precision and ranking separately |
| New finite-horizon patient pilot | Freeze the same objective explicitly; publish raw cost, losses, completions, waiting and terminal-active obligations separately |
| Intended claim requires full lifecycle | Amend the protocol before the pilot: justify a continuation or terminal valuation and prevent double counting; new approval and fresh data required |
| Valid objective, usable actions, delayed/noisy credit remains | Propose one bounded credit-assignment/shaping study; do not change coefficients post hoc |
| Patient outcomes worsen while weighted cost improves | Report trade-off, do not declare overall success; justify constraints or tolerances before confirmation |

For now the strongest demonstrated obstruction is the previous policy's
insufficient score range relative to its fixed prior, not an accounting error.
The next diagnostic tests whether sampled rewards support improvement after
removing that obstruction in an invented task. A pass will not establish
patient headroom, successful long-horizon credit assignment or deployment
adaptation. A failure will not license an automatic reward search.

## Evidence

- `results/candidate_reference_prior_pilot_20261001/payload/locks/effective-execution.json`
- `reports/2026-10-01-p2-training-diagnostic/result.json`
- `reports/2026-10-01-reference-prior-integration/terminal-audit.json`
- `specs/2026-10-01-rl-improvement-workflow/review-inputs.json`
- `specs/2026-10-01-p2-training-diagnostic/readout.md`

This audit does not approve new optimization, reward changes, a patient pilot,
clinical use or manuscript claims of online RL effectiveness.
