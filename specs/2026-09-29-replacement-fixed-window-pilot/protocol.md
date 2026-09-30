# R3 execution amendment: replacement frozen-policy value pilot

Zhaowei explicitly replied "continue" to the named data-only pilot approval
question on 2026-09-29. This authorizes collector acceptance and one capped
attempt. It does not assert Howard approval or authorize online training.
Original R3 design, missing-F1 record, R4/R5 evidence and Stage E stay unchanged.

## Fixed scope

Use all three R4 replacement actors/gates, not historical F1 weights. R5 proves
sampled loading compatibility only, not their performance or competence. Input
hashes and unchanged simulator/model source are inherited from the R5 input
lock config and its R4 manifest. No critic/optimizer/replay is restored.

Use the exact stored nominal-history environment dictionaries, retain and hash
the actual built dataclasses, and require matching environment hashes across
all three actors. Do not apply their training hotspot randomization. Decisions
are t=0,13,26,39 on one fresh frozen trajectory per actor. All other R3 choices
remain fixed: 52-step incurred absolute cost, scale1e-9, gamma1, six ordered
requests, 8 discovery + 8 validation RNG starts per state. Native done stops
rollouts. Outstanding obligations remain explicit; no lifecycle claim.

Limits: 12 states, 1,152 logical continuations, 37,596 actual simulator steps,
3,600 seconds, one serial scientific attempt, zero optimizer updates. A failed
attempt is preserved, not repaired and rerun automatically. No formal stream,
reward change, extra state/scenario, fitting, or automatic training gate.

## Seeds and coupling

The config records a 112-bit namespace derived from a fixed string, with
offsets 0..2 for trajectories and 3..194 for actor/step/block/draw in that order.
These are NumPy PCG64 seed entropies, not torch/random legacy uint32 seeds.
Before recording, scan prior tracked source/config/document numbers against
the namespace and all 195 explicit seeds; save scanned file hashes and hits.
Prior registered low-integer CRN streams are not reused. Unrecorded external
streams cannot be certified. Engineering uses offset65535, disjoint from pilot.

Continuation clones replace only env.rng, retaining the sampled state and
episode/patient identities. Patient-ID-keyed transport losses remain fixed;
only RNG-driven futures are independently initialized across blocks. Shared
initialization is not event-aligned CRN after action-dependent draw consumption.
Intervals are conditional on these states and fixed keyed losses, not population
uncertainty, independent episodes, or clinical noninferiority certification.

## Action identity and support

Order: frozen, MDL2-first, specimen-.05, +.05, -.10, +.10. Other action slices
stay anchored. Preserve request projection and operational constraints. For
each draw execute each first request once, save its real route events and raw
costs, then deduplicate only if full post-step physical/RNG state, route events
and cost agree. Ignore only cumulative_blocked_specimen_requests in that hash:
source inspection shows it is a reporting counter, never observed or causal.
Each alias retains its own first-step trace and references its representative's
remaining tail. No extra exploratory step is used to determine identity.
Deduplication is draw-conditional, not asserted for unobserved random futures.

Certify the frozen action by construction. For other requests test the inverse
residual proposal through the unchanged hard gate and deployment map, within
the [-1,1] proposal box and request error1e-6. A successful witness certifies
this map only, not realizability by a shared neural network or learnability.
An unsuccessful witness is diagnostic-only/uncertified, not a proof of global
unreachability. No candidate is deleted based on reachability or clinical cost.

## Engineering acceptance and storage

Unit tests use synthetic data. A separate real seed60 MPS smoke runs two
identical two-step paths for each of the six first requests: exactly24 steps,
offset65535, no outcome selection. It checks cloning/RNG/reward/trace identity,
unchanged weights, and early truncation. This is not scientific outcome data.
Freeze implementation/config/protocol before smoke and pilot. The pilot may
start only after this acceptance and independent tests pass.

Save source snapshot, runtime, input locks, actual env configs, stream inventory,
12 full selected snapshots/observations, chronological compressed traces with
raw cost components and clinical info, requests/actual route identities,
selection files written before validation, and unfiltered outcomes. No source
or artifact overwrite. Archive with per-member SHA256 and Dropbox local-byte
verification; cloud sync and collaborator access are separate unknowns.

## Prespecified analysis

Choose the lowest discovery mean cost, first-index ties, and seal that JSON
before collecting validation for the state. Independently recompute raw costs,
logical aliases, selections, and draw-paired validation differences vs frozen
and MDL2-first/frozen-followup. Cost delta <0 is improvement. Report all12
selections, per-actor descriptive signs, clinical losses/completions and cutoff
service/manufacturing-loss rates. No filtering or changing choice after seeing
validation. Report sample SD/sqrt8 and descriptive t7 intervals (2.364624251)
for the conditional mean; they are not multiplicity-adjusted, powered efficacy
tests, nor independent-state aggregate inference. Also report certified and
uncertified witnesses separately, without selecting a new policy post hoc.

Stop after independent verification, decision memo and archive. Positive labels
would justify discussing a separate critic-learning gate, not show online RL
benefit. Null/negative labels are retained without reward tuning or expansion.
