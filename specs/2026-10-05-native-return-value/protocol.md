# Native-Return-Grounded Value Comparison

Status: engineering prepared; new numerical scope NOT approved. The literal
request `继续下一步` authorizes this additive preparation, not an unpresented
18-hour scientific package. No scientific model was loaded or scored and no
native patient environment or optimizer was run during preparation.

## Question and Trigger

The completed fixed-parent public-model-tail comparison failed its primary
criterion. Persistent cost intervals versus H8, existing frozen value-MPC and
matched adaptive-TD crossed zero; existing-frozen also had mean extra patient
losses. That result does not prove a reward defect or establish a causal source
of failure. It motivates one different, bounded test: does learning from actual
native-simulator returns, rather than only public-predictor returns, improve
cost and patient outcomes against competent controls?

Old action: refine only public-predictor continuation labels. New action: add
trainer-only native continuations at the same prescribed reference roots.
Reward weights, physical scenario, observation schema, graph architecture,
candidate support, settlement and evaluation metrics remain unchanged. E1 field
calibration remains absent. No old source, result, failure or archive is changed.

## Fixed Design

- Five historical GCN ancestors listed by path/hash in the config. Each has 1536
  historical value updates. Use identical within-block weights with fresh Adam,
  sampling state and new-update count for three arms. No outcome-based selection.
- 120 new shared plain-H8 reference worlds: 24/block, three conditions, eight
  references/condition/block. At reference index i, roots are i and i+24. Candidate
  is `(i + 8 * root_slot) % 16`, one candidate/root, not a ranking search.
- At each root clone the complete native state and RNG on the same future tape.
  Clone once, never construct/reset or replay from time0. Clone the controller
  through explicit state restoration into an unwrapped controller, not its
  runner-owned act closure. Private state is not exposed to learner inputs.
- Execute the H8 candidate prefix. Switch candidates change to public adaptive
  allocation after offset2. All actions at epoch48 or later are zero support
  allocation, as in the inherited settlement policy. Observe every public
  receipt and record the unchanged common operation throughout the prefix.
- Thereafter run the frozen historical parent's H8 MPC to epoch48, followed by
  unchanged settlement through64. Exclude the forced prefix from suffix targets.
  Preserve original float64 costs/requests, full patient identities, raw labor
  receipts and failure/8-step/full boundaries. Restore reference-side global RNG.
- 240 native branches share the 120 reference tapes. They are dependent partial
  continuations, not 240 independent new worlds or event-level CRN guarantees.
- Native TD8 and native MC use the same 7800 suffix rows, same update count and
  same sampling indices within block. MC uses complete native suffix return;
  TD8 bootstraps from a frozen pre-fit value snapshot, with terminal bootstrap0.
  Targets/loss remain float64; differentiable network outputs are cast from
  float32. Each method gets 32 updates/reference, 3840 total, 768/model.
- A forecast-TD control uses the existing paired public-tail collector at the
  same root/candidate and three quantiles. Its frozen-parent tails alone train
  the control. The collector's adaptive companion is retained and fully charged,
  but does not create an extra learner, screen or scientific follow-on.
- All 15 final models and five ancestors seal before any evaluation. Evaluate
  60 new worlds (four/condition/block), each with six frozen controllers:
  plain-H8, existing frozen GCN-value-MPC, forecast-TD, native-TD, native-MC and
  plain-H16. Total360 evaluation trajectories; test optimizer updates0.

Native-observed and forecast-endpoint features share a schema, not a distribution.
The forecast/native contrast changes states, transitions, row counts and cost
targets, not only labels. Deployment still scores forecast endpoints. It is
native-grounded versus forecast-grounded training, not proof of an exact model
bias correction. Native TD versus MC is the matched-data estimator contrast.
Fixed-parent evaluation is not exact evaluation of the changing student policy.

## Streams and Inputs

Config: `experiments/configs/capacity_native_tail_20261005.json`.
Runtime namespace `capacity-native-tail-20261005-v1`; reference base65100000,
evaluation base65120000. World enumeration and keyed substreams use the unchanged
explicit design helper; branch tapes intentionally inherit the reference.
Sampler seeds551270510/530/550/570/590, bootstrap seed65190001.
Prospective manifest has180 worlds and1091 unique allocations. Initial bounded
collision check against1991 declared seed/config files passed; approval-time
freeze will bind the exact then-current manifest. No reseeding after approval.
The inherited physical/reward proposal and five input models/metadata must match
the configured hashes. CPU float32, four threads, one worker, RSS4GiB.

## Complete Nonrefundable Budget

| Resource | Maximum |
|---|---:|
| Complete reference/evaluation trajectories | 120 + 360 = 480 |
| Native partial branches / clone admissions | 240 / 240 |
| Main native steps / branch steps | 30720 / 9720 |
| All native steps / operations including main construction/reset | 40440 / 41400 |
| Value optimizer / actor optimizer | 11520 / 0 |
| Optimizer example presentations | 737280 |
| Neural module forwards | 40045 |
| Main planning prediction steps | 9953280 |
| Forecast-parent nested planning steps | 4723200 |
| Native-parent nested planning steps | 1574400 |
| Forecast prefixes / paired suffix steps | 5760 / 46800 |
| All prediction steps | 16303440 |
| All filter hypothesis transitions | 6960000 |
| New final seals / ancestor bindings | 15 / 5 |

The ledger separates native steps from forecast calls, and native branch calls
from main-world calls. Full forecast/filter chunks are reserved before dispatch;
failed or partial chunks are not refunded. Cloning is separately counted even
though it does not call the native constructor/reset. Nested native parent
planning is4100 decisions, not240 nominal branch starts; forecast parent has12300.
605 TD bootstrap forwards and11520 fitting forwards are included. H16 consumes
twice the H8 planning queries. Same TD/MC updates do not mean equal forward work.

| Wall owner | Seconds |
|---|---:|
| Admission and lock binding | 600 |
| Reference plus all label generation, clones and recording | 34200 |
| Value fitting and seals | 3600 |
| Frozen evaluation | 18000 |
| Analysis and single archive | 2400 |
| Failure preservation and shutdown reserve | 6000 |
| Total, including I/O | 64800 (18 hours) |

Per-owner caps: reference world including all branches900s; H8 evaluation180s;
H16 evaluation300s; fit60s; seal60s. Owners/phases cannot borrow or refund budget.
Disk caps raw6GiB + archive6GiB = combined12GiB, at most9000 files. These are caps,
not a promised runtime or a measured resource requirement.

## Decision and Reporting

Preserve the previous primary screen: native-TD must have positive persistent
cost savings in all five blocks and a positive descriptive95 lower bound versus
each of plain-H8, existing frozen and matched forecast-TD, with no mean extra
patient losses in any condition for these pairs. Bootstrap2000 over five blocks,
not the old three-block code. Report all15 controller pairs x3 conditions x5
blocks, harmful worlds/blocks, cost components, losses/deliveries, requested,
committed and applied hours, decision latency and all training/label compute.

MC and long-H16 are required interpretation controls. An advantage shared by MC
does not isolate TD's benefit. H16 is a higher-query compute reference, not an
equal-budget opponent. Historical training blocks are not new independent
training replications. This is synthetic simulator value learning, not clinical
groundtruth, deployed online adaptation, DDPG, full TD-MPC or isolated GCN proof.
The complete native raw branch reader is ancillary, not another fitting gate.

One attempt only. Any terminal failure ends this attempt; no repair-and-retry,
extra epochs, sample expansion, best-checkpoint selection, reward search or
automatic next experiment. Save evidence and close on either outcome, update
the manuscript only from complete supported results, and pause the same monitor.

## Authorization and Launch

The single execution question covers this entire package. On an exact subsequent
approval: record literal approval, bind proposal/protocol, commit source; freeze
source/runtime/inputs/seed manifest and commit authorization; launch once into
`results/capacity_native_tail_20261005`. Complete collection, fitting, sealing,
evaluation, raw analysis and archive without another routine startup question.
No approval is implied for Howard, StageE, formal holdout, remote compute,
push/PR/merge, Dropbox or external messages. Old attempts remain closed.
