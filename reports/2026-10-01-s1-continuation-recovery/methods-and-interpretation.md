# Continuation Recovery: Methods and Interpretation

Methods text prepared before continuation outcomes were inspected, for the
approved packet at execution commit
`63c2383d33f5795f22ec1173bd75222a30a63a4a` (parent 51957 / child 51969).
Completed scientific results and separately tracked archive/closure status
are in the **[canonical readout](readout.md)**. This guide retains the design
and interpretation rules without duplicating result tables or adding a gate.

## Methods Text

The continuation study evaluates whether simulator-interaction reinforcement
learning improves a graph-aware routing selector beyond its own qualified
initializer and continued imitation. The fixed setting is a 20-facility,
52-step patient-indexed manufacturing simulation (`routing_nominal_history`).
The selector uses public observations and the specimen-transport graph
(`specimen_routes`). Its dynamic candidate bank comprises R4, full MDL-2,
and four prespecified specimen-transfer corrections (+/-0.05 and +/-0.10).
At most six original requests are canonicalized by integer requested-net
classes, not by realized-action equivalence. Non-specimen request components
are common within each bank; no hidden feasibility mask is added. Thus,
reagent, capacity, and replenishment dynamics remain in the simulator, but
the learned intervention is restricted specimen coordination, not independent
joint optimization of all operations. R4 retains its historical representation.

Three training blocks (60, 61, 62) reuse their exact saved R4-imitation
initializers without refitting or repeating qualification. Within each block,
identical actor, critic, and buffer states initialize own-frozen, own-PPO,
and own-BC-CONTINUE branches. The actor and independent critic have 31,346
and 28,721 parameters, respectively (encoder width 16; head width 32).
Continuation uses fresh Adam optimizer moments and matched initial PPO/BC
sampling states. Own-frozen receives no updates. Each continuation arm uses
32 full simulator episodes per block, organized as eight four-episode rollouts,
with four epochs and minibatches of 64/64/64/16 rows per rollout. PPO performs
128 actor and 128 critic updates per block; BC-CONTINUE performs 128 actor
updates using R4-class cross-entropy on states visited by its current sampled
policy, with its critic unchanged. Interaction and actor-update budgets are
matched, but total computation and subsequent trajectories are not identical.

The reward is negative absolute simulator step cost scaled by 1e-9, with
unchanged cost weights, gamma = 1, GAE lambda = 1, zero terminal bootstrap,
and no added terminal cost. PPO uses fixed sampled behavior advantages and
terminal Monte Carlo returns, with advantages normalized once per rollout;
there are no counterfactual labels or extra critic warmup. Adam learning rate
is 0.0003, the separate actor/critic gradient norm cap is 0.5, and PPO clip,
entropy, and value coefficients are 0.2, 0.01, and 0.5. Computation uses the
frozen deterministic CPU float32 runtime. Unresolved terminal patient
obligations are reported separately rather than treated as completed care.

The fixed final-only artifacts for all nine learned branches are sealed before
evaluation. Five controllers (own-frozen, own-PPO, own-BC-CONTINUE, R4, and
full-MDL2) are evaluated on twelve paired development-test world starts per
block: 36 distinct starts and 180 controller-episodes in the complete design.
Learned selection is greedy with lexicographic class-key tie breaking; there
are no test-time parameter updates or checkpoint selection. Pairing means
identical starting world seeds, not identical event-level random realizations
after actions diverge. Preflight, training, and test roles remain disjoint;
historical qualification outcomes are not pooled into this evaluation.

## Evidence and Scope

The [saved qualification readout](../2026-10-01-s1-saved-qualification/readout.md)
reports 104/104 overall and multiclass greedy class matches on each of six
saved paths (R4 and initializer-greedy paths for each block), with zero reused
paired differences in cost, losses, completions, and terminal-active patients.
This establishes initialization fidelity on those paths, not identical policy
probabilities, generalization, or an RL gain. The 624 scoring operations are
not independent trained replicates. These are reported saved findings, not
rescored evidence in this document.

Simulator-trained RL updates policy parameters using simulated interaction
before evaluation. Deployment adaptation would update parameters during use;
this study does not test it. State-responsive actions from a frozen policy
are not parameter adaptation. Initializer imitation and BC-CONTINUE learn
R4 action labels; their gains must not be renamed RL gains. PPO versus its
own frozen initializer measures the continuation increment, while PPO versus
BC-CONTINUE addresses whether that increment exceeds continued imitation
under the stated budgets, not under equal total compute.

This graph-only mechanism comparison cannot isolate message-passing value,
establish superiority over a corrected DDPG method, demonstrate full joint
control, or establish clinical noninferiority or deployment readiness. The
[existing claims draft](../../specs/2026-10-01-adaptive-paper-delivery/manuscript-claims-draft.md)
keeps historical package-level gains separate from the unestablished historical
final-versus-frozen increment; the present study neither replaces those results
nor pools its development observations with formal holdout evidence.

## Fixed Contrasts

For every endpoint, compute paired differences as first controller minus
comparator on the same block/world. Preserve all five contrasts and their
original priority regardless of the observed direction.

| Priority | Fixed contrast | Interpretation |
| --- | --- | --- |
| Primary | own-PPO minus own-frozen | Increment beyond the same saved initializer |
| Secondary | own-PPO minus own-BC-CONTINUE | Increment beyond continued R4 imitation |
| Secondary | own-PPO minus R4 | Comparison with the reference package |
| Secondary | own-PPO minus full-MDL2 | Comparison with the full anchor |
| Secondary | own-frozen minus R4 | Initializer/reference difference on new test worlds |

For primary cost, let `d_b` be the mean of the twelve paired cost differences
in block `b`, and `f_b` the mean own-frozen cost in that block. Report raw
block differences and their equal-block mean. The prespecified relative
change is `mean_b(100 * d_b / f_b)`, not a percentage obtained by pooling
denominators. A nonpositive `f_b` makes the relative endpoint undefined, not
passed. The existing development screen requires relative change <= -1%
and favorable cost direction in all three blocks; adverse patient direction
means a trade-off, not overall success. This is not a clinical margin.

Use the fixed 10,000 hierarchical paired block-then-world bootstrap draws:
three sampled blocks and twelve sampled worlds within each sampled block,
with the original analysis seed and controller pairing retained. Report the
saved interval definition and uncertainty output without substituting another
resampling scheme. Three trained blocks provide low-precision descriptive
development uncertainty, not 36 or 180 independent training replicates and
not confirmatory, equivalence, or clinical evidence.

## Outcome Reporting

The [canonical readout](readout.md) replaces the unfilled reporting template
and owns all numerical result tables, saved-data reconciliation, the linked
coordinator-owned post-hoc diagnostic, and archive/closure fields.

Retain controller means and every paired world/block alongside summaries.
Lower cost, losses, waiting, and terminal-active burden are favorable;
fewer completions are adverse. Report adverse components explicitly even
when the aggregate cost improves. Costs are modeled objective units, not
validated monetary savings. The approved one-attempt ceilings (387 new
episodes, 20,184 environment calls, 1,152 optimizer calls, 17,400 seconds)
are budget limits, not measured utilization or evidence of completion.

## Interpretation Wording

- Favorable versus both own-frozen and BC-CONTINUE: "The results support a
  PPO continuation benefit beyond the shared initializer and continued
  imitation within this restricted simulator development comparison."
  Qualify this with effect sizes, uncertainty, block consistency, patient
  directions, and unequal critic computation; do not imply confirmation.
- Favorable versus own-frozen but not BC-CONTINUE: "Continuation improved
  on the initializer, but an advantage beyond continued imitation was not
  established." Reference-controller gains alone do not establish an RL
  increment.
- No supported own-frozen increment: "This bounded intervention did not
  establish incremental PPO benefit." This is not proof of equivalence,
  absence of learnable headroom, or ineffectiveness of RL in general.
- Lower cost with adverse patient direction: "The intervention produced a
  modeled cost/patient-outcome trade-off." Do not call it overall success
  or clinical noninferiority. Mixed block directions remain visible.
- Incomplete execution: "The attempt ended before the planned comparison
  was complete; available records document execution, not the full-design
  performance claim." No automatic retry or favorable-subset conclusion.

## Provenance

Scientific details remain in the
[continuation protocol](../../specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-protocol.md)
and [frozen packet](../../specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-frozen.json),
packet identity `8f0698cc8bdf41725e07ed78b0a856695473588cdcbbb817d90fa9d8c0223107`,
implementation `82a947404db18ba034fed241155336d4786b9898`.
The separate [authorization](../../specs/2026-10-01-adaptive-paper-delivery/continuation-recovery-authorization.json)
records approval at `2026-10-02T02:03:18Z`; the
[locked-plan amendment](../../docs/patient_indexed_specimen_routing_locked_execution_plan.md#complete-continuation-only-attempt-authorized)
records its scope. Retained preparation-era false approval flags and future
tense in earlier documents are historical metadata, not a new approval
request. The old S1 failure remains closed and A1 remains failed; this guide
does not relabel either as passed. No tests, scoring, model loads, simulations,
source changes, or commits were performed for this manuscript preparation.
