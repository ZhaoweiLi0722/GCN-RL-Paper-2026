# Prospective integration acceptance and next decision

September 29, queue packet N3. **Software preparation only. No new experiment
is approved or launched.** The closed formal campaign, holdout, previous null
results and historical agents remain unchanged. Passing these tests is not
evidence of improved performance, corrected historical training, or a validated
manufacturing scenario.

## What is now implemented

- [N1](../2026-09-29-replay-repair-preparation/contract.md): explicit reward and
  replay semantics, true-adjacency checks, actual-length multi-step returns and
  a NumPy Bellman reference.
- [N2](../2026-09-29-matched-input-preparation/contract.md): common-information
  graph/flat/gate views, proposal conditioning and neural-only message ablation.
- N3: `src/rl/prospective_adapter.py` and
  `tests/test_prospective_adapter.py` join those contracts without an environment,
  optimizer, actual policy, cache loader or checkpoint. Tests use invented
  numerical fixtures explicitly labeled synthetic, not manufacturing data.

The adapter creates current actor/critic inputs and bootstrap actor inputs from
validated return windows. It does not evaluate the target actor or target critic.
Both future calibration and online-update implementations should use its one
target convention, but neither historical path has been replaced.

## Replay-to-input mapping

The `ReplayInputContract` requires both the complete `InputSchema` and the
`ReplaySemantics`. With `definition_id = D`, the replay state ID is exactly
`D/actor-flat`, and its action ID is `D/action`. Dimensions must agree with the
declared ordering:

```text
state = [node values (node-major, feature-major),
         globals, physical links (row-major), anchor action]
state width = N*F + G + N*N + A
recorded action width = A
```

The anchor is stored at each observation, not recomputed later with possibly
different policy/config information. A return uses the first recorded action and
the last record's next state, including that next state's own anchor. Next-state
values are allowed as learning targets after the transition, not as observations
available to the preceding decision. Requested, projected, gated and executed
actions must not share an ID unless they really have the same declared meaning.

`pack_actor_state` verifies the N2 schema and tensor contract before encoding.
`prepare_replay_batch` requires typed one-step windows; it validates N1 before
decoding and validates N2 again afterward. An invalid physical-link matrix does
not bypass checks just because it came from a numerically valid replay vector.
Dtype and device are explicit. Conversion overflow, invalid target shape and
nonfinite values fail. Target Q is detached; gradients must not cross bootstrap.

`completed_segment_windows` retains one window for **every** start position of
a declared closed trajectory segment, including all short tails. The segment
must explicitly end in termination or truncation. Adjacency, source and episode
boundaries are checked even when requested return length is one. Independent
counterfactuals are separate one-step windows, never a trajectory segment.
This helper is not a streaming collector, checkpoint queue or resume manager.

For each actual window length `n`, tensor targets are:

```text
R = sum(gamma**i * raw_reward[i] * reward_scale)
d = 0 if terminated or (truncated and not bootstrap_on_truncation)
d = gamma**n otherwise
y = R + d * detached_next_Q
```

No extra reward scaling or extra gamma is applied at this boundary. Tensor tests
compare against N1's NumPy reference and an independently coded backward
recursion across dtype, discount and termination/truncation cases. This verifies
the declared arithmetic, not that a chosen reward/discount matches the paper's
objective. Runtime consumers must not mutate returned tensor fields.

The N3 adapter does not build gate-training examples from behavior actions.
Behavior actions may already be gated or projected; they are not automatically
the actor's pre-gate proposal. A future collector must retain those quantities
separately to test the actual N2 gate path.

## Historical data cannot be auto-upgraded

| Missing or conflicting evidence | Required treatment |
| --- | --- |
| Independent counterfactual records without genuine successor lineage | Never concatenate; do not invent source/episode/state-token metadata |
| Counterfactual reward horizon or continuation not verified | Do not call the reward one-step just to satisfy this API; recover generation evidence or exclude it from prospective Bellman replay |
| Absolute versus anchor-relative rewards | Do not pool under one critic contract or relabel numeric values. A scientifically specified conversion would need the necessary original outcomes |
| Unknown anchor/config/normalization version | Do not guess the anchor from a current implementation; preserve the old artifact and mark new-use eligibility unresolved |
| Old graph/flat inputs with differing features or gate proposals | Shapes or parameter counts are insufficient. A producer audit and explicitly matched new representation are necessary |
| Missing terminal/truncation classification | Do not infer it from a filename or zero successor vector |
| Existing ranker/teacher supervision | Keep its label horizon/support/provenance distinct from Bellman replay; this adapter does not certify it |

The unchanged historical evidence is useful for reporting and diagnosis. It is
not automatically eligible for a revised learning pipeline. Matching observation
values, IDs and declared tokens are assertions rather than proof of simulator
and RNG continuity. Exact producer/clone audits remain necessary. Persist the
full schema/contract manifest at collection and resume; a reused definition ID
with changed units or column meaning defeats this prototype's checks.

## Next work that would require a new approval

**Recommendation: approve a bounded prospective implementation/verification
amendment first, not a full training campaign.** The immediate objective would
be to make one real producer and opt-in agent path satisfy these interfaces.
Keep the primary manuscript's executed method and numbers distinct from the
revised implementation. Do not change a simulator merely to make RL win.

The following choices are not locked by N1/N2/N3:

| Decision | Recommended constraint; remaining choice |
| --- | --- |
| Reward / horizon | One explicit objective for prospective critic data. Prefer starting from documented environment cost increments; justify any anchor-relative alternative. Freeze gamma, terminal obligations and discounted-versus-reported-cost interpretation before a scientific run |
| Offline supervision | Do not migrate the uncertified cache into multi-step Bellman replay. Decide whether verified supervised labels are retained, fresh trajectories are collected, or a new offline dataset is needed; account for its cost and provenance |
| Model parity | One real observation producer and action transform; explicit critic anchor and pre-gate proposal in both arms. Choose readout/head treatment and parameter budget without looking at new outcomes |
| Integration scope | Separate opt-in collector/agent path, no replacement of historical defaults. Approve any bounded mechanics smoke separately from outcome-oriented training/evaluation |
| Scenario | Use a defensible lever and information model. Qualified setup-support booking remains a hypothesis, not calibrated continuous production control |
| Study contract | Lock seeds, fresh development streams, run budget, endpoints, reset rules, compute accounting, stopping policy and approval before launch. None are selected by this packet |

Implementation acceptance for a real path must directly capture its calibration
and update targets and compare both with this reference; audit real state/RNG
lineage, reward availability, actor proposal versus executed request, target
network detachment, queue tails, resume behavior and matching features. Passing
the isolated harness cannot substitute for these tests.

## Scenario and learning gates

The [E1 measurement contract](../2026-09-29-qualified-support-contract/decision_contract.md)
still lacks three operational evidence bundles: a real task/bookable lever;
time-available task/staff records; and units, economics, persistence and closure.
Do not treat the existing toy queue as calibration, or create continuous staffing
control when crews/slots are the actual decision. Its solved cells do not need
DDPG; the simple-rule gap elsewhere does not prove that RL is necessary.

For any later approved scenario, require in order:

1. Correct mechanics/accounting and a material action-to-outcome effect.
2. Safe headroom and state-dependent value that replicate under independent
   changes, with no withholding of information already available in practice.
3. Competent adaptive rules and censored-data system-identification plus MPC,
   using the same available information and their own realized histories.
4. An adequately trained frozen history/belief-conditioned policy. Changing
   actions using memory is not evidence of online parameter-update benefit.
5. A bounded frozen-versus-online DDPG contrast initialized from identical policy
   parameters, with matched observations, action execution and evaluation rules;
   disclose online queries, replay/optimizer updates, resets and wall time.

Realized histories can differ because policies choose different actions. Match
the available information and exogenous test worlds, not a privileged realized
trajectory. Diagnostic oracles must be labeled as bounds, not deployable agents.
Benefits must be practically meaningful and reproducible across independent
training/change realizations, with prespecified uncertainty and guardrails.
Null results or a stronger conventional controller are valid outcomes to retain.

## Reproduce acceptance

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest tests.test_prospective_adapter tests.test_validated_returns \
  tests.test_matched_inputs -v
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
git diff --check
```

No fresh training seeds, experiment configs, outcome rows or performance claims
are produced by this harness. Fourteen new acceptance tests and the combined
154-test regression suite passed, together with full Python compilation and
`git diff --check`. The regression includes N1/N2, historical contract/evidence
checks, offline replay, service-effort/queue checks and existing GCN feature/head
tests. Existing tracked source, execution configs, evidence and reports are
unchanged from `1472083`; the only new source is the opt-in adapter. A reference
scan finds no existing agent integration. N3 finishes the finite preparation
queue; do not keep the automation running to invent an unapproved experiment.
