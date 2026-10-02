# One-shot time-baseline comparison

Prospective numeric package. Direction approved; the exact numeric scope was
first presented to Zhaowei on 2026-10-02 and is not yet approved. Engineering is
not frozen. This document is not a launch permit or a reopening of a closed run.

## Why this test, and when it ends

The completed saved-data diagnosis found near-constant collection-time V values
despite strongly time-dependent returns. Normalized time does reach the critic.
This motivates one control-variate intervention, not repeated critic architecture,
learning-rate or reward search. Low explained variance is a clue, not causal proof.

For four independently seeded complete training trajectories under one behavior
policy, use `b_i(t) = sum(G_j(t), j != i) / 3`, and `A_i(t) = G_i(t)-b_i(t)`.
G is the existing full-horizon, once-scaled raw-cost return. Exclude the entire
owned trajectory before summation; never use test trajectories, other treatment
arms, hidden simulator state or counterfactual rollouts. All four episodes are
collected before updating. Normalize A once across the same208rows as original
PPO; old log probabilities remain fixed through its four optimization epochs.
This is a time-conditioned leave-one-episode-out Monte Carlo baseline, not a
newly learned value network, a reward change or online deployment adaptation.

The target transform alone changes. Critic MSE targets, architecture, optimizer
updates and reporting remain unchanged in both PPO arms, for matched compute.
The treatment's critic output no longer determines its policy advantage; do not
describe success as a repaired critic. The baseline does not condition on full
patient state, and four trajectories give a noisy estimate. It is a cheap,
specific test of whether removing common time-dependent return level helps the
policy learn useful action rankings. It is not guaranteed to reduce all gradient
variance. PPO clipping and sample normalization preclude a blanket claim of exact
finite-sample unbiasedness or monotonic improvement.

## Matched design

Reuse the exact three qualified graph initializers for blocks60/61/62 and their
R4 inputs. No initializer fit, qualification rescore or final-model selection.
Fork four learned controllers per block: own frozen, original learned-V PPO,
time-baseline PPO and BC-CONTINUE. All start with identical actor/critic payloads;
continuation optimizers are fresh. R4 and full MDL-2 remain contextual references.
Do not borrow the prior trained PPO result as the new active comparator.

Same52step20facility scenario, graph, candidate support, float64 submitted
requests, CPUfloat32, absolute-cost reward, gamma/lambda1, no added terminal
cost, entropy.01, Adam3e-4, PPOclip.2, gradientcap.5 and greedy test rule.
Each training arm receives32episodes per block,8updates,4epochs/update and
64/64/64/16 minibatches. Candidate has no extra trajectories or optimizer steps.
Clinical outcomes and unresolved terminal obligations are still reported in raw
units. Model-loss improvement is not the primary endpoint.

New namespace and ordinal roles are in proposal.json. Each training ordinal is
shared across the three arms; four episodes within one rollout have distinct
environment seeds. Test and preflight ordinals are disjoint. Pairing ensures the
same starting world, not identical subsequent RNG events after divergent actions.
Neural sampler/shuffle streams should be derived from namespace plus purpose and
block, with identical sampler starts across paired arms and separate preflight
copies; reuse the existing64-bit hashing convention. Analysis uses its own
purpose-derived seed. Freeze the actual decimal manifest and scoped local
collision check before execution; no silent seed replacement on a collision.

## Whole-attempt budget

Exact phase and owner caps are in proposal.json. Total522episodes comprise
18preflight,288training and216evaluation. Total27,216 env.step calls comprise
27,144trajectory and72mandatory restore-clone calls. Max1,920optimizer calls:
1,152actor and768critic. Three layouts plus522episodes give525environment builds.
Exactly six historical model loads; no repeated numerical qualification.
All12learned models are sealed before the216evaluation episodes.

At preflight step26, each of18controllers restores a disposable midpoint clone
and executes4comparison steps. Reuse original restore/identity contracts. Read
18midpoint envelopes and216sealed test envelopes,234scheduled reads total.
For the12learned forks, check same-start greedy probabilities and requests on
the52own-frozen contexts per block without extra environment calls. These
forwards, collection/update forwards, restoration and all I/O are inside the
listed phase clocks, not permission for another scoring campaign.

Global elapsed cap is10,800seconds (3hours) from outer startup; sum of phase caps
is10,440seconds. No phase may borrow another's unused time or calls. Include
imports, binding, preflight, training, evaluation, raw verification and both
local payload/terminal archives. Serial execution, one exclusive result root,
durable pre-call nonrefundable debits, external supervisor and reserved closure
time. Timeout/native error/terminal scientific failure consumes the attempt:
save partial evidence and stop, no repair-and-retry. Keep old attempts intact.

## Readout and switch rule

Primary: paired raw cost time-baseline PPO minus own frozen; report the
intervention-specific contrast versus current PPO and attribution versus BC.
Retain all worlds and block effects. Descriptive hierarchical paired bootstrap
uses3training blocks and12worlds/block,10,000draws; this is low-precision
development evidence, not confirmatory inference. Seed fixed before outcomes.
Do not pool with historical test results or access a formal holdout.

Reuse the prior development promotion screen: >=1% equal-block cost reduction
versus frozen and favorable direction in all3blocks. Require comparisons against
current PPO and BC to support their respective claims; do not label an inherited
gain as RL benefit. Any adverse completion/loss/waiting/terminal-active direction
is a trade-off, not clinical superiority. Report smaller/inconsistent positive
effects honestly even when the screen is not met. No unapproved margin is invented.

If the screen is not met, greedy actions remain unchanged, or improvement is only
in baseline/advantage diagnostics, close this baseline route after this attempt.
Do not add epochs, seeds, critics, alternative baselines or repeat toy gates.
Move immediately to the prepared reward-design memo, using existing records and
source inspection. This switch authorizes diagnosis/design, not unspecified
reward coefficients, new objective training or an automatic reward search.
If a terminal engineering failure prevents the comparison, preserve it and state
that the scientific hypothesis is unresolved; no further baseline trial by default.

## Delivery boundary

Implement target selection, serial integration, state restoration and independent
raw readout with artificial/mock zero-update fixtures; reuse unaffected checks.
No fit-only prerequisite. Before a real attempt, bind exact source/runtime/input
and stream locks, record the numeric approval, append locked-plan change control
and commit locally. Routine execution then needs no per-phase approval. No remote,
Dropbox, messages, paid compute, Stage E reopening or collaborator approval claims.

The primary methodological distinction follows the policy-gradient baseline
identity: an action-independent baseline changes the estimator, not the task's
reward objective. See [OpenAI policy-gradient documentation](https://spinningup.openai.com/en/latest/spinningup/rl_intro3.html#baselines-in-policy-gradients).
