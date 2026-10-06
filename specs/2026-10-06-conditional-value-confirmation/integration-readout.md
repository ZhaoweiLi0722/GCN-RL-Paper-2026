# Conditional Value-MPC Integration Readout

Checkpoint: 2026-10-06T09:51:39Z. Entry HEAD
9496f82c7da745378d4b2fb91f1b4def50d03f52 on
codex/september-research-integration. Zhaowei requested `全面的推进`.
This checkpoint completes implementation and necessary zero-update validation;
it is neither a scientific result nor approval to spend the proposed budget.

## Research Decision

Proceed with one prospective GCN TD value-guided MPC comparison, not a return
to DDPG or an open algorithm search. The prior native-return attempt remains
closed with its failed persistent-condition primary screen. Its secondary fast
condition motivated the new prospective focus; all three conditions remain in
the new test. Existing evidence does not establish universal RL benefit or an
isolated GCN contribution for this new method.

Five fresh independent training blocks replace historical initialization as the
main inference unit. Seven evaluation roles cover plain H8/H16, a newly trained
frozen initializer, a historical evaluation-only reference, graph TD, self-only
TD and graph MC. Graph/self-only have identical trainable parameter counts,
layers and paired initial tensors, differing in adjacency. They share training
records and sampler indices. TD/MC share native branches; MC is itself policy
evaluation, not a generic non-RL baseline.

The 48 shared plain-MPC warmup worlds per block standardize data for attribution.
They change the old 24-plain plus 24-learner-continuation recipe. Therefore this
is a newly standardized method, not an exact replication of the old treatment.
Self-only initialization ancestry stays separate from the common graph parent
that generates continuation data. No graph buffer is copied into self-only.
No reward, physical condition, candidate support or old result is changed.

## Delivered Implementation

- Config and full numerical protocol: capacity_confirmation_20261006.json and
  protocol.md in this package. Frozen authority files are intentionally absent
  until an exact new numerical approval arrives.
- Matched model and strict warm/native learners in capacity_confirmation_value
  and capacity_confirmation_learner; float64 loss with float32 parameters,
  strict checkpoint/optimizer validation, separate ancestry and continuation
  hashes, fresh Adam/RNG at native-tail forks.
- Additive design, resource ledger, serial runner and exclusive execution entry.
  Reuse the existing native collector, simulator, filter, planner, raw recorder,
  archive implementation and five-block summary functions without editing them.
- Saved-data reader covers all 780 main trajectories and 240 dependent branches,
  21 pairs across three conditions and five blocks, cost/patient/labor/action
  reconciliation, model binding, seal barrier, fit receipts and matched source
  hashes/sampling indices. Matching stored hashes is not a claim that every
  feature tensor is independently regenerated from raw receipts.
- Explicit fast-minus-stable and fast-minus-persistent interactions resample
  training blocks jointly but separate condition worlds independently. The
  prespecified screen and bootstrap intervals are descriptive development
  evidence, not a powered or multiplicity-adjusted confirmatory trial.

## Verification Completed

Coordinator command, using the repository virtual environment:

```text
python -m unittest tests.test_capacity_confirmation_runner tests.test_capacity_confirmation_learner -v
26 tests passed in 20.769s
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python -m compileall -q .
exit 0
```

The actual runner entry and saved reader execute the complete 780-main/
240-branch schedule with artificial backends. No real simulator, historical
scientific checkpoint inference or optimizer update is performed. Small
artificial neural tensors are used in unit tests; they are not experiment
trajectories or training evidence. Tests cover all seals before evaluation,
paired initialization, identity adjacency persistence, model/Adam corruption,
TD/MC formulas and common data, nonrefundable owners, checkpoint barriers and
sampling tampering. The initial interaction assertion was changed from exact
binary floating-point equality to numerical tolerance; no scientific algorithm
or outcome was changed. System Python lacked torch; final commands use the
existing project virtual environment with no installation.

Popper delivered only the matched model, learner and learner tests; reported
63 passing tests including reused related tests. Avicenna gave one read-only
critical-path review. Both finite agents are completed and closed. Their test
counts overlap the coordinator tests and must not be added as unique tests.

Fixed arithmetic: 240 warmup +120 reference +420 evaluation =780 complete
trajectories, plus240 dependent clones; 49,920 main +9,720 branch native steps;
61,200 operations plus240 clones; 26,880 value updates,0 actor; 46,220 forwards
including840 training bootstraps; 17,057,280 prediction steps; 5,964,000 filter
transitions. Ten warm and fifteen final seals precede tests. Phase sums reconcile.
All these are proposed caps or mock ledger checks, not observed scientific work.

## Remaining Boundary And Next Action

One consolidated question has been sent for the entire 24h/12GiB single attempt,
including all phase/owner caps. At this checkpoint no response to that question
has arrived. Preparation is ready; freeze, scientific launch and results do not
exist. The old attempt's authority is consumed and cannot authorize this run.

After approval: bind the literal response and unchanged protocol/config, commit
implementation and authority, make the mandatory current source/input/runtime/
stream collision locks, commit the packet, and launch once without another
routine startup question. The first counted world is the scientific preflight.
Necessary validation is complete; do not add a new toy-study or audit gate.

Once running, preserve all results, terminate on the frozen failure rules, and
complete raw readout, truthful manuscript update, local commit and the existing
additive Dropbox handoff. Do not retry, extend budgets or adjust rewards on a
negative result. No new push, merge, public sharing or collaborator message is
implied. The same monitor's last verified state is PAUSED; it was not modified
or treated as evidence of training during preparation. New-run export is due
only at a complete or terminal failed boundary. Prior local Dropbox copies do
not establish cloud synchronization or collaborator access.
