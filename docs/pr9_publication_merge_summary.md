# PR #9 Publication Merge Summary

## Final result

No policy was trained in this follow-up line.

| Study | Final classification | Publication status |
| --- | --- | --- |
| Patient-indexed routing labels | Original Stage G1 negative stands; later budget study did not reproduce its baseline | Closed result plus an uninformative follow-up |
| Continuous overtime | Held-out state-dependence +0.4500% below the 0.5% gate; fitted top-1 0.267 below 0.50 and worse than state-blind 0.311 | Channel closed; no actor authorized |
| Stochastic procurement lead time | Reported variability cost +0.176%; post-hoc reproduction +0.1169%, but unsigned draft required a different optimality-gap benchmark | Exploratory diagnostic with reproduced direction and scale; no formal gate decision |

## What the evidence supports

- The current action space and strong MDL-2/AFD initialization leave little
  independently attributable room for online actor-critic updates.
- The overtime extension showed real exploratory state dependence, but it did
  not generalize to the reserved scenario and was mostly captured by a tuned
  constant policy.
- Changing the encoder or the small supervised ranking model did not rescue
  overtime learnability.
- The lead-time diagnostic suggests that a lead-aware heuristic absorbs much
  of this single-resource uncertainty, but the drafted optimality-gap question
  was not tested.

## What the evidence does not support

- A claim that online DDPG updates created the observed Stage E gain.
- A claim that stochastic procurement formally failed a preregistered gate.
- A claim that a new algorithm alone can overcome the existing action-space
  geometry and missing headroom.

## Required before merge

- [ ] Replace the stale PR body, which currently headlines the exploratory
      E2b pass, with this final-stage summary.
- [x] Add an executable config, 72 paired rows, summary hashes, and artifact
      inventory for a post-hoc reproduction of the variability estimand; keep
      it explicitly exploratory because the original benchmark was not run.
- [x] Reconcile the lead-time protocol text with the implemented
      per-facility/per-epoch random draw and actual configuration fields.
- [x] Move post-lock change-control entries out of the historical locked plan.
- [x] Run `compileall`, all PR-specific tests, and the full suite; document any
      baseline failures and require no new failure relative to `main`.
- [x] Align manuscript claims so the complete AFR controller is credited while
      online-update attribution remains explicitly unsupported.

## Merge decision

The negative outcome is not a reason to withhold the PR. Merge once the items
above make the branch internally consistent, reproducible at the strength of
each claim, and safe to cite. Any new intertemporal capacity-control scenario
must start in a separate preregistered study and must not delay this archival
merge.

## Verification record — 2026-09-01

- `python -m compileall -q .`: passed.
- PR-specific suite: 101/101 passed.
- `main.tex` compiled to an 84-page PDF with resolved citations and references;
  pages 72--76 were rendered and visually checked after the manuscript edit.
- Full suite with the repository PyTorch environment: 839 tests run. The only
  outcomes outside pass/skip were three historical locked-asset hash errors
  and one Apple-MPS-only `gcn_ppo` sanity-threshold failure.
- The three hash errors concern the locked execution plan,
  `evaluation/train_multiscenario_network_residual.py`, and
  `src/rl/training_state.py`. The PR does not change the latter two, and this
  audit restores the locked plan byte-for-byte to `origin/main`; none is a new
  PR #9 regression.
- For the PPO sanity check, clean `origin/main` and this branch produce the
  same Apple MPS result (`30.245M` learned versus `57.465M` random): the learned
  policy beats random but misses the test's stricter half-random threshold.
  Both pass on CPU. Initial parameters, the first 240 training steps, and the
  actor hash after 2,000 steps match between branches. This is recorded as
  baseline device-specific test debt; the threshold is not changed here.
