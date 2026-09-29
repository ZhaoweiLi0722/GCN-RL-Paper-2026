# R4 preflight record

Source before smoke: `14fc8aea702fc6724ac8b729927c5b06b7ca2908`.
62 focused tests and full compileall passed. No production rebuild started.

One MPS smoke performed one MDL-2 demonstration transition, one imitation
epoch, one cached-teacher distillation epoch and one offline update, zero online
episodes. Outputs remain in `results/frozen_baseline_rebuild_smoke_20260929`.
The outer smoke verifier exited 1 after training with:

```text
KeyError: 'local_search_demonstration_source'
```

The existing benchmark wrapper renames that key to
`advantage_distillation_demonstration_source`. The new R4 verifier had repeated
the incorrect lower-level prefix. Correct the verifier and add a regression
test before the production attempt. Do not rerun this smoke's training;
validate the existing saved config/summary/checkpoints and archive them.

The smoke stdout's `actor_loss=nan` is the existing logger's placeholder when
the first update has no actor step (critic warmup 500, actor frequency 2), not
a persisted loss.
The persisted summary is finite and reports the critic update. Keep the log
unchanged and explicitly distinguish this placeholder from numeric divergence.
The production 500-update summaries must still pass finite checks. No legacy
logger or scientific code was changed to suppress the message.

Verifier fix committed as `1845339`. Sixteen new/64 related tests and full
compileall pass. Read-only verification of the existing smoke artifacts then
passed, including exact actor/gate equality with the full state. No training
was repeated. The seven-file smoke archive SHA256 is
`3bcc715070fbc2ec2e2787d2874c59f3712a8b6d127fccd45ab311a705988729`.

The production rebuild subsequently completed all three seeds. All 500
offline updates remain within critic warmup, so production logs also contain
the same absent-actor-loss placeholder. Persisted
`offline_rl_actor_updated_mean` and `offline_rl_actor_updated_final` are both
zero for all seeds; saved metrics/model tensors are finite. This is the locked
pretraining recipe, not an unexpected online actor failure. Actual actor
learning occurred in imitation and advantage distillation.
