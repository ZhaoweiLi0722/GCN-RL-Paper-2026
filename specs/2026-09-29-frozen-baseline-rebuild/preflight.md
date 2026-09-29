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
the first update has no actor step (actor frequency 2), not a persisted loss.
The persisted summary is finite and reports the critic update. Keep the log
unchanged and explicitly distinguish this placeholder from numeric divergence.
The production 500-update summaries must still pass finite checks. No legacy
logger or scientific code was changed to suppress the message.
