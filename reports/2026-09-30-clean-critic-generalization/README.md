# R6 audit and preservation receipt

Single scientific execution97c79a6b316740b835d82bc3f2be2683624908f9.
Raw run completed exit0; independent raw arithmetic and saved-weight prediction
replay pass. `verification.json` preserves all six trajectory contrasts.
`fit_generalization.json` is post-run descriptive error accounting, reproduced
exactly without new simulation or fitting. Readout/protocol live under
`specs/2026-09-30-clean-critic-generalization/`.

The scientific triage failed; no actor training or retry was launched. Stderr
is empty. All initial/final critic, Adam and CPU/MPS/NumPy/Python RNG states,
frozen actor/gate payloads, effective configs, snapshots, raw paired traces,
seals and source archive are retained. MPS bitwise recovery is not promised.

## Verified preservation

Archive: `results/clean_critic_archives_20260930/r6_clean_critic_generalization.tar.gz`.
All3659 members match their per-file SHA256 values and unchanged source bytes.
Size611157514 bytes. Archive SHA256:
`bce84b741c3dc714a9439cc49f8226e8be7dee646450a4ca53ca23eaca0977f0`.
Manifest SHA256:
`7ec8e270975a55185fd88de08c3cd8f45c3ded0a7849465a212b38c068eef8cd`.

The package keeps repository-relative paths for the raw run, stdout/stderr,
verification and derived accounting, protocol/readout, and post-run helper/test.
Its embedded `source.tar.gz` is the exact experimental source97c79a6. The later
accounting helper is separately inventoried; it did not change that execution.
This external preservation receipt is not self-included in its own archive.

`dropbox_receipts.json` confirms both archive and manifest local bytes under:
`/Users/lizhaowei/Library/CloudStorage/Dropbox-GaTech/Zhaowei Li/GCN-DRL Paper 2026/Research Artifacts/clean_critic_generalization_20260930/`.
Cloud sync and Howard access are unverified. No sharing setting, message or
remote Git action. Originals and staging payload remain; no evidence deleted.
