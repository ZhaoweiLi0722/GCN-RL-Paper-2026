# R3 replacement pilot evidence

Readout and decision:
`specs/2026-09-29-replacement-fixed-window-pilot/readout.md`.

`verification.json` is the stdlib recomputation from all raw logical traces,
not the collector's own summary. It includes all12 prespecified contrasts,
paired validation cost draws, clinical mean differences, action-map flags and
per-draw deduplication counts. Cost delta<0 means improvement; conditional
intervals are descriptive, not multiplicity-adjusted efficacy conclusions.

The raw scientific run exited0 with1,152 records/24,348 steps in600.139s.
Separate mechanical acceptance used24 steps. No optimizer update or actor/critic
fitting occurred. All24 old input files and83 old source hashes remain intact.
Twelve new/81 related unit tests, full compileall and diff checks passed before
recording. Exact source commit:195d0eb3f7d095c3f53c211a42bcdbac8fd1e67a.

Raw roots remain in the persistent worktree under
`results/replacement_fixed_window_{pilot,engineering}_20260929`.
Preservation receipts are additive; no old source/result is replaced. The
conditional improvements must not be described as an online RL contribution.
