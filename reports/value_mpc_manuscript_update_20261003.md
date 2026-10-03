# Prospective Value-MPC Manuscript Update

2026-10-03. Manuscript-only change on `codex/september-research-integration`.

## Changes And Evidence

- Active sources: `paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks/main.tex`
  and its `references.bib`. `make_diff.sh` identifies `main.tex` as the editable
  manuscript; `main_diff*.tex` are generated historical review files.
- Added a visibly prospective methods subsection and discussion paragraph for
  TD-learned GCN long-term value guiding the existing public-input MPC. The
  coordinator subsequently added the eight-step residual TD target, 64-step
  settlement horizon and proposed 288-world/144-evaluation comparison. The
  numerical execution approval and experiment remain pending. This is neither a DDPG action head nor
  an implementation of full TD-MPC. No new gain is claimed.
- Added Hansen, Su, and Wang (2022), checked against the
  [PMLR primary source](https://proceedings.mlr.press/v162/hansen22a.html), as
  conceptual inspiration only.
- Added the completed fixed-budget development result from
  [terminal readout](../specs/2026-10-03-fixed-budget-allocation/terminal-readout.md)
  and [independent saved-data readout](fixed_budget_capacity_20261003_independent_readout.md):
  three training blocks, 36 reference/36 training worlds, 108 frozen evaluations,
  all six condition/comparator means, primary uncertainty, and all three failed
  benefit criteria. No new calculations or scientific executions were needed.
- Retained the historical manuscript and all evidence, including the
  [frozen evidence map](../experiments/evidence/patient_indexed_specimen_routing_publication_evidence_map.json).
  The later [manuscript evidence checkpoint](../docs/team_updates/2026-09-29-manuscript-evidence-checkpoint.md)
  qualifies its graph-attribution language as a complete-controller contrast,
  not isolated message passing. The new prose preserves that distinction.

## Validation And Boundaries

- Validation passed: `git diff --check`; balanced LaTeX environments and braces;
  49 unique labels with no unresolved cross-references; 41 cited keys resolving
  to 58 unique bibliography entries; all six comparison rows and both primary
  intervals matching the saved readouts; and all local evidence links resolving.
  A line-order check against `HEAD` confirmed every pre-existing manuscript and
  bibliography line is preserved. These are static checks, not typesetting.
- No TeX engine (`pdflatex`, `xelatex`, `lualatex`, `tectonic`) or `latexmk` is
  available on PATH. `biber` alone cannot typeset this multi-file manuscript.
  PDF compilation and visual layout verification remain unperformed; the
  existing `main.pdf` is unchanged and does not contain this update.
- `latexdiff` is unavailable, so generated review sources remain unchanged.
  The manuscript-only worker did not edit code, tests, specifications, workflow,
  historical results or the evidence map. The coordinator separately completed
  the protocol, additive implementation, zero-update tests and compileall; see
  the new integration readout. No scientific execution or push occurred.
  This note is not a numerical execution permit.
