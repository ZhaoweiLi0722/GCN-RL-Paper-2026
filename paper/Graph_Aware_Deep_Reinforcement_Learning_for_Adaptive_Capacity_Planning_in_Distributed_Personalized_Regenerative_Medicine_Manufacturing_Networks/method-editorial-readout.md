# EAAI Native Value-TD Editorial Readout

Date: 2026-10-11. Scope: the user's explicit local manuscript rewrite and DDPG
archival request, followed by the correction **"是 eaai"**. This supersedes
the earlier no-manuscript-edit boundary only for this editorial deliverable.
It does not amend any scientific packet, numerical budget or process authority.

## Finding and Delivered Change

Before this edit, `main.tex` already described a Value-TD controller, but for
the separate **four-site support-hour** problem. Its title, 48+16 horizon,
residual-value formula, architecture and results did not describe the current
original twenty-site experiments. The separate highlights and title page still
described DDPG. The old `main.pdf` predated the source, so it was not evidence
of a current compiled manuscript.

The current `main.tex` now describes the native twenty-site, 52-epoch method
as the principal proposal. The method section gives the native cost objective,
public-information score, graph and flat models, normalized TD8 target,
episode-level updates, A candidate construction, B sequence CEM, pseudocode,
design rationale, computational trade-offs and limitations. It distinguishes
the MDL-2 operational, H8 learned-tail and Flat architecture contrasts. It
keeps A/B as distinct implementations without selecting a winner or treating
the broader B search as an equal-compute ablation.

The DDPG method has a dedicated archival record and a bounded historical
appendix in the main manuscript. Positive historical cost/service findings,
unestablished incremental actor-critic learning, adverse transport sensitivity,
and implementation limitations remain visible. The text does not claim that
the current TD method has beaten DDPG. The full historical research source
and `references.bib` are unchanged.

The main source is self-contained with eight numbered references so the
desktop LaTeX editor can compile it without loading external project files.
The existing Elsevier `elsarticle` review format is retained. Metadata,
highlights and availability wording now match the actual method and draft status.
The README provides a short bilingual Howard meeting explanation and entry points.

## Evidence Binding

`method-evidence.json` records the exact main-source hash, 22 local source
hashes, ten claim groups linked to section/equation labels, eight public
reference records, and byte-level preservation checks for the five replaced
legacy artifacts. These are assistant source checks, not completed human
verification. Accountable-author review remains explicitly pending.

The principal method evidence is the implemented source plus each original
frozen A/B protocol. The training status is used only to establish that
independent evaluation is incomplete, not to support efficacy. A's saved
17:05:36 UTC boundary was still training with zero evaluations. B's
`train-b0-graph` 17:06:31 UTC boundary was still training, so its all-model
barrier had not been reached. This editorial task did not perform a new
process-identity audit, compute patient effects, or represent a heartbeat as
scientific progress.

## Five Writing Passes

1. **Clutter:** Removed the old multi-campaign numerical narrative from the
   current abstract; it now states task, mechanism, comparison design and
   pending evidence. The old text is preserved, not silently overwritten.
2. **Voice and verbs:** Assigned actions explicitly to controller, planner,
   environment and learner. For example, the model predicts cost while the
   planner selects and the environment executes requests.
3. **Sentence architecture:** Separated physical task, value learning, finite
   search, sampled search, and evidence status. Equations and two comparison
   tables replace dense lists of mixed method and result claims.
4. **Terminology:** Replaced the old support-hour use of Value-TD with the
   explicitly defined Native Value-TD. Distinguished twenty physical sites
   from twenty-one graph nodes, state value from action value, planning
   horizon from TD span, and simulation training from deployment adaptation.
5. **Numbers and citations:** Matched TD parameters, network dimensions,
   candidate counts, per-world prediction steps, test counts and comparison
   denominators to source/protocol. Corrected the candidate-pattern formula
   to include its actual `1e-6` normalization floor. Retained the distinction
   between formal ratio-of-means savings and exploratory training summaries.

No new numerical results, algorithm convergence theorem, novelty priority,
clinical-safety claim, deployment guarantee or acceptance guarantee was added.
The historical DDPG interval described in the archive is explicitly a separate
post-hoc sensitivity and does not replace the original declared analysis.

## Validation and Limits

- Desktop `compile_latex_document` returned **success** for the final source.
- Static source checks passed: 24 unique labels; all internal references and
  citations resolve; eight bibliography entries; title/main metadata agree.
- The abstract has 202 whitespace-delimited words. The five highlight strings
  have 73, 77, 73, 75 and 70 characters, excluding their bullet prefixes.
- The archived old main source, title page, highlights, availability statement
  and old PDF are byte-identical to their pre-edit Git HEAD versions.
- The original historical manuscript and bibliography are unchanged. No paper
  paths occur in the inspected A, B or B-migration source/input lock lists.
- Scientific source, configuration, protocols, results and monitors were not
  modified. No environment, model forward, optimizer, scientific test, restart,
  migration, archive regeneration or remote action was executed.
- There is no terminal TeX engine on the checked PATH. Compilation used the
  built-in editor; no standalone new `main.pdf` was exported. The previous PDF
  is clearly named `legacy-main-unmatched.pdf` in the preservation directory.
- PDF page-by-page visual inspection has not been performed. Successful
  compilation does not certify page layout or final submission readiness.
- The current official EAAI guide URL returned HTTP 403 during this task.
  Existing journal format and conservative scholarly writing conventions were
  retained; latest full journal-rule compliance has not been verified.

The public primary records for TD, graph convolution, DDPG, TD-MPC, CEM and
domain context are mapped in the evidence record. Some publisher full-page
requests were unavailable; the record distinguishes opened sources from
publisher metadata/search results. Scientific Agent Skills was materially
used, so its primary arXiv metadata was checked and the paper is cited only
in the truthful writing-assistance disclosure. All affected claims,
affiliations, author declarations and final submission wording need human review.

## Finite Roles and Remaining Work

Epicurus delivered only `archive/ddpg-method-record.md`; Bohr gave one
read-only efficiency review. Both assignments finished and were closed.
The coordinator integrated the main manuscript, source mapping, preservation,
metadata and compilation. No additional review or numerical gate was added.

The usable deliverable is a Howard-readable methods draft. The remaining
scientific blocker is the already-running independent A/B evaluation and its
saved-data readout, not more manuscript tooling. A's supplemental interrupted-
evaluation replay approval remains outstanding; this editorial authorization
does not supply it. No upload, submission, collaborator message or external
copy is authorized or performed by this revision.
