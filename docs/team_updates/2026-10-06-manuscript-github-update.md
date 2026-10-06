# Manuscript Revision And GitHub Publication

Date: 2026-10-06 UTC. Editorial entry commit: `1129a8e`.

## Authorization And Scope

Zhaowei explicitly requested that these updates be pushed to GitHub `main`,
that the manuscript be updated, and that a suitable scientific-writing skill
be found on GitHub, installed, and used. The typed `mian` is interpreted as
`main`, the verified default branch of the public repository
`ZhaoweiLi0722/GCN-RL-Paper-2026`.

The fetched `origin/main` at `393ec08` is an ancestor of the entry commit:
344 local commits and zero remote-only commits separate them. Publication
therefore includes the accumulated research record, not just this editorial
change. The intended operation is a fast-forward push, never a force push.
This authorization does not reopen an experiment or change a frozen contract.
The completed native-return attempt and PAUSED monitor retain their status.

## Writing Guidance And Software Attribution

Installed the MIT-licensed `scientific-writing` skill, version 2.3, from
[K-Dense-AI/scientific-agent-skills](https://github.com/K-Dense-AI/scientific-agent-skills/tree/92ace75ac21efe19a620434e0ca4e356081fe807/skills/scientific-writing).
Pinned commit: `92ace75ac21efe19a620434e0ca4e356081fe807`.
Installed `SKILL.md` SHA256:
`511a2096bebecd8cf0f8212b9d262d37154ae1ba566eb385d93ac968682edbd1`.

Its evidence-bound writing, section-consistency, and table-reporting guidance
was read and applied alongside the existing `manuscript-writing-review` skill.
No third-party script was executed on the manuscript or data, and no manuscript
was uploaded to an editing service. The skill's optional command-line tools
were not needed; existing saved-data provenance was reused rather than
generating new registries falsely marked as human-verified.

Software reference: Kassis, T., Agarwal, V., He, Y., Patel, D., and Brueckner,
A. M. (2026), *Scientific Agent Skills: A Library of Procedural Knowledge for
Research Agents*, [arXiv:2609.00065](https://doi.org/10.48550/arXiv.2609.00065).
The title, authors, year, and current v2 record were checked on arXiv on this
date. This is attribution for editorial guidance, not scientific evidence
about the manufacturing controller or journal endorsement.

The [official Elsevier EAAI page](https://shop.elsevier.com/journals/engineering-applications-of-artificial-intelligence/0952-1976)
was checked on this date. It calls for a clear AI contribution and engineering
application in the abstract, defined acronyms, and single-column formatting.
It also emphasizes real-world application and validation using public datasets.
The complete ScienceDirect Guide for Authors returned HTTP 403. This revision
therefore does not certify complete current journal compliance or acceptance.
The uncalibrated synthetic setting remains an evidence and journal-fit limitation.

## Five-Pass Editorial Review

Scope: abstract, research framing, value-MPC methods, the latest results,
conclusions, and repository entry points. This is not a new literature review
or a certification of every historical sentence in the draft.

1. **Clutter and structure.** The abstract previously accumulated successive
   experimental stages, including the 13.75% initial-model comparison. The
   revised abstract is approximately 235 whitespace-delimited words and leads
   with the application, controller, formal result, attribution limit, and
   latest development result. Earlier findings remain in the results sections.
2. **Voice and accuracy.** Replaced the statement that the method learns the
   cost function and expectation operator with the implemented claim: it learns
   policy corrections and value estimates while dynamics and objective weights
   remain specified. Corrected "retains ... and add" to "retains ... and adds."
3. **Sentence architecture.** Replaced the dense five-question paragraph with
   three questions addressed by completed evidence. Explicitly retained
   patient/expiry component attribution and temporal-encoder comparison as
   unresolved work, not newly completed experiments.
4. **Terminology.** Separated formal AFR-GCN-DDPG from the development-only
   native-return GCN-value-MPC candidate. Added the actual candidate-score
   equation and six-controller table. Plain MPC retains its terminal heuristic;
   value learning is not actor training; MC policy evaluation is not labeled
   a non-RL control. README now uses these same distinctions.
5. **Numbers and interpretation.** Expanded the persistent-only table to all
   three conditions and five comparators, preserving primary/secondary status,
   absolute cost intervals, paired-world percentages, and patient losses.
   Every displayed number matches the saved five-block readout. Conclusions
   retain the failed primary criterion, forecast-TD patient trade-offs, favorable
   secondary findings, and non-equivalence caveat for H16.

The abstract expresses the existing formal cost-change intervals as positive
savings intervals, reversing both signs and endpoint order. It does not change
their values, analysis population, or inference method. No result was pooled
across formal and development studies.

## Evidence Bindings

- Formal result: `docs/patient_indexed_specimen_routing_stage_e_evidence_synthesis.md`
  and `experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/`.
- Native-return result and design: `specs/2026-10-05-native-return-value/`
  `protocol.md`, `terminal-readout.md`, and `terminal-saved-data.json`.
- Candidate scoring: `src/baselines/capacity_planner_tail_mpc.py`, `act`.
- Native target semantics: `src/rl/capacity_native_tail_targets.py` and its
  existing planner-tail target helpers. No model was loaded for this revision.

## Validation And Publication Boundary

- All 15 new table rows and 75 displayed numbers reconcile with
  `terminal-saved-data.json`, including sign, rounding, and denominator.
- LaTeX static checks find no missing or duplicate labels, missing bibliography
  keys among 41 cited keys, or mismatched/unclosed environments.
- `git diff --check` passes. No scientific source, config, lock, seed, raw
  result, checkpoint, or archive changed. The 43 previously passed necessary
  zero-update tests remain applicable; no training or evaluation was rerun.
- Whole-worktree `python -m compileall -q .` passes with the existing project
  Python and cache output redirected to `/private/tmp`; it imports no model
  and executes no experiment.
- A publication-specific scan checked 2,659 incoming Git blobs totaling
  288,832,723 bytes for common credential/private-key signatures and sensitive
  filenames, with no matches. The largest incoming blob is 26,713,832 bytes.
  A separate in-memory scan covered 442 compressed blobs, 456 members, and
  223,955,436 decompressed bytes without matching the checked signatures.
  These bounded checks are not proof that every possible secret pattern is absent.
- No TeX engine (`pdflatex`, `xelatex`, `lualatex`, `latexmk`, or `tectonic`)
  is installed on the current PATH. A newly rendered PDF and visual layout
  verification are therefore not claimed.

Large raw archives and checkpoint tensors remain outside this Git publication;
the existing preservation receipts and manifests are retained. This revision
does not rearchive historical experiments or turn a Dropbox local-copy receipt
into cloud-sync or collaborator-access evidence.

## Remaining Submission Work

The manuscript remains a draft. Authors must review the final scientific claims,
authorship and conflict declarations, and the required disclosure of AI-assisted
editing (OpenAI Codex and the recorded writing guidance). No author approval,
clinical calibration, publication acceptance, or journal submission is implied.
The next editorial validation is compilation and visual inspection in a TeX
environment, followed by the full current journal checklist when accessible.
