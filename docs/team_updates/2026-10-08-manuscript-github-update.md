# Fixed-Support Manuscript Revision and GitHub Delivery

## Scope and authority

On2026-10-08 Zhaowei requested:continue the next step, push this stage's
results to GitHub, and revise the manuscript. This authorizes publishing the
current research branch to the existing origin repository and editorial work.
It does not authorize a new numerical experiment, changed frozen sources,
retry, new training, merging, force pushing, journal submission or another
person's approval. Entry commit:85da13daa0726bfb368bb380426a4456b74af4c9.

Destination:ZhaoweiLi0722/GCN-RL-Paper-2026,branch
codex/september-research-integration. Fresh fetch finds origin/main at eeed57b
with47local-only and0remote-only commits at entry. The research branch did
not yet exist remotely. Publish the accumulated stage record without changing
main or merging other branches. Large raw archives/checkpoints stay outside Git.

## Editorial changes

The abstract now distinguishes the formal routing study, the failed
five-family H8 screen, and the posthoc fixed-support supplement. Introduction
and contributions use the same distinctions. A new results subsection records
the exact fixed2support rule,120new/360reused trajectories and two tables:
four-role means and Value-TD's three-condition contrasts. Discussion retains
the H8 no-winner result and does not claim isolated graph or RL gains.

Value-TD saves2.1565% mean cost and5.675simulated losses/world versus fixed
MDL-2; all5block cost means favor it, but34cost-harmed and26patient-harmed
worlds remain. Fast-condition intervals cross zero. SAC's unfavorable results
and H8's cost/service tradeoff remain visible. The fixed comparator is not
represented as retrospectively prespecified or a substitute for the H8 primary.
README now points to the latest results rather than an earlier development run.

The existing scientific-writing and manuscript-writing-review guidance was
used for evidence binding, concise wording, terminology and numeric consistency.
Its previously verified software attribution remains in the2026-10-06
publication note; no new literature, author approval or declaration is invented.
No manuscript was sent to an external editing service. GitHub publication is
the external transfer explicitly requested by the user.

## Reproducible evidence and checks

reports/2026-10-08-manuscript-publication/evidence.json is a portable curated
projection of the completed comparison:all4role summaries,6pairs,conditions,
blocks,world harm,compute and interpretation flags. It binds source SHA256/bytes,
the frozen protocol and terminal receipt. No raw/world/model reconciliation was
rerun. bind_evidence.py checks all7new table rows and12overall interval endpoints,
all45citation keys,labels and balanced LaTeX environments. No missing/duplicate
labels or missing bibliography keys were found. Human author review remains
unrecorded; automated matching does not replace it.

The built-in compiler reproduced the pre-existing elsarticle/biblatex author
counter collision. The manuscript now uses the repository's native elsarticle
natbib numeric stack and elsarticle-num.bst; existing cite/citet commands are
supported and bibliography content is unchanged. A second compile passed the
former collision and reached the first figure, then failed because the built-in
standalone compiler cannot access the external project image
`figures/Figure 1.png`. The image exists locally;
it was not removed, replaced or faked. No terminal TeX engine is on PATH.

Consequently a newly built PDF, complete citation rendering and visual layout
verification remain unavailable. Existing main.pdf predates this revision and
must not be presented as its output. The source was opened in the built-in
editor. Whole-worktree compileall and git diff --check pass. Scientific checks
are reused:23family tests and12supplement tests; no new environment/model call.

## Next step delivered

One finite advancement agent delivered
reports/2026-10-08-value-td-next-step/{read_development.py,diagnostic.json,report.md}.
Only development/training outcomes enter its calculations:72evaluation worlds
per existing recipe and576fit receipt files. Patient-facing cost savings are
partly offset by reagent purchase/shortage costs; direct extra-support charges
are a small offset. Lower recorded TD loss does not imply better operation.
Support volume correlations and component decomposition are descriptive, not
causal or evidence for changing reward weights. Fast fluctuation was not the
weakest development cost condition, so a final-test subgroup cannot be promoted
to a development-selected mechanism.

One hypothesis is fixed attenuation of the learned value residual toward H8,
compared with unattenuated Value-TD and H8 on untouched future worlds. This is
not implemented or run. A coefficient, finite budget and patient guardrails
must be frozen prospectively before any future numerical execution. Existing
test and MDL-2 worlds remain consumed. No parameter grid, reward change or
automatic retraining is authorized by this editorial/publication step.

A separate read-only efficiency evaluator advised reusing existing readouts,
keeping the new manuscript subsection focused, and ending the diagnostic at
one decision-changing hypothesis. Both finite assignments completed and closed;
there is no background scientific job or new scheduler. The old gcn-rl remains
PAUSED. Git publication verification is recorded after the push; local Dropbox
handoff receipts remain distinct from cloud sync/collaborator access.
