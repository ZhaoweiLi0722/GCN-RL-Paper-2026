# S3 independent action-ranking readout

## Decision

S3 completed successfully, but **did not establish a stable improvement over
the booked-backlog rule or a reason to launch online DDPG**. Increasing forecast
accuracy changes the old low-budget requests. In three of four unique contexts,
the frozen selection is exactly the simple rule. In the remaining context its
advantage over that rule changes sign across independent validation blocks.

This is a conditional-model diagnostic, not actual-world policy evaluation.
It neither proves that the old pretrained policy is globally near-optimal nor
that online learning cannot help in another appropriately specified problem.

## Execution and audit

- Protocol/config commit: `b91dd3b`; execution source commit:
  `a6d450906f57d1fac1718562ba99145b175e8af7`.
- One recorded run, 2026-09-29, PID 23324, exit 0, 8.998 seconds including its
  bounded audit. All 16 context/block records and 256 raw chunks completed.
  The process returned normally; subsequent process inspection found no
  related evaluator. No retry, sample expansion or protocol change occurred.
- Six archived S2 aliases map to four unique public contexts. Persistent and
  fast-independent family aliases share identical public contexts at each
  capacity. They were computed once, never counted as independent replicates.
- 491,520 forecast paths, 10,409,928 model transition queries, zero new actual
  environment episodes, zero optimizer updates. These are not patient outcomes.
- Independent read-only audit passed: all raw-array hashes, 48 block/budget
  records, 5,040 paired contrasts and all 4 source + 40 historical input locks.
- Scalar replay of 960 prespecified action/path combinations agrees with raw
  batched costs over 20,085 transitions. Full settlement and prepaid tails are
  retained. The audit did not rerun the whole matrix.
- 17 new tests, 140 combined relevant tests, full compileall and diff check pass.
  No runtime traceback/nonfinite costs/error artifact was observed.

## Prespecified results

Actions are `(site0 hours, site1 hours)`. Validation A/B/C are independent
forecast integration blocks, each with 2,048 paired samples per action. They
are not independent operating worlds. Negative cost difference favors the
action frozen using only the selection block.

| Public context aliases; capacity | Frozen action | Selection / A / B / C argmin at 2048 | Frozen minus booked rule in A / B / C | Decision |
| --- | --- | --- | --- | --- |
| Unchanged; 1 slot | (0, .5) | (0, .5) in all four blocks | 0 / 0 / 0 | Ranking agrees; exactly the rule |
| Unchanged; 4 slots | (0, .5) | (0, .5) in all four blocks | 0 / 0 / 0 | Ranking agrees; exactly the rule |
| Persistent + fast-iid aliases; 1 slot | (.25, .5) | (.25, .5) / (.25, .5) / (.5, .5) / (.25, .5) | -.015564 / +.164734 / -.034485 | Ranking and improvement direction unresolved |
| Persistent + fast-iid aliases; 4 slots | (.5, .5) | (.5, .5) / (.5, .5) / (.25, .5) / (.5, .5) | 0 / 0 / 0 | Ranking unresolved; frozen action equals rule |

For the 1-slot changed-context comparison, paired Monte Carlo standard errors
are .044564 / .058224 / .048853 respectively. These concern integration under
the stipulated model, not uncertainty about model validity, patient outcomes
or operational benefit. No hypothesis test or clinical margin is claimed.

All 12 validation context/block combinations give lower conditional mean cost
for the frozen S3 selection than the archived S2 depth1/16 and depth1/64
requests. This is not a 12-trial population result. The references coincide in
the unchanged contexts, and the two changed-family aliases are duplicates.

| Context; capacity | Frozen minus S2 depth1/16: A / B / C | Frozen minus S2 depth1/64: A / B / C |
| --- | --- | --- |
| Unchanged; 1 | -.310181 / -.230652 / -.218750 | -.310181 / -.230652 / -.218750 |
| Unchanged; 4 | -.233765 / -.299561 / -.319946 | -.233765 / -.299561 / -.319946 |
| Changed public context; 1 | -.520935 / -.483154 / -.499695 | -.335693 / -.294800 / -.295654 |
| Changed public context; 4 | -.644409 / -.506104 / -.639709 | -.349182 / -.253601 / -.313599 |

The frozen selection's excess over each validation block's own sampled minimum
is zero except block B in the changed 1-slot context (+.164734) and changed
4-slot context (+.022888). These are **in-sample descriptive gaps**, not unbiased
regret or bounds on the optimal policy.

Only six of 16 context/block records retain the same argmin across all three
nested budgets. Every 128/512/2048 action mean, tie set and all 105 action-pair
differences are preserved in `blocks.json`; the larger budget does not erase
the lower-budget results or trigger another sampling round.

## Interpretation and manuscript boundary

1. The S2 low-budget action choices are not reliable evidence of useful control
   headroom. The independent larger-budget calculation improves their
   conditional score, mostly by returning to the existing simple rule.
2. Two contexts still have competing actions at the locked maximum budget.
   We stop as planned, rather than extending samples to force a winner.
3. S3 optimizes one request followed by the same restricted feedback fallback,
   at a constant archived posterior-mean response. It does not test updated
   beliefs, a stronger receding-horizon controller or actual distribution shift.
4. Even a confirmed feedback-policy gain would not establish that online neural
   weight updates are needed. A fixed history-to-action policy can use feedback.
   A later online attribution study still needs a matched frozen policy and
   non-neural adaptive comparators with the same information.
5. Keep the existing paper's package-level graph/distillation evidence distinct
   from online-update attribution and the previously documented encoder-only
   parity limitations. S3 adds a transparent diagnostic, not a new positive
   performance claim or evidence that the whole clinical task is solved.

Suggested limited wording for a diagnostic appendix:

> In four unique archived synthetic decision contexts, independent model-based
> resampling improved the predicted cost of previously selected low-budget
> requests, but did not establish a stable improvement over the public feedback
> rule. Three frozen selections coincided with that rule; the fourth changed
> improvement direction across validation blocks. These conditional forecasts
> do not demonstrate an incremental benefit from online policy updates.

No manuscript performance table or formal Stage E evidence was changed.

## Evidence and reproduction

Root: `reports/2026-09-29-completion-action-ranking/run/`.
`contexts.json` contains public states and alias mapping; `actions.json` the
grid; `selection/` the four frozen choices; `chunks.jsonl` the raw-cost and
uniform hashes; `raw/` every per-path cost; `events.jsonl` the selection-before-
validation sequence; `blocks.json`, `summary.json`, `audit.json`, `claim.json`,
`status.json` and `inventory.json` the results and provenance.

| Artifact | SHA256 |
| --- | --- |
| summary.json | `252a032c7b82c8575dfe06e0c26c9c93669fd3d140d8f360466e5cf8ba6a2815` |
| blocks.json | `b8279d95d0ccc7a401b848b6e2fdc6193e5788210d56124fc95e5655abba5aff` |
| chunks.jsonl | `9a30280b7158074255f91045159c40ac571fbc9a4b8749e2782416110dafb571` |
| audit.json | `81729918dbf37c60f523cc7ee314309b9ed17813ce03e670819df3447ad4b3d1` |
| inventory.json | `debb344e650cb4fea0ba31181c2f0a67d1547b88681ba5133e6bca6caae46377` |

Run only the `--audit-only` command in `engineering.md` for verification. The
recorded output directory is immutable. Further study requires a new approved
protocol; see `decision.md`.
