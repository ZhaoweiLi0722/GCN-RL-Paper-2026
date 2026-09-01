# Change Control

## 2026-09-01: add episode-level J2a pre-screen

Recorded after the engineering smoke and before any J2 discovery or validation
seed was consumed.

The smoke existed only to exercise JSON parsing, full-scale patient dynamics,
graph observations, budget enforcement, delayed activation, and RNG alignment.
It used seed `99299000`, ran 12 steps, and made no gate decision. It suggested
that the mechanics were worth screening, but no environment weight, schedule
amplitude, commitment parameter, or scientific threshold was changed after
observing it.

J2a was added to avoid spending fixed-state counterfactual compute on a channel
that cannot beat a tuned static allocation at the episode level. Its fresh
seed families, thresholds, schedule variants, and stop rule are frozen before
execution. Passing J2a is necessary but not sufficient for J2.

## 2026-09-01: freeze fixed-state implementation and execute J2

Before any fixed-state scientific seed was consumed, implementation review
made three clarifications without changing the frozen scientific threshold:

- the complete generated seed ranges, rather than only their starting values,
  must be mutually disjoint and must avoid all forbidden seeds;
- a material state means a clinically noninferior saving of at least `0.005`
  of that state's comparator cost, rather than an unscaled dollar cutoff;
- distinct actions count realized shared-budget allocation vectors rounded to
  `0.001`, rather than merely counting policy parameter labels.

The fixed-state config then ran without amendment. It failed J2. No threshold,
environment weight, schedule, comparator, or candidate was changed after J2
began. The only subsequent computation was a labeled post-hoc decomposition of
the already persisted rows. J3, J4, policy training, and formal confirmation
remain prohibited for this configuration.
