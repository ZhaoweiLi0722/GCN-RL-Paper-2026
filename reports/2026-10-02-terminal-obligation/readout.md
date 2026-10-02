# Terminal Obligation Diagnostic

## Decision

Do not add an arbitrary penalty for each unfinished patient. Prefer designing
a fixed enrollment window with explicitly defined follow-up to modeled infusion
or loss. This changes the objective horizon, not historical results. It has not
been simulated, fitted or approved as a numerical experiment.

The new saved-data finding also refines the previous report's interpretation:
more terminal-active patients are **unresolved obligations**, not proof of worse
clinical care. The prior adverse-direction flag remains a conservative reporting
flag; do not erase it, but do not interpret it as observed harm.

## What the Saved Records Establish

All 216 completed development-test controller episodes were read, covering six
controllers, three training blocks and 12 paired worlds/block. The source is the
completed time-baseline attempt, not new test data. The diagnostic verifies 432
final-state/outcome files against the existing archived manifest plus that
manifest's receipt, and checks all 433 inputs unchanged after reading. Raw patient
identities, locations and enrollment counters reconcile. No models or environment
were loaded, advanced or trained. [Diagnostic](diagnostic.json).

| Mean per episode | Frozen / PPO / time-PPO / BC / R4 | Full MDL-2 | First group minus MDL-2 |
| --- | ---: | ---: | ---: |
| Enrolled | 5,949.806 | 5,949.806 | 0 |
| Completed infusion | 2,929.194 | 2,921.694 | +7.500 |
| Modeled losses | 2,311.917 | 2,335.333 | -23.417 |
| Active at step 52 | 708.694 | 692.778 | +15.917 |
| Final-step newly enrolled waiting | 122.500 | 122.500 | 0 |
| Older active | 586.194 | 570.278 | +15.917 |
| Waiting, all ages | 569.972 | 564.250 | +5.722 |
| Manufacturing | 126.278 | 126.750 | -0.472 |
| Specimen transit | 12.444 | 1.778 | +10.667 |
| Finished-product return | 0 | 0 | 0 |

Each role's mean uses 36 worlds; these are not 216 independent trained policies.
The five first-group controllers have identical full final states in all 36
paired worlds, not merely equal active counts. Immutable enrollment attributes
also match MDL-2 in all 36 worlds; final patient outcomes differ from MDL-2.
The existing raw cost contrast remains -0.564443%, inherited from reference
behavior rather than an RL increment. No new hypothesis test is performed.

### Administrative Censoring Is Not Automatic Harm

Final-step arrivals enter after production and aging; they have zero completed
treatment opportunities at the reporting cutoff. Their 122.5 mean accounts for
17.2853% of the frozen controller's active terminal count. However, it explains
**none** of the between-controller active-count difference because both groups
have exactly the same final-step entrants. The complete age histogram is retained
in the diagnostic; between-group active differences occur at recorded ages 3-6,
not at ages 0-2 or 7. These are descriptive simulator epochs, not calibrated
clinical time units or mortality probabilities.

Patient conservation supplies the exact reconciliation:

`delta(active) = -delta(completed) - delta(lost) = -7.5 - (-23.416667) = 15.916667`.

Thus the R4-like behavior has more patients still active while also having more
infusions and fewer modeled losses. This does not tell us the eventual outcome
of the unresolved patients, establish causality for a particular route, or prove
overall clinical superiority. It does show why minimizing active counts alone
can reward premature losses. Likewise, additional active patients cannot be
valued as certain deaths. The prior block61 completion/waiting concern remains.

### Reweighting Cannot Create a Retrospective RL Gain

Original PPO, time-baseline PPO, BC and R4 each have full-state equality with
frozen in all 36 paired worlds, including patient and resource/pipeline inputs.
For any common deterministic terminal function `L` on those states,
`delta(C + L(s_52)) = 0`. This holds without choosing or sweeping a coefficient.
It is an algebraic consequence of equal costs and equal inputs, not a new fitted
result. A stochastic follow-up under the same controller and common random stream
would also begin from the same state; no such follow-up has been run here.

Reward redesign could change **future training**, but cannot relabel the current
zero RL increment as positive. This is why post-hoc reward-weight selection or
rescoring these same policies is not our next performance experiment.

## Next Integrated Design

See [prospective design](../../specs/2026-10-02-terminal-obligation/design.md)
and the independent [source contract](../../specs/2026-10-02-terminal-obligation/source-contract.md).
Keep the original 52-step cost/outcome report, and define an additional closed
cohort endpoint before any fitting. Prefer recording later primitive costs under
a common declared follow-up rule over assigning an invented per-patient penalty.
No post-infusion clinical model or monetary validation is supplied by this change.

The next scientific package should compare old-window versus cohort-objective
PPO from the same initializer, with frozen and BC controls and R4/MDL-2 context.
Use matched trajectory acquisition including follow-up for both reward arms;
only the training target's inclusion of post-window cost differs. This is a
proposal, not a launch permit or a claim that either objective will improve RL.
Do not add another critic-fit or baseline-selection gate. Fix numeric caps,
fresh streams, cost settlement and stopping rules in one complete approval packet.

## Reproducibility

- Parser/config frozen in local commit `5260cff` before the saved-data read.
- Scope: `specs/2026-10-02-terminal-obligation/analysis-scope.json`.
- Nine invented-JSON tests passed; full repository `compileall` passed.
- Diagnostic SHA256: `4d97cca522ba34bfc7308e25a66f9a7123fc7d0b1ea7abf1efe5d83e3933216b`.
- Input hashes, per-world cohort/age/risk counts and contrasts are in the JSON.
- No historical result was rewritten; all data remain previously observed
  development evidence, not fresh confirmation or a new reward-training set.

Command, run from the worktree with the project Python and `PYTHONPATH=.`:

```bash
python reports/2026-10-02-terminal-obligation/analyze.py --config specs/2026-10-02-terminal-obligation/analysis-scope.json --output reports/2026-10-02-terminal-obligation/diagnostic.json
```

The writer refuses an existing output. Reproduction must use a new output path.
Process evidence: the prior runner/child PIDs were absent in the read-only host
check; this diagnostic has completed and no new experiment is running.
