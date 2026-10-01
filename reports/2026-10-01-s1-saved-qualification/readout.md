# Saved-Model Qualification Readout

## Decision

The separately approved saved-model-only qualification completed once with
exit code 0. All three initialized policies pass both saved-path qualification
checks. This closes the previously missing frozen-scoring qualification result;
it supports **initialization fidelity to R4**, not an RL performance benefit.
The original failed S1 attempt remains closed. This result grants no authority
to resume S1, fit models, collect trajectories, open final tests, or continue
PPO/BC training.

Sources: [qualification.json](../../results/dynamic_candidate_saved_qualification_20261001/qualification.json)
and [terminal.json](../../results/dynamic_candidate_saved_qualification_20261001/terminal.json).

## Scope And Execution

Execution commit: `e6eec08dcd362f3084d15f1eb10a8ca66a9cfaca`.
Frozen packet: `0f46307d6ecca4f3b317fe8a18b88c6291af5b6b866c86c0764be209475e2afb`.
The [parent claim](../../results/dynamic_candidate_saved_qualification_20261001/claim.json)
records PID 48610, PPID 30450; the
[child claim](../../results/dynamic_candidate_saved_qualification_20261001/child-claim.json)
records PID 48618, PPID 48610, matching the terminal supervisor receipt.
These are historical process receipts, not a claim that either process is live.

| Quantity | Recorded result |
| --- | ---: |
| Completed qualification attempt | 1 |
| Checkpoint loads | 4 |
| Saved initialized policies | 3 |
| Saved state-path scorings | 624 |
| New environment calls / rollouts | 0 / 0 |
| Optimizer updates / final-test episodes | 0 / 0 |
| Total elapsed seconds | 8.584168000030331 |
| Child supervisor elapsed seconds | 8.489900000044145 |
| Authorized total time cap, seconds | 900 |

The child received the remaining watchdog allowance of
899.9103270000778 seconds. Termination was `child_exited`, not timeout or forced
kill; the terminal receipt records no error and no automatic retry.

## Six Saved-Path Verdicts

Each path contains 104 scored records, all multiclass. Both overall and
multiclass greedy action-class agreement with the R4 reference are 100%.
The corresponding path files exactly match their entries in `qualification.json`.

| Block | R4 path | Initializer-greedy path | Verdict |
| --- | --- | --- | --- |
| 60 | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block60-r4.json) | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block60-initializer_greedy.json) | Pass |
| 61 | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block61-r4.json) | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block61-initializer_greedy.json) | Pass |
| 62 | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block62-r4.json) | [104/104](../../results/dynamic_candidate_saved_qualification_20261001/paths/block62-initializer_greedy.json) | Pass |

The 624 figure counts state-path scoring operations, not 624 independent
experimental replicates. Matching greedy classes does not establish identical
policy probabilities or generalization beyond these saved paths.

## Independent Receipt Reconciliation

Read-only JSON analysis independently tallied all 628 records in
[debits.jsonl](../../results/dynamic_candidate_saved_qualification_20261001/debits.jsonl):
four checkpoint-load debits followed by 624 scoring debits. Sequences 0-627,
operation counters, predecessor links, and every canonical JSON SHA256 matched.
Monotonic timestamps were ordered and inside the authorized interval; the last
debit occurred 8.27276099997107 seconds after the parent claim. The verified
chain tail matches the qualification result:
`1e6c240e5a32782c08f272b7e818d2cda360d91c0a7bb6827acae5edcc3fc632`.
The generic supervisor's zero ledger sequence is not this scoring ledger;
the counts above come from the separate saved-qualification debit chain.

The qualification result's four outcome fields (cost, losses, completions,
terminal active patients) exactly match the previously verified
[saved readout](../2026-10-01-s1-saved-readback/readout.json): initializer minus
R4 is zero in each of the six paired worlds, and every block mean is zero.
That earlier readout also records zero differences in waiting-patient steps
and expiry losses. These are reused outcomes, not new simulations or new
patient-performance estimates. The old 39-episode arithmetic and historical
file hashes were not recomputed for this readout.

The 42 recorded before/after input entries agree in the qualification receipt;
this review compared those receipts without rehashing the historical payload.
No inconsistency was found in the debit chain, six path verdicts, process
lineage receipts, or reconciled paired outcome claims. No research checkpoint
was loaded or scored, and no simulation, optimizer call, or test was run during
this independent readout.

## Interpretation And Next Boundary

The saved initializers reproduce the intended R4 starting behavior on both
qualification paths, with unchanged saved outcomes. This removes an
initialization-qualification blocker; it does not answer whether PPO improves
on frozen control or continued imitation, demonstrate online adaptation,
establish clinical noninferiority, or justify a reward change. The result
explicitly records `rl_performance_claim=false`,
`clinical_noninferiority_claim=false`, and
`automatic_training_authorized=false`.

A future same-start frozen/PPO/BC-CONTINUE comparison remains a separate,
prospectively bounded decision. Passing this qualification neither reopens the
consumed S1 attempt nor makes its unused budget available for continuation.
