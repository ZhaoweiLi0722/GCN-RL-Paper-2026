# Completed Five-Block Value-MPC Comparison

Readout: 2026-10-04 UTC. Status: completed, not a new scientific attempt.
Supervisor and child exited with code0 at22:30:27Z; terminal elapsed16943.539s
within29855s. No related experiment process remained at the terminal check.
All failure/overrun/preservation-error markers are absent; stderr, stdout and
the detached launcher log are empty. No model forward, optimizer or environment
call was made by this terminal analysis.

## Decision

The primary strong-baseline screen FAILED. GCN value-MPC did not establish
stable cost superiority over plain MPC. The separately prespecified secondary
graph-versus-flat absolute-cost screen PASSED under persistent change. This is
a limited representation-package development signal, not proof that adding RL
reliably improves a competent planner, isolated message-passing causality,
deployment online adaptation, clinical safety or independent confirmation.

Close this package. Do not add epochs, seeds, reward variants or retries.
Recommended next decision: consolidate the paper around supported graph-aware
controller evidence and bounded value-learning findings, rather than promote
the secondary result to the failed primary claim. Any new scientific comparison
needs its own complete prospective scope and approval. Stage E stays closed.

## Completed Execution

- Shared initialization120/120; continuation240/240 (graph120,flat120).
- Frozen evaluation240/240:60 per controller,48 per block,20 per condition/role.
- Graph7680 and flat7680 value updates; total15360 = old3040 + new12320.
  Actor updates0; evaluation optimizer updates0.
- Initial/final seals10/10 each, all before evaluation. Final1536 updates/model.
- Native steps38400/38400 = old4581 + new33819; native operations39600.
- New world starts528 plus the saved partial world's completion;71 complete
  worlds reused. Total600 complete trajectories, not601.
- Cumulative planner epochs9953664 include the explicitly authorized384-epoch
  replacement reservation; old failed dispatch remains consumed.
- Recovery phase times: admission1.631s, initialization3475.953s,
  continuation6923.565s, frozen evaluation6457.416s, analysis/archive84.558s.
  No phase transfer, refund or extra fit occurred.

Implementation7ee58eec114a4009212724593048f8e73a18a46e;
execution10f62ca6a1fff79253c0987114b1616ecef2aafc;
packet e1d427952b15a9bf2a66d385072f596c6f314aaf9b93e06f69ffb65d65ebf2be.
The old failed root remains unchanged. The recovery deliberately resumed the
same seeds from epoch37; block0 has mixed historical/repaired predictor training
history. All evaluation uses the repaired predictor. This is not a replication.

## All Prespecified Contrasts

Savings = comparator minus first controller. Positive favors the first.
Absolute values and intervals below are millions of synthetic objective units
per world, not observed currency. Percentages are the mean of paired-world
percent savings, not the ratio of pooled costs. Extra losses = first minus
comparator; deliveries change by the opposite amount in the settled cohort.
Intervals are descriptive95% two-level bootstrap intervals:2000 resamples of
FIVE independent blocks and four paired worlds per block/condition. They are
not multiplicity-adjusted confirmation. Positive blocks use absolute savings.

| First / comparator | Condition | Savings M [95%] | Mean paired % | Extra losses | Positive blocks |
|---|---|---:|---:|---:|---:|
| Graph / plain | No change | 0.333 [-0.511,1.222] | 0.702 | -7.30 | 3/5 |
| Graph / plain | Persistent | 0.314 [-0.610,1.462] | 0.607 | -5.15 | 3/5 |
| Graph / plain | Fast fluctuation | 0.633 [-0.036,1.397] | 1.818 | -12.40 | 4/5 |
| Flat / plain | No change | -0.153 [-1.304,0.919] | -0.673 | -1.55 | 2/5 |
| Flat / plain | Persistent | -0.449 [-1.290,0.626] | -1.423 | +0.65 | 1/5 |
| Flat / plain | Fast fluctuation | 0.424 [-0.317,1.076] | 0.775 | -6.55 | 3/5 |
| Graph / flat | No change | 0.486 [-0.125,1.216] | 1.222 | -5.75 | 4/5 |
| Graph / flat | Persistent | 0.764 [0.051,1.391] | 1.952 | -5.80 | 5/5 |
| Graph / flat | Fast fluctuation | 0.209 [-0.654,1.067] | 0.959 | -5.85 | 4/5 |
| Graph / uniform | No change | 1.573 [0.550,2.491] | 3.716 | -2.45 | 5/5 |
| Graph / uniform | Persistent | 0.860 [-0.051,1.737] | 2.067 | -3.05 | 4/5 |
| Graph / uniform | Fast fluctuation | 0.207 [-0.613,0.987] | 0.832 | -0.75 | 3/5 |
| Flat / uniform | No change | 1.087 [0.014,2.027] | 2.432 | +3.30 | 5/5 |
| Flat / uniform | Persistent | 0.096 [-0.746,0.937] | 0.041 | +2.75 | 3/5 |
| Flat / uniform | Fast fluctuation | -0.002 [-0.787,0.851] | -0.220 | +5.10 | 2/5 |
| Plain / uniform | No change | 1.240 [0.374,2.096] | 2.994 | +4.85 | 5/5 |
| Plain / uniform | Persistent | 0.546 [-0.279,1.257] | 1.397 | +2.10 | 4/5 |
| Plain / uniform | Fast fluctuation | -0.426 [-1.253,0.285] | -1.083 | +11.65 | 1/5 |

Persistent graph/plain block savings (IDs0..4), in millions:
2.004428,-0.116245,0.477993,0.204280,-0.999091. The absolute-cost interval
crosses zero and two blocks have negative means. The loss-direction criterion
passes only at the condition-mean level. Persistent block4 has0.5 extra losses.

Persistent graph/flat block savings, in millions:
1.019019,0.610060,0.631618,1.476842,0.080185. All five are positive and its
absolute-cost interval excludes zero. Its paired-percentage interval still
crosses zero: [-0.166213%,3.805774%]. Do not report a uniformly significant
percentage gain. Persistent block2 has1.25 extra losses despite lower cost.
The persistent mean extra-loss intervals also cross zero: graph/plain
[-11.35125,0.05], graph/flat[-12.45,0.25]. Other graph/flat adverse loss blocks
are no-change block1(+0.5) and fast-fluctuation block1(+4.0).

## Resource And Patient Context

| Condition | Plain cost M / losses / applied h | Graph cost M / losses / applied h | Flat cost M / losses / applied h | Uniform cost M / losses / applied h |
|---|---:|---:|---:|---:|
| No change | 38.885 /174.40 /302.95 | 38.552 /167.10 /342.81 | 39.038 /172.85 /324.79 | 40.125 /169.55 /384.00 |
| Persistent | 38.266 /153.10 /323.60 | 37.952 /147.95 /356.80 | 38.715 /153.75 /349.60 | 38.812 /151.00 /384.00 |
| Fast fluctuation | 40.211 /159.70 /305.14 | 39.578 /147.30 /364.35 | 39.786 /153.15 /361.99 | 39.785 /148.05 /384.00 |

Mean committed and applied totals reconcile in every trajectory after the common
settlement tail; requests and actual applications remain distinct saved fields.
Graph/plain actions differ at622/960,692/960,728/960 control boundaries across
the three conditions; graph/flat at660/960,719/960,741/960. Change is not itself
benefit. Persistent graph/plain lowers patient-loss cost by257500 and reactor
shortage cost by286559.52 units, but increases reagent purchase by164478.59,
reagent shortage by250863.02, and flexible labor by3777.01. These are arithmetic
contributions, not established causal mechanisms. Plain and flat have more
mean losses than uniform in every condition. Favorable means do not ensure
individual-world safety or clinical noninferiority.

## Independent Saved-Data Verification

A separate inline reader (standard JSON/gzip, NumPy arithmetic; no simulator or
model imports) reconstructed all600 trajectories/38400 rows from `payload/raw`.
It checked sequential world/role/epoch identity, nonfinite values, component sums,
reward=-cost, patient uniqueness and terminal-state persistence, integer patient
counts, loss events, full64-step settlement, zero tail requests, requested/
executed/applied labor and no trajectory-time optimizer updates. It reconciled
all raw costs, components, counts, actions and hours to the saved comparison.
All60 evaluation world groups have matching cohorts/tapes and correct final
model seals. All18 condition/contrast cells, every block/pair, component
differences, action differences and five-block bootstrap intervals independently
recompute (seed63090001, original contrast order). Largest floating arithmetic
reconciliation difference:5.21540641784668e-08 objective units; no material mismatch.

Reuse the14 passing zero-update tests/compileall, the560-source/2-runtime checks,
15360 finite fit receipts, completed-fit-state check and all-model test barrier.
Scientific source/config paths have no diff from the frozen implementation.
The terminal current-input locks and model hashes remain valid. No historical
campaign re-audit, scientific rerun or second archive was performed.

Existing archive `archives/payload.tar.gz`:467290525 bytes,4071 files. Each live
payload and each archive member was hashed independently against the receipt;
4070 inventory entries also reconcile (inventory excludes itself). No missing,
duplicate or unexpected file. SHA256:

- Archive:6871460bd881ffd1e453b0875b9dc2d4b70c06188a5e1c0a5dc2378f3b35a927
- Comparison:3e7fabd224ea6f427d5ddc5f365e665843ece0a30c48e2317bc96d32beef0d79
- Inventory:0d30cdb036496ed757e9d474bfa84b0b501ca0e1b70e97f066d75df7a7918d55

Local bytes verified only. No Dropbox export, cloud-sync check, collaborator
access check, push, PR, merge or external message. The finite independent
interpretation agent Dalton completed and was closed; no new audit gate.

## Claim Boundary And Handoff

Graph3169 versus flat3155 parameters (0.4418% gap) receive the same public
information and environment/update/query budgets. Depth and graph edges are not
isolated, and continuation data depend on each policy. Fixed training budget
does not prove convergence. Five blocks and descriptive intervals limit inference.
The uncalibrated synthetic support process and missing E1 field data limit
external validity. The earlier13.7493% updated/initial result is a separate
three-block study with a weak initializer, not this strong-baseline estimate.

The manuscript now distinguishes the failed primary cost screen, passed secondary
representation screen and patient trade-offs. The experiment package is consumed;
the same `gcn-rl` monitor is PAUSED (tool and saved configuration verified).
Manuscript source checks passed; no local LaTeX compiler is available, so PDF
compilation and rendered layout were not verified. No further
scientific execution is authorized by remaining time or this readout.

Sources: this directory's frozen protocol/authorization and
`results/capacity_value_comparison_20261004_recovery1/{launcher/terminal.json,
launcher/child-completed.json,payload/comparison.json,payload/artifact-inventory.json,
archive-receipt.json}`. Original raw/summary/model evidence remains in that payload.
