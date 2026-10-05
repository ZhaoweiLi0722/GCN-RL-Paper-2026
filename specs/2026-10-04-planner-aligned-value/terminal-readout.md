# Planner-Aligned Value Comparison: Terminal Timeout And Saved Results

2026-10-05 UTC. Entry b9f18ef was clean on the expected branch. This single
attempt is FAILED, not a completed480-trajectory comparison. The supervisor
recorded exit1/child exit1 at05:31:13Z, elapsed19560.153425s. Read-only process
checks at05:37Z found neither PID95697 nor95709 and no related experiment.
No relaunch, repair, model load/forward, optimizer or environment call occurred
during this handoff. The ordinary stdout/stderr/outer log files are empty, but
child-failure.json contains a real TimeoutError traceback; empty stderr is NOT
a clean-success indication.

## Execution And Cause

The frozen evaluation phase reached14400.002917s against its14400s cap,
not the overall32400s cap. The active H16 world had used13.669818s of its240s
owner cap. Reference collection took5083.981840s and fitting70.959470s.
Unused global or other phase time was not transferred. The earlier measured
timing warning materialized. This is an engineering time-budget failure,
not evidence that training failed or that the method has no benefit.

Reference120/120; each of three methods120/120fit cohorts and3840/3840value
updates,11520total,actor0. All15new models and5ancestor bindings were sealed
before evaluation. Native calls30666/30720; native operations31626/31680;
forwards29040/30000; evaluation optimizer0. All480worlds were started but only
479completed:120reference plus359evaluation. The five H8roles each have60
complete evaluations; H16 has59. Blocks0..3 each have72,block4 has71.

The only unfinished world is evaluation-b4-c2-j3-plain_h16, seed64024203:
10raw native rows (epochs0..9) are saved;54native steps remain, comprising
38control and16settlement. Its partial cost is excluded from every comparison.
The 768-epoch pending forecast reservation dispatched411 calls, has no completion
receipt and remains consumed. Full native completion is not synonymous with
exact forecast-level resumability. The subsequent960saved-training-root forwards
were not performed, so ranking diagnostics are unavailable.

Implementation cbe0b7d; frozen881997cf3029d0d6b5369e429b01cec5493ddf19;
execution46eba5b65f61a6c63053cf6bf32bd2f2d5059fb3; packet
a0cbcef8d523c69234c0b9568536ba7a0bd42bcd138e4d63111410beabafba74.
Scientific files/configs are unchanged. Reward, scenarios, model structure,
seeds, support and thresholds were not revised.

## Supported Answer

The COMPLETE primary comparisons give a favorable development signal against
plainH8 under persistent change: aligned TD saves0.947873M synthetic objective
units per world (95% descriptive interval[0.432159,1.484289]M), mean paired
savings2.562%, all5block means positive, and12.55 fewer simulated losses/world.
It also beats same-data observed-TD continuation under that condition:
1.324003M[0.364893,2.430482]M,3.671% paired savings,13.80 fewer losses.

However, the required incremental comparison against the existing frozen model
does NOT pass:0.246458M[-0.181340,0.679558]M,0.679%, only4/5positive blocks.
Thus the prespecified conjunction of ALL THREE primary screens fails on
complete primary data. The new update is not established as an improvement
over the existing GCN model, even though its controller beats plainH8 here.

Aligned TD versus same-tail MC is not distinguishable in cost: persistent
savings-0.024627M[-0.251761,0.169010]M. Do not claim a unique TD advantage.
MC is direct-return policy evaluation, not automatically a non-RL method.
Observed-TD continuation can degrade the already-trained controller; outperforming
that continuation is not enough to claim improvement over the frozen ancestor.

The COMPLETE persistent H16 contrast also includes zero:
0.512787M[-0.109672,1.088919]M. H16 uses double the planning queries. Its fast-noise
cell has only19pairs and is descriptive only. The missing final H16/noise world
cannot change the already complete three primary comparisons, but it prevents
calling the six-role matrix or original attempt complete.

## All Saved Contrasts

Savings = comparator minus first controller; positive favors first. M means
millions of synthetic objective units, not observed currency. Percentages are
means of paired-world ratios, not pooled-cost ratios. Extra losses = first minus
comparator; settled deliveries change oppositely. Complete cells use2000
two-level resamples of FIVE blocks and four worlds/block with the frozen
bootstrap seed and original contrast order. Incomplete H16/noise cells have
19pairs, no imputation and no interval; their block4 mean uses three worlds.
Nominal RNG draws for each unavailable cell are consumed without computing an
interval, preserving downstream complete-cell bootstrap positions. These are
descriptive development intervals, not multiplicity-adjusted confirmation.

| First / comparator | Condition | n | Savings M [descriptive95%] | Mean paired % | Extra losses | Positive blocks |
|---|---|---:|---:|---:|---:|---:|
| planner_tail_td / plain_h8 | No change | 20 | 0.539 [-0.267,1.486] | 1.401 | -7.80 | 4/5 |
| planner_tail_td / plain_h8 | Persistent | 20 | 0.948 [0.432,1.484] | 2.562 | -12.55 | 5/5 |
| planner_tail_td / plain_h8 | Fast fluctuation | 20 | 0.546 [-0.213,1.420] | 1.450 | -14.95 | 3/5 |
| planner_tail_td / existing_frozen | No change | 20 | -0.056 [-0.776,0.738] | -0.142 | -2.35 | 3/5 |
| planner_tail_td / existing_frozen | Persistent | 20 | 0.246 [-0.181,0.680] | 0.679 | -3.40 | 4/5 |
| planner_tail_td / existing_frozen | Fast fluctuation | 20 | 0.093 [-0.547,0.838] | 0.224 | -1.50 | 2/5 |
| planner_tail_td / observed_td | No change | 20 | 0.949 [-0.601,2.939] | 2.591 | -11.05 | 2/5 |
| planner_tail_td / observed_td | Persistent | 20 | 1.324 [0.365,2.430] | 3.671 | -13.80 | 5/5 |
| planner_tail_td / observed_td | Fast fluctuation | 20 | 1.542 [0.484,2.705] | 3.343 | -20.85 | 5/5 |
| planner_tail_td / planner_tail_mc | No change | 20 | 0.016 [-0.444,0.415] | -0.093 | -0.30 | 3/5 |
| planner_tail_td / planner_tail_mc | Persistent | 20 | -0.025 [-0.252,0.169] | -0.261 | -1.15 | 3/5 |
| planner_tail_td / planner_tail_mc | Fast fluctuation | 20 | 0.224 [-0.171,0.800] | 0.575 | 0.45 | 3/5 |
| planner_tail_td / plain_h16 | No change | 20 | 0.146 [-0.629,1.122] | 0.368 | -4.85 | 3/5 |
| planner_tail_td / plain_h16 | Persistent | 20 | 0.513 [-0.110,1.089] | 0.924 | -6.15 | 4/5 |
| planner_tail_td / plain_h16 | Fast fluctuation | 19 | 1.137 [not estimated] | 2.443 | -11.63 | 5/5 |
| plain_h8 / existing_frozen | No change | 20 | -0.595 [-1.355,0.089] | -1.659 | 5.45 | 2/5 |
| plain_h8 / existing_frozen | Persistent | 20 | -0.701 [-1.406,-0.074] | -2.041 | 9.15 | 0/5 |
| plain_h8 / existing_frozen | Fast fluctuation | 20 | -0.454 [-1.136,0.214] | -1.324 | 13.45 | 1/5 |
| plain_h8 / observed_td | No change | 20 | 0.411 [-1.327,2.566] | 1.003 | -3.25 | 2/5 |
| plain_h8 / observed_td | Persistent | 20 | 0.376 [-0.764,1.637] | 1.064 | -1.25 | 3/5 |
| plain_h8 / observed_td | Fast fluctuation | 20 | 0.996 [-0.234,2.418] | 1.805 | -5.90 | 3/5 |
| plain_h8 / planner_tail_mc | No change | 20 | -0.523 [-1.284,0.201] | -1.682 | 7.50 | 1/5 |
| plain_h8 / planner_tail_mc | Persistent | 20 | -0.973 [-1.631,-0.379] | -3.008 | 11.40 | 0/5 |
| plain_h8 / planner_tail_mc | Fast fluctuation | 20 | -0.323 [-0.900,0.264] | -1.007 | 15.40 | 0/5 |
| plain_h8 / plain_h16 | No change | 20 | -0.393 [-1.346,0.354] | -1.180 | 2.95 | 2/5 |
| plain_h8 / plain_h16 | Persistent | 20 | -0.435 [-1.261,0.334] | -1.796 | 6.40 | 2/5 |
| plain_h8 / plain_h16 | Fast fluctuation | 19 | 0.674 [not estimated] | 1.121 | 2.26 | 4/5 |
| existing_frozen / observed_td | No change | 20 | 1.006 [-0.429,2.920] | 2.651 | -8.70 | 3/5 |
| existing_frozen / observed_td | Persistent | 20 | 1.078 [0.173,2.139] | 2.987 | -10.40 | 5/5 |
| existing_frozen / observed_td | Fast fluctuation | 20 | 1.450 [0.171,2.966] | 3.047 | -19.35 | 5/5 |
| existing_frozen / planner_tail_mc | No change | 20 | 0.072 [-0.713,0.856] | -0.056 | 2.05 | 2/5 |
| existing_frozen / planner_tail_mc | Persistent | 20 | -0.271 [-0.804,0.202] | -1.001 | 2.25 | 1/5 |
| existing_frozen / planner_tail_mc | Fast fluctuation | 20 | 0.131 [-0.512,0.828] | 0.276 | 1.95 | 3/5 |
| existing_frozen / plain_h16 | No change | 20 | 0.202 [-0.404,0.841] | 0.432 | -2.50 | 4/5 |
| existing_frozen / plain_h16 | Persistent | 20 | 0.266 [-0.512,0.992] | 0.198 | -2.75 | 3/5 |
| existing_frozen / plain_h16 | Fast fluctuation | 19 | 1.006 [not estimated] | 2.060 | -10.32 | 4/5 |
| observed_td / planner_tail_mc | No change | 20 | -0.934 [-3.227,0.774] | -3.313 | 10.75 | 3/5 |
| observed_td / planner_tail_mc | Persistent | 20 | -1.349 [-2.387,-0.437] | -4.274 | 12.65 | 0/5 |
| observed_td / planner_tail_mc | Fast fluctuation | 20 | -1.319 [-2.608,-0.247] | -2.966 | 21.30 | 0/5 |
| observed_td / plain_h16 | No change | 20 | -0.804 [-2.726,0.819] | -2.742 | 6.20 | 2/5 |
| observed_td / plain_h16 | Persistent | 20 | -0.811 [-2.209,0.411] | -3.049 | 7.65 | 2/5 |
| observed_td / plain_h16 | Fast fluctuation | 19 | -0.489 [not estimated] | -1.213 | 9.84 | 2/5 |
| planner_tail_mc / plain_h16 | No change | 20 | 0.130 [-0.620,0.897] | 0.458 | -4.55 | 3/5 |
| planner_tail_mc / plain_h16 | Persistent | 20 | 0.537 [-0.005,1.045] | 1.184 | -5.00 | 4/5 |
| planner_tail_mc / plain_h16 | Fast fluctuation | 19 | 0.902 [not estimated] | 1.814 | -12.11 | 4/5 |

Machine-readable block means, intervals, harm counts and counts are in
terminal-saved-data.json. The original reader first reconciled all479completed
raw trajectories (30656rows), including component/reward sums, integer patients,
identity and terminal persistence, settlement, requested/committed/applied
two-step labor, tape/cohort matching, model bytes and the all-sealed barrier.
The compact tables reuse those reconciled summaries. This is saved-data
analysis only, not a rerun or a relaxed480-trajectory success reader.

## Trade-offs And Compute

Persistent TD/plainH8 has3cost-harm worlds out of20, despite5positive block means,
and no extra-loss worlds. TD/frozen has9cost-harm and5extra-loss worlds out of20.
Outside persistent change, TD/plainH8 cost intervals cross zero; patient harms
occur in7no-change and2fast-noise worlds. TD/MC fast-noise averages0.45extra losses.
Favorable condition means are not a clinical-safety or noninferiority guarantee.

Persistent TD/plainH8 actions differ at757/960control boundaries. TD uses53.602
additional applied flexible hours/world. Cost savings include627500patient-loss
and450000expiry units, offset by305761.49more reagent-purchase units and5992.20more
flexible-labor units. These are accounting contributions, not identified causes.
All candidate action/component contrasts are preserved in terminal-saved-data.json.

Measured median / p95 decision seconds from complete trajectories:
- planner_tail_td: 0.699025 / 0.857350 (2880 decisions).
- plain_h8: 0.700974 / 0.860053 (2880 decisions).
- existing_frozen: 0.700559 / 0.860846 (2880 decisions).
- observed_td: 0.703314 / 0.864429 (2880 decisions).
- planner_tail_mc: 0.700875 / 0.858935 (2880 decisions).
- plain_h16: 1.414903 / 1.652984 (2832 decisions).
H8 TD and plainH8 have similar measured decision times here; H16 is roughly twice
as slow. This does not erase training costs:11520new updates,11520tail copies and
374400extra tail-prediction steps were used. Ordinary planning charged9924864
prediction steps, including the pending768reservation. TD/MC share training data
and update counts but not bootstrap-forward compute. Historical ancestor training
is separate,5x1536updates; the five models are not freshly independent replicas.

## Preservation And Limits

Original root results/capacity_planner_tail_20261004 remains untouched.
Failure-state.pkl.gz is33224bytes; model bytes for all20bindings verify. Its
pickle/loadability was not inspected or promised. Run inventory at handoff:
2868files/618946681bytes, tree SHA256
460450435dae931fc7c1453115fa4bb83114b736f75847f1572e703e5420dc72.
Digest convention: sort files lexicographically; form relative path/bytes/sha256
records; hash compact sort_keys JSON of the list. Key artifact hashes are in
terminal-saved-data.json. This binds preserved local files, not an off-device copy.

The runner aborted BEFORE comparison.json, artifact-inventory.json, completion.json
and archives/payload.tar.gz were created. No archive receipt exists; do not claim
archive verification. This handoff adds separate non-locked documents only, not a
replacement success receipt or a duplicate archive. Dropbox/cloud sync/Howard access
remain unverified and no export was made. Existing38tests/compileall and frozen
source/runtime/seed checks were reused; no scientific code changed.

Public forecast-tail labels are approximate model returns, not native patient
counterfactuals. The five ancestors retain historical training and block0mixed
predictor provenance. No isolated GCN-edge, deployment-online adaptation, DDPG,
full TD-MPC, clinical safety or publication claim follows. Synthetic support
labor remains uncalibrated; E1data are still missing and StageE remains closed.

## Handoff

Close this attempt without retry, added training, reward search or changed caps.
The manuscript now records both the favorable persistent TD/plainH8 comparison
and the failed required TD/frozen increment, with terminal incompleteness explicit.

Hegel's finite read-only sidecar is complete and closed. Static inspection finds
component restore interfaces but no current runner/budget remaining-only entry;
_episode creates a new environment and the interrupted candidate loop is local.
A separate authorized recovery would need a verified additive restore path,
explicit treatment of the consumed incomplete reservation,54remaining native
steps and the960original diagnostic forwards, plus readout/IO budgets. None of
that work is authorized by this terminal handoff or unused old time.

The immediate decision is whether to commission that strictly remaining-only,
evaluation/diagnostic completion package. No new training is needed for it.
A full numerical recovery proposal and explicit approval must precede scientific
execution. This monitor is PAUSED after recording the terminal handoff; no
background training or further result collection remains running.

