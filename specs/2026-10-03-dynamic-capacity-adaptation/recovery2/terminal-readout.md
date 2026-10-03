# Dynamic Capacity Recovery2: Complete, No Reliable Online Benefit

2026-10-03 UTC. The one approved remaining-work run completed with exit0 in
2213.987seconds (36.90minutes). This is a completed comparison, not an engineering
abort. The prespecified `candidate_online_signal` is false. Do not retry or add
epochs, samples, reward variants or a confirmation run under this authorization.

## Execution And Evidence

- Implementation674bf60; execution9690b4d; packet
  d09725aba00cbc11c23d4f1883e0d66c150f7bfeb73acc7eb6b8b4e0f42f7240.
- Teacher36/36, offline36/36, model seals3/3, six-arm evaluation216/216;
  288complete64-step trajectories including the35imported and1continued.
- Newactor3104/3104 andcritic3104/3104; including importedblock0,
  actor3936/3936 andcritic3936/3936. Evaluation includes1440real online DDPG
  actor/critic pairs, distinct from offline imitation, warmup and offline RL.
- Newnative16145/16145; native operations16651, forwards21344,
  forecastepochs885120, newfilter1614500, savedfilter75100; exact counters
  and all time-owner limits pass. Old attempts remain separately consumed.
- Supervisor35197 andchild35211 exited; no related runner command remains.
  No failure/overrun marker; stderr and detached launcher log are empty.
- Current packet/input/source hashes match. Three sealed model byte hashes match.
  Existing archive951members and source files match; archive190323059bytes,
  SHA25668ff62373ccf58d7f2206ab445e2ba5799e83197cf18044bafd0bac0b1471a9f.
  No archive recreated, prior history re-audited, Dropbox export or cloud claim.

The run is `results/dynamic_capacity_adaptation_20261003_recovery2`.
Its `payload/comparison.json` is the frozen reader's complete result;
SHA256163d64e4664d593d36680568fdd9ad4d2e387c2c0541776be947180d94d2d4f6.
Independent raw arithmetic is in
`reports/dynamic_capacity_recovery2_20261003_independent_raw_check.json`;
terminal/archive/process checks are in
`reports/dynamic_capacity_recovery2_20261003_terminal_evidence.json`.
No model or patient-environment execution was used for the terminal readout.

Hypatia independently read all216evaluation gzip files/13824rows without the
project analysis helper:388published mean numbers,504component totals and all
patient counts agree exactly. Raw cost is the sum of64recorded row costs;
outcomes come from final patient identities, not an inferred rate. The copied
training data remain provenance-bound; no claim of a new independent rerun.

## Complete Performance Comparison

Each condition contains3training blocks x4paired worlds. Savings are reference
cost minus online cost; positive means cheaper online. Costs below are thousands
of modeled units, not calibrated currency. Percent is the mean of paired-world
percent savings, not a ratio of aggregate costs. Extra lost is online minus
reference, per world; extra delivered is its negative because cohorts settled.

| Reference | Condition | Savings, thousands | Mean savings % | Extra lost |
|---|---|---:|---:|---:|
| Frozen history | No change | 214.642 | 0.542 | 3.417 |
| Frozen history | Persistent change | 90.484 | 0.341 | 4.500 |
| Frozen history | Fast fluctuation | -300.556 | -0.780 | 6.250 |
| Matched-exploration frozen | No change | -67.825 | -0.219 | 2.250 |
| Matched-exploration frozen | Persistent change | 53.460 | 0.066 | 2.750 |
| Matched-exploration frozen | Fast fluctuation | -41.671 | -0.147 | 5.333 |
| Adaptive rule | No change | -151.133 | -1.051 | 15.250 |
| Adaptive rule | Persistent change | 525.232 | 1.103 | 14.083 |
| Adaptive rule | Fast fluctuation | 91.301 | -1.438 | 19.667 |
| ID-MPC | No change | -1936.668 | -5.798 | 12.250 |
| ID-MPC | Persistent change | -1991.140 | -4.934 | 16.667 |
| ID-MPC | Fast fluctuation | -614.359 | -2.438 | -1.083 |
| Fixed allocation | No change | -562.466 | -1.871 | 12.833 |
| Fixed allocation | Persistent change | -1828.762 | -4.623 | 21.750 |
| Fixed allocation | Fast fluctuation | -1681.802 | -4.754 | 24.333 |

Fixed-allocation contrasts were requested but absent as a separate contrast in
comparison.json; independent raw arithmetic supplies them, without changing the
locked primary screen. Opposite signs for adaptive fast-fluctuation mean cost
savings and mean percentage are possible because their denominators differ.
All individual-world and block results are retained in the independent JSON.

Persistent-change primary block savings, thousands:

| Reference | Block0 | Block1 | Block2 |
|---|---:|---:|---:|
| Frozen history | -292.490 | 734.864 | -170.923 |
| Matched-exploration frozen | -0.000574 | 0.001907 | 160.378 |

The descriptive hierarchical95%bootstrap intervals for persistent savings are
[-673.980,846.155]thousand against frozen history and
[-458.600,625.240]thousand against matched exploration. Corresponding mean
paired-percent intervals are[-1.533,2.411]% and[-1.019,1.246]%. These are
three-block pilot intervals, not confirmatory significance or clinical safety.
Both primary contrasts fail every prespecified criterion: all-block positive
savings, interval above zero, and nonpositive mean additional patient losses.

Executed hours differ at1728/1728control boundaries versus frozen history and
1404/1728versus matched exploration (locked1e-9L1 threshold). The latter's mean
cumulative absolute difference per world by block is0.01205,0.05234,82.85130hours.
Thus the updates run and some actions change; substantial differences are
concentrated inblock2. Neither changed actions nor optimizer counts prove gain.

## What Changes The Next Decision

1. The original "perhaps there is no useful control headroom" explanation is
   insufficient here: ID-MPC has lower mean cost and fewer losses in persistent
   change in each block versus online DDPG. Even fixed allocation outperforms it
   on these two mean endpoints. This is an observed comparator gap, not a proof
   that neural online updates can recover the gap or that MPC is globally optimal.
2. Cost improvement is not aligned with improved patient outcomes in the small
   persistent primary averages. Against frozen history, patient-loss cost rises
   225.000thousand and expiry150.000thousand, while purchase falls224.928thousand,
   reagent-shortage165.800thousand and bioreactor-shortage113.116thousand. The
   net90.484thousand saving is a modeled trade-off, not clinical improvement.
   This supports reconsidering the objective/constraints; it does not establish
   an accounting bug, double charging or that an arbitrary reward weight is wrong.
3. Do not assume offline initialization was strong or near-optimal. The block
   spread and simpler-controller gap make competence an unresolved limitation.
   The historical teacher data and corrected predictor are retained
   as mixed initialization, not silently treated as homogeneous corrected training.

## Closure And One Next Decision

Close this attempt and pause the samegcn-rlmonitor. Preserve negative findings.
Recommend no further epochs or reward-weight sweep for the current DDPG package.
The next decision is whether to reframe the next development package as
**patient-outcome-constrained policy improvement against the demonstrated
ID-MPC/fixed-allocation baselines**, rather than unconstrained aggregate-cost
optimization. First specify the acceptable outcome constraint and a bounded
complete comparison; no new training or reward change is authorized by this
readout. A guardrail in a future objective is not a present safety guarantee.

E1field calibration is still absent. This is an uncalibrated synthetic support
process and development seed reuse, not clinical evidence, independent
confirmation, isolated graph attribution or a publication guarantee. Some
productive/idle-hour secondary breakdowns are missing for144/216trajectories;
primary costs/outcomes are complete. Do not impute missing hours or reopen
StageE/holdout. No scientific code changes or tests were needed in this readout;
reuse the16passed artificial tests and prior compileall.
