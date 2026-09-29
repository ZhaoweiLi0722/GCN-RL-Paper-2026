# Formal teacher/replay supervision audit

## Decision and scope

The reporting audit completed with methodology findings. It does not launch
training, change historical evidence, establish a causal explanation for the
null increment, or authorize a corrected formal rerun. The historical null
result applies to the implemented training procedure, not all online DDPG.

Evidence: `reports/2026-09-29-formal-replay-contract/audit.json`.
Auditor: `evaluation/audit_formal_replay_contract.py`.
Source: `9c0718b0da49d34f7f878ed2f036e96bf7bb3693`.
The locked teacher hash matches exactly:
`9ba2ac0873c0f68e6ecc4b443e0eace8230f8e151cceb485fafe218a7ac78d92`.
The historical training-manifest hash also matches. Ten effective configs,
ten summaries and 1,000 episode rows were read. Their current hashes were
checked again after analysis; they are not all independently historical hashes.

No PyTorch model was loaded. Only the reviewed historical numeric replay
insertion functions were extracted from Git and executed against an in-memory
recorder. No simulator transition, training update, or evaluation was run.
An independent index recurrence checked every reconstructed return, state,
action, terminal flag, first-step reward and discount multiplier.

## Findings

### 1. Offline and online rewards have different meanings

Heuristic pretraining directly inserts 1,040 one-step absolute environment
rewards, scaled by 1e-9. The AFD loader passes 314 cached absolute rewards to
`agent.observe`; with the effective four-step mode, this emits 290 length-four
items and eight each of lengths one, two and three. Their returns are still
absolute costs, not anchor-relative rewards. Cached immediate rewards range
from -125,730,528 to -616,025.1875.

New online data instead use four-step sums of one-step anchor-relative rewards.
The buffer retains the offline rows and samples uniformly because no explicit
online fraction is configured. A single critic consequently fits heterogeneous
reward definitions without conditioning its prediction on data source. This
is a target-consistency limitation, not proof that it caused the measured null.

Source: `src/models/gcn_ddpg.py:1237-1299,2381-2427`;
`evaluation/run_gcn_residual_sweep.py:1843-1890`;
`src/rl/replay_buffer.py:40-89`.

### 2. Some cached records do not form a trajectory

Of 306 nonterminal adjacent cached pairs, 153 have a next observation that
differs from the following record's current observation. The historical GCN
and flat replay paths both emit 153 windows containing such a discontinuity,
out of 314 emitted items. No pending tail remains.

This is not merely floating-point uncertainty about a full hidden state.
Observed-state inequality already disproves exact adjacency. Conversely,
equality would not prove full simulator/RNG continuity, so the check is a
necessary-condition diagnostic, not a complete trajectory certificate.

The collection code explains the mechanism: the teacher stores a one-step
counterfactual transition from a clone, while DAgger collection advances the
actual trajectory with the student action when teacher-behavior probability
is zero. Joining consecutive counterfactual records as if they were one
trajectory therefore has no justified multi-step Bellman interpretation.

Source: `evaluation/network_residual_headroom.py:697-740`;
`evaluation/collect_multiscenario_dagger.py:278-320`.
The numeric reconstruction is evidence about the frozen data path, not a
direct inspection of the final binary replay checkpoint or a rerun of training.

### 3. Calibration also uses a different bootstrap convention

The 100-step pre-online calibration helper computes `reward + gamma * Q_next`
with terminal masking, without reading the replay discount multiplier. The
main update helper reads that multiplier. For the 306 multi-step AFD items,
the calibration path therefore applies gamma rather than gamma^n to the
serialized multi-step return. Heuristic one-step items do not have this
particular discrepancy. This was documented, not retroactively repaired.

Source: `src/models/gcn_ddpg.py:1452-1510,1545-1621,1816-1876`.

### 4. Teacher support is broader than policy correction support

The cache contains 13 options: anchor plus specimen, reagent, and combined
routing/network corrections of +/-0.05 and +/-0.10. The best feasible groups
are anchor 29, specimen 154, reagent 89 and combined 42. Thus 131/314 winners
have group labels outside the formal actor's anchor/specimen correction set.
All ten summaries report 314 calibration samples and zero excluded samples.

The critic regresses `Q(teacher action) - Q(anchor)` against the nonnegative
best feasible option advantage. The code calls this a ranking loss, but under
these settings it is pointwise MSE: optional margin and pairwise terms are
zero. Lack of group filtering is a support mismatch to disclose; group labels
alone do not prove that every corresponding decoded action is unreachable.

Source: `src/models/gcn_ddpg.py:775-810,1314-1450`;
`src/rl/critic_advantage.py:43-91`.
Prior support-alignment development results remain separate evidence and must
be consulted before proposing to repeat that intervention.

### 5. Logged settings are not cache-generation provenance

Training summaries echo lookahead 52, epsilons 0.32--1.0, and a capacity group
from caller settings even when an existing cache is loaded. The actual cache
has the 13 options above, no capacity-only option, and no serialized label
horizon or replication count. The generation configuration points to earlier
scenario configs that were not found at the referenced persistent-root paths
in this audit. The caller's 52/3 values therefore do not verify how cached
labels were generated. No replacement horizon was guessed.

Source: `evaluation/run_gcn_residual_sweep.py:490-531,736-742` and
`experiments/configs/patient_indexed_specimen_routing_mac_mps_bounded_gain01_groupgate_broad_dagger1.json`.

### 6. Online updates did occur; coefficients do not prove domination

Each of the ten runs has 5,200 online critic updates and 2,600 actor updates.
The 500-update actor warm-up was consumed by the 500 offline critic updates;
it was not an additional 500-update online actor delay. First-episode logs
already report actor updates on half the critic updates.

The following are ranges across five seeds of each run's mean over 100 episode
means, not confidence intervals. Equal update counts per episode were checked.

| Logged quantity | GCN range | Flat range |
| --- | ---: | ---: |
| Bellman MSE | 0.008399--0.010157 | 0.007364--0.015553 |
| Weighted teacher advantage MSE | 0.000111--0.000128 | 0.0000535--0.0000626 |
| Teacher/Bellman loss-value ratio | 1.13--1.39% | 0.40--0.73% |
| Actor imitation loss (weight 1) | 0.051269--0.053821 | 0.011884--0.013134 |
| Weighted frozen-reference loss (weight 500) | 0.000171--0.000239 | 0.000444--0.000832 |
| Weighted self-imitation loss | 0.0000271--0.0000344 | 0.0000149--0.0000246 |
| Active self-imitation sample fraction | 10.04--11.74% | 8.03--8.87% |

All audited metrics are finite. A coefficient of 500 is not evidence that the
reference term numerically dominates the objective. Equally, small teacher
loss values do not prove its gradients were overwhelmed. Gradient norms,
directions and causal ablations are absent from this audit. Per-minibatch
online sampling fractions are not logged under uniform sampling; do not
substitute the self-imitation eligibility fraction for them.

## Consequence

Preserve the package-vs-anchor and graph-vs-flat cost observations, but make
their scope the implemented procedure. Do not convert a null final-vs-frozen
comparison into a claim that the environment is globally optimal or that
online RL is inherently ineffective. These audit findings motivate clean
data-path requirements for any future development protocol:

- Define one consistent value target, or explicitly separate critics/targets.
- Build multi-step returns only from actual contiguous trajectories; retain
  independent counterfactual records as one-step supervision.
- Keep serialized horizon/discount semantics identical in every update path.
- Align teacher action support with the tested policy and verify actual
  cache-generation metadata instead of trusting echoed caller settings.
- Log measured loss contributions and gradient diagnostics before diagnosing
  an update as dominated or ineffective.

These are requirements, not a prescription to retune after seeing the formal
holdout. Any correction-and-retrain study needs a new approved development
protocol and fresh evaluation streams. It must not silently replace the old
results. Existing support-alignment and online-only replay explorations also
need to be distinguished from a jointly consistent replay implementation.

## Reproduction and limits

```bash
python -m evaluation.audit_formal_replay_contract \
  --training-root /Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/confirmation/ddpg_routing_primary_100/patient_indexed_specimen_routing_mac_mps_ddpg_confirmation_100 \
  --teacher /Users/lizhaowei/gcnrl-patient-routing-persistent/results/patient_indexed_specimen_routing_mac_mps_primary/teachers/bounded_gain01_dagger1/teacher_train_dagger.npz \
  --output /private/tmp/formal-replay-audit-new-output.json
python -m unittest tests.test_formal_replay_contract
```

Use the persistent project's virtual-environment interpreter. The output must
not exist. The audit needs the pinned historical Git objects and refuses
teacher or manifest hash mismatch. No checkpoint is deserialized. Full
cache-generation provenance and gradient attribution remain unresolved, not
silently passed. The remaining autonomous queue can proceed independently.

Validation: 94 focused tests, full Python compilation and `git diff --check`
passed. A second read-only replay audit reproduced the JSON byte for byte.
The manuscript was edited at source level; PDF compilation/visual review
remains unavailable because no TeX engine is installed. No remote action was
performed.
