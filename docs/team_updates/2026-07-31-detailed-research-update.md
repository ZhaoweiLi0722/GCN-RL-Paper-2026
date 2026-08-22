# Detailed Research Weekly Update

- Reporting period: July 20-31, 2026
- Audience: GCN-RL project team
- Primary question: Can graph-aware residual reinforcement learning improve
  adaptive capacity and resource-transfer decisions in a patient-condition-aware
  20-clinic regenerative medicine manufacturing network?

## Executive Summary

This week we separated three claims that had previously been discussed
together:

1. whether learning bounded corrections around a strong heuristic is useful;
2. whether the graph representation contributes value beyond a matched flat
   residual policy;
3. whether online actor-critic updates add value beyond graph-policy
   pretraining.

The evidence supports the first two claims under clearly defined matched
scenarios. AFR-GCN-DDPG produced a small but statistically stable improvement
over MDL-2 under demand-prior drift. The network extension produced a
materially larger and statistically significant advantage over both MDL-2 and
a parameter-matched flat policy under geographically structured regional
demand shifts. A subsequent online AFR-GCN-TD3 screen also produced a final
graph policy that outperformed MDL-2 and matched flat AFR-TD3 in the regional
screen.

The third claim is not established. In the matched regional screen, the
increment from online TD3 updates over the frozen pretrained graph policy was
directionally favorable but not statistically resolved. The completed
five-scenario zero-shot evaluation then found that the same regional-specialist
checkpoints did not generalize reliably across nominal, severe global,
regional, abrupt-shift, and compound-stress regimes. This is a useful
diagnostic result, but it is not a fair final test of a generalist policy
because the checkpoints were trained as regional specialists rather than on a
multi-scenario distribution.

## 1. Environment And Experimental Design Updates

The current environment extends the patient-condition framework introduced in
Howard's pull request with the following network features:

- 20 geographically located clinics;
- distance-based transfer lead times of one to three periods;
- delayed transfers of fungible reagents and idle bioreactor capacity;
- identity-bound patient specimens with specimen pooling and transfer
  prohibited;
- patient deterioration while waiting and during manufacturing;
- therapy discard and bioreactor cleaning following manufacturing-period
  ineligibility;
- regional demand drift, abrupt regime shifts, and compound regional stress;
- paired common-random-number evaluation for operating and clinical outcomes.

These changes align the action space with the graph structure. Earlier
replenishment-only residual policies had limited ability to convert geographic
information into network decisions. The network residual policy can modify
node-level replenishment and edge-level reagent and idle-capacity transfers.

## 2. Proposed Method And Terminology

The canonical proposed method remains AFR-GCN-DDPG.

AFR means advantage-filtered residual and describes the complete controller,
not a new actor-critic algorithm and not a DDPG-only mechanism. It combines:

1. an MDL-2 operational anchor;
2. a GCN state encoder for clinic and geographic-edge information;
3. an actor-critic backbone that produces bounded corrections to the anchor
   action;
4. advantage-filtered distillation, which retains a nonzero teacher correction
   only when paired look-ahead predicts an improvement over the anchor;
5. feasibility projection and deployment safeguards.

Advantage-filtered distillation is a training-data construction and
regularization step inside AFR. It is not a second deployed policy. The current
AFR implementation has been evaluated with both DDPG and TD3 backbones:

| Name | Uses AFR? | Role |
|---|---|---|
| AFR-GCN-DDPG | Yes | Canonical proposed controller |
| AFR-GCN-TD3 | Yes | Principal matched backbone and stability ablation |
| AFR-Flat-DDPG / AFR-Flat-TD3 | Yes | Representation-matched controls |
| Pure GCN-DDPG / GCN-TD3 | No | Anchor and residual-formulation ablations |
| GCN-SAC / GCN-PPO | No in the current study | Secondary from-scratch family baselines |

The RTX 4090 conservative TD3 campaign used an MDL-2 anchor, graph residual
actor, advantage-filtered pretraining, deployment safeguards, and TD3
fine-tuning. It should therefore be called **AFR-GCN-TD3**, not simply
GCN-TD3. Earlier notes sometimes used the shorter label; this update uses the
precise name. DDPG remains canonical because it is continuous with the prior
paper and performed better than matched AFR-GCN-TD3 in the five-seed
replenishment-residual comparison. TD3 remains a serious matched ablation, not
a non-AFR method.

## 3. Verified Experimental Results

### 3.1 AFR-GCN-DDPG Under Demand-Prior Drift

Protocol:

- 300 training episodes;
- five independent training seeds;
- 100 paired evaluation replications per seed;
- 500 paired outcomes in total;
- integrated patient-condition, geography, and demand-prior drift setting.

AFR-GCN-DDPG versus MDL-2:

- mean cost difference: -$207,485;
- relative cost difference: -0.016360%;
- paired 95% confidence interval: [-$332,641, -$91,367];
- paired wins: 323 of 500;
- completion service difference: +0.000340;
- eligibility difference: +0.000211;
- average waiting-time difference: -0.000612;
- patients-lost difference: -2.32;
- at-risk-unserved difference: -2.55.

The result establishes a statistically stable anchor improvement, but the
absolute economic effect is small.

The matched flat residual DDPG policy also improved on MDL-2:

- relative cost difference: -0.013216%;
- paired 95% confidence interval: [-$286,899, -$46,230].

The GCN policy improved the point estimate by a further 0.003145% relative to
the matched flat policy, but the graph-versus-flat confidence interval
[-$89,702, $14,025] included zero. This replenishment-only experiment supports
residual learning, but does not independently establish a graph advantage.

Traceable result:
[demand-drift robustness results](../../specs/2026-07-20-demand-drift-robustness/results.md).

### 3.2 Matched DDPG And TD3 Comparison

The matched TD3 study used the same 300-episode, five-seed protocol and the
same anchor, graph representation, residual action, teacher targets, and
evaluation streams.

AFR-GCN-TD3 versus MDL-2:

- relative cost difference: -0.001248%;
- paired 95% confidence interval: [-$142,607, $99,437].

AFR-GCN-DDPG versus AFR-GCN-TD3:

- relative cost difference: -0.015112%;
- paired 95% confidence interval: [-$234,659, -$148,401];
- every training-seed mean favored DDPG.

This comparison supports DDPG as the canonical proposed backbone. TD3 remains
useful as a stability ablation, but its conservative twin-critic target appears
to suppress part of the sparse positive correction signal in this setting.

Traceable result:
[RL family comparison](../../specs/2026-07-23-rl-family-comparison/results.md).

### 3.3 Network AFR-GCN-DDPG Under Regional Abrupt Shift

This experiment combined patient-condition dynamics, clinic geography,
distance-based transfer delays, regional demand heterogeneity, and an abrupt
regional regime shift. The action space included graph-edge corrections for
reagent and idle-bioreactor transfers.

Protocol:

- three independently initialized graph policies;
- 100 paired holdout replications per training seed;
- 300 paired outcomes in total;
- parameter-matched flat residual DDPG attribution control.

Network AFR-GCN-DDPG versus MDL-2:

- relative cost difference: -1.1337%;
- mean cost difference: -$25.39 million per 52-week episode;
- paired 95% confidence interval:
  [-$29.76 million, -$20.95 million];
- paired wins: 237 of 300;
- completion-service difference: +0.001357;
- completion-service 95% confidence interval: [0.000240, 0.002458];
- patients-lost difference: -0.70;
- aggregate patient-loss noninferiority passed.

Network AFR-GCN-DDPG versus matched flat residual DDPG:

- relative cost difference: -1.1166%;
- mean cost difference: -$25.01 million;
- paired 95% confidence interval:
  [-$29.35 million, -$20.61 million];
- completion-service difference: +0.002836;
- patients-lost difference: -9.86;
- patients-lost 95% confidence interval: [-15.24, -4.31].

This is the strongest current evidence that graph representation creates
deployable value when the policy directly controls geographically structured
edge actions.

The limitation is important: this checkpoint derived most of its value from
advantage-filtered teacher distillation. Earlier offline DDPG and TD3
fine-tuning attempts were unstable. The result therefore supports a
graph-policy advantage but does not, by itself, prove that online
actor-critic learning generated the improvement.

Traceable result:
[regional network AFR results](../../specs/2026-07-25-regional-regime-network-afr/results.md).

### 3.4 Conservative Online AFR-GCN-TD3 Screen

The RTX 4090 screen trained matched graph and flat AFR-TD3 residual policies
for 100 online episodes across seeds 0, 1, and 2.

Each run completed:

- 5,200 update calls;
- 5,073 actual critic updates;
- 2,286 actor updates;
- finite optimization diagnostics;
- nonzero actor drift;
- no Traceback, NaN, OOM, or CPU fallback.

Final AFR-GCN-TD3 versus MDL-2:

- seed 0 relative cost difference: -0.051936%;
- seed 1 relative cost difference: -0.139249%;
- seed 2 relative cost difference: -0.119193%;
- pooled mean cost difference: -$2.544 million;
- pooled 95% confidence interval:
  [-$3.872 million, -$1.131 million];
- clinical noninferiority passed for all three seeds.

Matched AFR-Flat-TD3 versus MDL-2:

- seed 0 relative cost difference: +0.049693%;
- seed 1 relative cost difference: +0.040470%;
- seed 2 relative cost difference: +0.097624%;
- clinical noninferiority failed for all three seeds, primarily because of
  increased patient loss.

Final AFR-GCN-TD3 versus matched AFR-Flat-TD3:

- relative cost difference: -0.165951%;
- mean cost difference: -$4.083 million;
- pooled 95% confidence interval:
  [-$5.633 million, -$2.436 million];
- completion-service difference: +0.001203;
- patients-lost difference: -7.05.

The final graph controller therefore significantly outperformed MDL-2 and
matched AFR-Flat-TD3 in the regional-drift screen.

The final-versus-frozen comparison was less conclusive:

- pooled GCN final-minus-frozen cost difference: -$0.350 million;
- 95% confidence interval: [-$1.032 million, $0.155 million];
- GCN-minus-flat difference-in-differences: -$0.485 million;
- 95% confidence interval: [-$1.489 million, $0.173 million].

Both point estimates favor online updates, but both intervals include zero.
The graph-aware final controller advantage is established within this screen;
the independent incremental contribution of online TD3 updates is not yet
established.

## 4. Completed Five-Scenario Zero-Shot Evaluation

This evaluation reused the six final and six frozen-pretrain AFR-TD3
checkpoints without retraining, checkpoint reselection, online updates, or
deployment retuning. It tested nominal demand history, severe global demand
drift, regional demand drift, abrupt regional regime shift, and compound
regional stress.

The protocol used three graph and three parameter-matched flat training seeds,
100 paired holdout replications per scenario and seed, a 52-week horizon, and
holdout seed `40900000`. All 6,000 cross-stage common-random-number keys and
scientific anchor outcomes matched. All values were finite; checkpoint hashes
were unchanged; and no retraining, online updates, Traceback, NaN, OOM, CPU
fallback, or risk-accounting recovery occurred.

Cost differences below are candidate minus baseline in millions of dollars;
negative values favor the candidate.

| Comparison | Pooled mean difference | Paired 95% CI | Conclusion |
|---|---:|---:|---|
| Final AFR-GCN-TD3 vs MDL-2 | +$2.562M | [+$0.360M, +$4.823M] | Significantly worse pooled |
| Final AFR-Flat-TD3 vs MDL-2 | +$1.382M | [+$1.001M, +$1.805M] | Significantly worse pooled |
| Final AFR-GCN-TD3 vs AFR-Flat-TD3 | +$1.180M | [-$0.729M, +$3.490M] | No pooled graph advantage |
| GCN final vs frozen pretrain | -$0.024M | [-$0.182M, +$0.150M] | No online-TD3 increment |
| GCN-minus-flat final/frozen difference-in-differences | +$0.017M | [-$0.166M, +$0.211M] | No graph-specific online increment |

Final AFR-GCN-TD3 minus MDL-2 by scenario was:

- nominal: +$2.400M, 95% CI [+$0.020M, +$4.975M];
- severe global drift: +$5.275M, 95% CI
  [+$3.866M, +$6.348M];
- regional drift: -$0.508M, 95% CI [-$1.440M, +$0.538M];
- abrupt shift: +$1.300M, 95% CI [-$1.441M, +$3.549M];
- compound stress: +$4.344M, 95% CI [+$0.401M, +$9.385M].

Clinical noninferiority for final AFR-GCN-TD3 versus MDL-2 failed pooled and
passed only in the regional scenario. Final AFR-Flat-TD3 failed pooled and in
all five scenarios. Final and frozen graph policies used residual corrections
on 69.85% and 69.27% of decisions, respectively, and their correction
magnitudes were nearly identical. The preregistered online-TD3 attribution
classification is therefore **Not established**.

This result does not invalidate the matched regional result. It shows that the
current checkpoints are regional specialists rather than generalists. Because
they were trained on a narrow regional distribution, the five-scenario
evaluation should be reported as an out-of-distribution stress diagnostic, not
as the final generalization test of a multi-scenario policy.

DDPG has also received a smaller external robustness screen. The
abrupt-shift Network AFR-GCN-DDPG checkpoint increased cost by 0.3798% under
gradual regional drift and worsened completion and patient loss. Under compound
stress, its mean cost decreased by 0.1228%, but the interval crossed zero and
clinical outcomes worsened. DDPG therefore has not demonstrated cross-regime
zero-shot robustness either. This earlier screen covered only two external
scenarios and did not include a matched final-versus-frozen DDPG attribution
test.

The verified archive for the TD3 zero-shot campaign has SHA256
`75f5f42ea8c130ca79f19f138aa9b662317969a4e3ea3209219dbef41119d7b9`.

## 5. Experiment Coverage Matrix

| Experimental question | DDPG status | TD3 status | Remaining requirement |
|---|---|---|---|
| Pure graph RL from scratch | Screened with GCN-DDPG; materially worse than MDL-2 | Screened with GCN-TD3; stronger than pure DDPG but still worse than MDL-2 | Retain as secondary ablations, not flagship evidence |
| Matched AFR replenishment residual | Completed: 300 episodes x 5 seeds; significantly better than MDL-2 | Completed: 300 episodes x 5 seeds; CI versus MDL-2 crossed zero | Complete |
| Graph versus matched flat, replenishment only | Completed; graph-flat CI crossed zero | Not the main attribution test | No graph claim from replenishment-only actions |
| Graph versus matched flat, network edge actions | Completed: 3 seeds x 100 holdouts; graph significantly better | Completed in a separate 100-episode regional online screen; graph significantly better | Repeat under one locked multi-scenario protocol |
| External zero-shot robustness | Limited two-scenario screen; failed | Full five-scenario final/frozen screen; failed pooled | Retrain a generalist before the definitive holdout |
| Online RL increment over frozen pretraining | **Not formally tested with matched multi-seed inference** | Tested in matched and zero-shot screens; not established | DDPG final-vs-frozen is mandatory |
| Multi-scenario/domain-randomized training | Diagnostic attempts only; no accepted final policy | Not completed | Mandatory next campaign |
| Final paper confirmation | Not completed | Not completed | 5 seeds x 500 paired replications after progression gates |

Pure GCN-SAC and GCN-PPO were also screened from scratch at the common
progression budget and remained substantially worse than MDL-2. They should
remain secondary family baselines. Full AFR-SAC and AFR-PPO are not required
unless the paper claims that AFR is a backbone-agnostic contribution. The
primary fair backbone comparison is AFR-GCN-DDPG versus AFR-GCN-TD3 because
both are deterministic continuous-control methods and can share the same
anchor, residual action, teacher data, safeguards, and evaluation protocol.

## 6. Supported And Unsupported Claims

Supported by current evidence:

- AFR-GCN-DDPG can improve on a misspecified MDL-2 policy under matched
  demand-prior drift.
- DDPG is the stronger canonical backbone in the matched five-seed
  replenishment-residual comparison.
- Graph representation provides significant value when geographically
  structured edge-level transfers are part of the action space.
- Final AFR-GCN-TD3 can outperform MDL-2 and matched AFR-Flat-TD3 in its
  matched regional training regime.

Not supported:

- Reinforcement learning outperforms heuristics in every scenario.
- The complete graph-controller advantage is caused by online actor-critic
  updates.
- Current DDPG or TD3 checkpoints generalize robustly across unseen regimes.
- The current three-seed network results are sufficient for a final
  five-seed, 500-replication manuscript claim.

## 7. Required Next Experiments

### 7.1 Mandatory core experiment

1. Train one multi-scenario Network AFR policy over randomized demand
   magnitude, shift timing, regional shock center, disruption severity, and
   patient deterioration.
2. Run the exact same AFR protocol for DDPG and TD3: identical graph/flat
   observations, teacher data, residual action space, safeguards, online
   episodes, checkpoint rules, and evaluation streams.
3. For DDPG, evaluate five locked arms with identical common random numbers:
   MDL-2, frozen AFR-GCN pretrain, final AFR-GCN-DDPG, frozen AFR-Flat
   pretrain, and final AFR-Flat-DDPG.
4. Estimate the online DDPG increment
   `J(final DDPG) - J(frozen pretrain)` and the graph-minus-flat
   difference-in-differences with a hierarchical bootstrap over training seeds
   and paired replications.
5. Separate evaluation into in-distribution trajectories, interpolation over
   unseen factor combinations, structural zero-shot transfer to unseen shock
   locations, and extrapolation stress. The last category is a diagnostic, not
   the primary success criterion.

### 7.2 Progression and final gates

- Start with 100 online episodes x 3 seeds x 100 paired replications.
- Advance only if graph beats MDL-2 and matched flat, final beats frozen
  pretraining, clinical noninferiority passes, and at least two of three seeds
  deploy nonzero residuals.
- Then run 300-500 episodes x 5 seeds x 500 paired replications.
- Include fixed-prior MDL-2, rolling/forecast-aware MDL-2, robust/tuned MDL-2,
  oracle MDL-2 as a nondeployable upper bound, ISO, MYO, pMYO, pure GCN-RL,
  and matched flat AFR controls.

## 8. Decisions Requested From The Team

1. Confirm AFR-GCN-DDPG as the proposed method and AFR-GCN-TD3 as the principal
   matched backbone ablation.
2. Confirm that pure GCN-TD3 and AFR-GCN-TD3 must use distinct labels
   throughout the manuscript and result tables.
3. Approve multi-scenario training before any further zero-shot claim.
4. Treat the DDPG final-versus-frozen attribution experiment as mandatory
   before claiming that online DDPG learning generated the observed advantage.
