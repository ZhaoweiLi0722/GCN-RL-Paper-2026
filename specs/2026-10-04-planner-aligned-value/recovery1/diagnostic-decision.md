# Recovery1 Diagnostic Decision

**Decision:** Planner-tail learning improves agreement with the saved public-model labels, but TD and MC are almost indistinguishable. This does not reverse the failed primary TD-versus-frozen comparison. Attach this descriptive explanation to the coordinator's readout; it is not another gate.

## Evidence And Definitions

- Read all 120 `*-diagnostics.json` files in [recovery1 training-tails](../../../results/capacity_planner_tail_20261004_recovery1/payload/training-tails/): 960 records, four roles on 240 matched roots from 120 reference worlds. Each block has 48 roots; each condition has 80. No records were excluded. A root is `(block, condition, replicate, decision_epoch)`.
- Definitions follow [original protocol](../protocol.md), [config](../../../experiments/configs/capacity_planner_tail_20261004.json), and the frozen [`_diagnostics` source, lines 184-207](../../../src/rl/capacity_planner_tail_runner.py). That source matches its original frozen SHA256 `02ab814c23ac24b2fdd612bb052a09eccba7f97b059fdfdc90cdc6b7dc8a6c29`.
- Each root compares 16 candidate sequences. Predicted score is the weighted sum of prefix cost + endpoint heuristic + learned residual; the public target is prefix cost + saved adaptive-tail cost through epoch 64. Both aggregate three response quantiles with weights `(0.25, 0.5, 0.25)`. `chosen` is the lowest predicted-score candidate.
- Agreement means the chosen candidate attains the minimum saved public target, including exact ties. There are 61 tied-best roots; using the first minimum index gives the same agreement counts here. This is candidate-sequence agreement, not verified equality of native applied actions. `in_model_regret = target[chosen] - min(target)`; lower is better. Tables use millions of the original cost units (M), not normalized percentages or realized savings.
- All roles share identical targets at each paired root. Saved choices and regrets reconcile with their JSON scores/targets. Means weight roots equally, hence blocks/conditions equally here. The 960 records are not 960 independent observations; no inferential interval or acceptance threshold is added.

## Role Summary

| Role | Agreement, n/240 (%) | Mean regret (M) | Median regret (M) |
|---|---:|---:|---:|
| existing_frozen | 83 (34.58%) | 0.720682 | 0.198143 |
| observed_td | 74 (30.83%) | 1.647056 | 0.373687 |
| planner_tail_td | 101 (42.08%) | 0.459773 | 0.048950 |
| planner_tail_mc | 100 (41.67%) | 0.463160 | 0.048950 |

## Same-Root Contrasts

Differences are candidate minus comparator; negative regret and positive agreement favor the candidate. Lower/equal/higher counts compare root-level regret. The last column counts blocks with strictly lower candidate mean regret, not statistical confirmation.

| Candidate - comparator | Mean regret difference (M) | Agreement difference (pp) | Lower/equal/higher roots | Lower-mean blocks |
|---|---:|---:|---:|---:|
| observed_td - existing_frozen | +0.926373 | -3.75 | 23/166/51 | 2/5 |
| planner_tail_td - existing_frozen | -0.260910 | +7.50 | 66/145/29 | 5/5 |
| planner_tail_mc - existing_frozen | -0.257523 | +7.08 | 64/147/29 | 5/5 |
| planner_tail_td - observed_td | -1.187283 | +11.25 | 97/111/32 | 5/5 |
| planner_tail_mc - observed_td | -1.183896 | +10.83 | 96/111/33 | 5/5 |
| planner_tail_td - planner_tail_mc | -0.003387 | +0.42 | 6/230/4 | 4/5 |

Block detail: each cell is **agreement count/48; mean regret (M)**.

| Block | existing_frozen | observed_td | planner_tail_td | planner_tail_mc |
|---|---:|---:|---:|---:|
| 0 | 11; 0.850945 | 8; 1.809838 | 15; 0.624049 | 14; 0.609093 |
| 1 | 18; 0.673086 | 18; 0.827614 | 22; 0.431007 | 22; 0.455464 |
| 2 | 21; 0.644222 | 22; 0.591801 | 25; 0.306370 | 25; 0.306373 |
| 3 | 17; 0.809275 | 7; 4.504000 | 18; 0.443171 | 18; 0.447835 |
| 4 | 16; 0.625885 | 19; 0.502027 | 21; 0.494266 | 21; 0.497033 |

Both tail learners improve agreement and mean regret versus frozen and observed TD in every block, but not every root. Observed TD's poor aggregate is strongly influenced by block 3; it actually lowers mean regret versus frozen in blocks 2 and 4. TD/MC choose the identical candidate on 230/240 roots (95.83%); TD's mean advantage reverses in block 0 and is only 0.000003 M in block 2. This is not compelling evidence for a unique TD mechanism.

Across conditions, both tail learners lower mean regret versus frozen in all three. Agreement is not uniformly better: in condition 2 it falls from frozen's 35/80 (43.75%) to 34/80 (42.50%) for either tail learner. In the primary persistent-change condition, frozen/TD/MC agreement is 23/35/33 out of 80 and mean regret is 0.768125/0.407125/0.422742 M. These remain training-root diagnostics, not primary-test confirmation.

## Interpretation And One Research Question

The labels evaluate a specified **public-model adaptive continuation**, not the learned receding-horizon controller's native counterfactual cost-to-go. They are **not native ground truth**. Better training-root ranking supports improved approximation of this finite candidate-set objective, not held-out generalization, calibrated native values, patient benefit, isolated GCN attribution, or deployment-time adaptation. MC here is direct-return model-based policy evaluation, not automatically a non-RL control. Shared-data TD/MC similarity does not establish which mechanism, if any, improves native performance.

The gap between better diagnostic rankings and the failed primary result makes continuation-policy/model mismatch a plausible explanation, not a demonstrated cause. **One next research question:** Would terminal values aligned with the deployed receding-horizon policy, instead of the fixed public adaptive continuation, translate the observed ranking improvement into native cost/service gains over `existing_frozen`? This is a research question only, not an experiment specification, permission claim, reward-weight search, or request for more epochs.

Scope: JSON-only aggregation plus protocol/config and diagnostic-definition source reads; no models loaded, forwards, simulation, optimization, new experiment, or commits. No code changed or tests rerun. Native-cost verification, manuscript and Live remain with the coordinator; nothing in this note must pass before that readout proceeds.
