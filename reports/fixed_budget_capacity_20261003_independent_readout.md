# Fixed-Budget Capacity: Independent Raw Readout

**Conclusion: no demonstrated patient-preserving training gain over the uniform same-start policy. All three primary criteria are false.** This is an independent calculation on the same saved data, not a new experiment or independent replication.

## Coverage And Agreement

Reconstructed 108 evaluation files x64 epochs = **6,912 raw rows**, covering 36 matched worlds and **72 trained-versus-fixed/MPC pairs**. Each world has three roles with identical patient-ID sets, enrollment epochs and collection sites; all seeds match `62820000 + 1000*b + 100*c + j`. Costs use all 14 raw components, including the 16-step settlement. Unique lost/delivered IDs, cumulative counts and newly-lost increments match exactly; no patients remain unresolved.

Compared 24,918 numeric fields and 218 exact metadata/screen checks with `comparison.json`. **Maximum numeric discrepancy: 7.450580596923828e-9**; none exceeds the recorded field tolerance. Maximum per-epoch component-sum versus recorded-cost discrepancy is 4.656612873077393e-10. These are summation-roundoff differences, not substantive discrepancies.

## Paired Condition Means

Positive cost change or extra lost is worse for the trained policy. Cost units are synthetic; percentages average the 12 paired-world percentage changes, not ratios of pooled mean costs. Extra delivered is exactly minus extra lost for every pair.

| Condition | Comparator | Cost change | Paired cost change | Extra lost | Cost wins |
|---|---|---:|---:|---:|---:|
| No change | Uniform fixed | +111647.35 | +0.2462% | +0.7500 | 6/12 |
| No change | ID-MPC | +1660214.72 | +4.2187% | +4.1667 | 1/12 |
| Persistent | Uniform fixed | +692496.69 | +1.8183% | +4.1667 | 2/12 |
| Persistent | ID-MPC | +1323733.01 | +3.2592% | -0.1667 | 4/12 |
| Fast fluctuation | Uniform fixed | +579698.21 | +1.4309% | +0.4167 | 4/12 |
| Fast fluctuation | ID-MPC | -347781.11 | -0.6333% | -23.0833 | 5/12 |

Persistent absolute means (cost; lost; delivered): trained **43861585.81; 166; 131**, uniform **43169089.13; 161.8333; 135.1667**, ID-MPC **42537852.80; 166.1667; 130.8333**.

## Block Means

Vectors are blocks 0 / 1 / 2; each block averages four paired worlds. Cost change is trained minus comparator in millions of synthetic units.

| Condition | Comparator | Cost change, M by block | Extra lost by block |
|---|---|---|---|
| No change | Uniform fixed | +0.391239 / +0.308553 / -0.364850 | -0.50 / +2.75 / 0.00 |
| No change | ID-MPC | +2.423616 / +1.196545 / +1.360483 | +2.25 / +6.75 / +3.50 |
| Persistent | Uniform fixed | +1.342206 / +0.839372 / -0.104087 | +2.50 / +8.25 / +1.75 |
| Persistent | ID-MPC | +0.536023 / +2.198858 / +1.236318 | -2.00 / +7.25 / -5.75 |
| Fast fluctuation | Uniform fixed | +0.850155 / +1.007973 / -0.119033 | +0.25 / +4.75 / -3.75 |
| Fast fluctuation | ID-MPC | -0.349842 / -0.351891 / -0.341610 | -21.00 / -24.75 / -23.50 |

Persistent uniform-relative savings are not positive in all blocks; mean extra losses versus uniform are positive in every condition. The reported descriptive 95% persistent savings interval is **[-1,441,883.59, +389,180.54]**, and extra-loss interval **[-0.91875, +8.41875]**. Both cross zero. Their signs and the main screen agree; bootstrap generation was not rerun. Thus the observed unfavorable mean does not establish clinical harm, and the interval does not establish safety or equivalence.

## Action Contract And Interpretation

For trained and uniform roles, all **3,456 control steps** satisfy exact `fsum(hours) == 8`, site bounds and strict non-exceedance by both `fsum` and ordinary `sum`. Ordinary summation's maximum absolute residual is **8.881784197001252e-16**, always an undershoot; native projection changes are zero. Trained site requests range **1.6618125515855833 to 2.603108868595447**; uniform is exactly 2 per site. Both commit 384 hours per world and apply 384 across the full control-plus-tail horizon. Tail requests/commitments are zero; delayed previously committed hours are retained in applied totals.

Greedy trained-versus-uniform actions differ above 1e-6 at **575/576, 575/576 and 576/576** boundaries in no-change, persistent and fast conditions, respectively: **1,726/1,728 overall**. Mean full-episode L1 allocation differences are **7.619770, 7.966589 and 7.339026 hours**. A frequently changed boundary therefore need not represent a large redistribution. All condition/block action magnitudes are in the JSON.

Persistent ledger changes versus uniform are bioreactor shortage **+456,651.87**, patient loss **+208,333.33**, expiry **+120,000**, reagent purchases **-77,318.99**, and reagent shortage **-36,043.55**. Flexible labor changes by only **+6.86** and switching by **+134.53**, despite equal total hours. These are additive cost contributions, not evidence for a causal mechanism or incorrect reward weights.

The within-run trained-versus-uniform contrast assesses learning from the exact same-start policy in a restricted fixed-total family. ID-MPC is an external performance comparator with variable total hours and site range [0,4], not a matched action-map ablation. Fast-fluctuation mean cost and losses favor trained versus MPC, but this does not overturn the uniform-relative result. No historical action-map effect, deployment-online adaptation, isolated GCN benefit or clinical safety claim follows. Only three independently trained blocks are available.

## Evidence

Raw aggregate SHA256: `2b22881b5740d1727fd77fd0266e697105857cab0a9088f14a9a17eb51441331`.

Comparison SHA256: `1ba799b31e766b408b60a51f9594629a1d0a0b40797f5fff7ba2f89b8638eec5`. The companion JSON stores all 108 raw hashes and world totals, 72 paired differences, condition/block/component means, action checks, formulas and discrepancies. No models, neural/environment/runner code, fitting, trajectory replay, analysis executable, bootstrap or old archives were used; no source/results changes or commits were made. Runtime, locks and archive verification remain coordinator-owned.
