# Weekly Team Update

- Reporting period: July 20-30, 2026
- Audience: Project team

## Highlights

- Extended the patient-condition environment to a 20-clinic geographic
  network with distance-based transfer lead times.
- Added delayed reagent and idle-bioreactor transfers while preserving
  patient-specific specimen identity.
- Established AFR-GCN-DDPG as the primary method and retained graph TD3 as the
  principal actor-critic stability comparison.

## Key Results

- In the demand-prior drift study, AFR-GCN-DDPG trained for 300 episodes across
  five seeds reduced cost by 0.01636% relative to MDL-2. The paired 95%
  confidence interval was fully below zero, although the absolute improvement
  remained small.
- In the regional abrupt-shift study, Network AFR-GCN-DDPG reduced cost by
  1.1337% relative to MDL-2 and by 1.1166% relative to a parameter-matched flat
  residual policy. Completion service also improved.
- In the 100-episode, three-seed online TD3 screen, the final graph policy
  significantly outperformed MDL-2 and matched flat TD3 while satisfying the
  clinical noninferiority criteria.
- The pooled improvement from online TD3 updates over the frozen pretrained
  graph policy was directionally favorable but not statistically significant.
  Online-RL attribution therefore remains unresolved.

## In Progress

- A five-scenario zero-shot evaluation is comparing final and frozen-pretrain
  GCN/flat checkpoints under nominal demand, severe global drift, regional
  drift, abrupt regime shift, and compound regional stress.
- The campaign is evaluation-only and reuses the completed checkpoints with
  paired common random numbers.

## Next Steps

1. Complete and independently verify the five-scenario evaluation.
2. Classify the online TD3 contribution as strong, scenario-specific, or not
   established.
3. If the cross-scenario gates pass, extend the final confirmation to five
   training seeds and 500 paired replications.
4. Otherwise, return to AFR-GCN-DDPG with multi-scenario conservative
   fine-tuning instead of increasing the same training budget.

## Discussion Points

- Keep AFR-GCN-DDPG as the proposed method and TD3 as a matched backbone
  ablation?
- Present graph representation, anchored residual control, and online-RL
  attribution as three separate claims?
