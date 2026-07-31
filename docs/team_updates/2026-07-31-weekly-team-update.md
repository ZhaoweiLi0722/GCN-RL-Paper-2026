# Weekly Team Update

- Reporting period: July 20-31, 2026
- Audience: Project team

## Highlights

- Extended the patient-condition environment to a 20-clinic geographic
  network with distance-based transfer lead times.
- Added delayed reagent and idle-bioreactor transfers while preserving
  patient-specific specimen identity.
- Standardized the method family terminology: AFR can use either DDPG or TD3.
  AFR-GCN-DDPG is the proposed method; AFR-GCN-TD3 is the principal matched
  backbone ablation; pure GCN-TD3 does not use AFR.

## Key Results

- In the demand-prior drift study, AFR-GCN-DDPG trained for 300 episodes across
  five seeds reduced cost by 0.01636% relative to MDL-2. The paired 95%
  confidence interval was fully below zero, although the absolute improvement
  remained small.
- In the regional abrupt-shift study, Network AFR-GCN-DDPG reduced cost by
  1.1337% relative to MDL-2 and by 1.1166% relative to a parameter-matched flat
  residual policy. Completion service also improved.
- In the 100-episode, three-seed online AFR-TD3 screen, the final graph policy
  significantly outperformed MDL-2 and matched AFR-Flat-TD3 while satisfying
  the clinical noninferiority criteria in the matched regional setting.
- The pooled improvement from online TD3 updates over the frozen pretrained
  graph policy was directionally favorable but not statistically significant.
  The five-scenario zero-shot evaluation confirmed the classification
  **Not established**.
- In that zero-shot evaluation, final AFR-GCN-TD3 was $2.562M more costly than
  MDL-2 pooled, with a 95% CI of [+$0.360M, +$4.823M]. Clinical
  noninferiority passed only in the regional scenario. The current checkpoint
  is therefore a regional specialist, not a cross-regime generalist.
- DDPG previously failed a smaller two-scenario external robustness screen and
  has not yet received a formal final-versus-frozen multi-seed attribution
  test.

## Next Steps

1. Train a multi-scenario Network AFR policy using randomized demand,
   shift timing, shock location, disruption, and patient-risk conditions.
2. Run matched AFR-GCN-DDPG and AFR-GCN-TD3 under the same training and
   evaluation protocol.
3. Add the mandatory DDPG attribution comparison: frozen AFR pretraining
   versus final online DDPG, with matched graph and flat controls.
4. Advance to five seeds and 500 paired replications only after graph,
   online-learning, and clinical gates pass.

## Discussion Points

- Keep AFR-GCN-DDPG as the proposed method and AFR-GCN-TD3 as the principal
  matched backbone ablation?
- Present graph representation, anchored residual control, and online-RL
  attribution as three separate claims?
- Treat the completed zero-shot campaign as an out-of-distribution diagnostic,
  rather than as the definitive evaluation of a generalist policy?
