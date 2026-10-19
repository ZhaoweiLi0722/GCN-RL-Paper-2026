# Completed Replication and Remaining Review Gaps

Updated 2026-10-19. This replaces the obsolete replication proposal with its
completed status and the outstanding questions raised in Howard's review.
Results-only update; no new numerical execution is authorized here.

## Replication Is Complete

The old Graph/Self recipe was independently retrained in five new blocks:
960 training trajectories, 30,720 updates and ten sealed final models.
A separately approved evaluation-only recovery reused all training and finals,
then completed 480 zero-update trajectories on 120 paired worlds. The original
failure root and archive were preserved. The recovery is terminal and archived;
there is no replication launch pending.

Graph/Self saves **0.4071%**, with descriptive interval **[-0.0867%, 1.0407%]**,
and has **-9.258** mean patient losses/world, with interval **[-28.298, 6.052]**.
The fixed joint rule fails. Graph/H8 saves **2.1364%**, but condition-specific
patient harms remain. The original approximately 1% Graph/Self panels reused
three old model pairs and cannot replace this smaller, uncertain retraining result.
All six replication pairs and harms are retained in
[replication-evidence.json](replication-evidence.json); see
[README.md](README.md) for all three separate panels.

## Outstanding Architecture Comparison

Howard identified a missing Flat arm in the five-block replication.
Matched Self and global Flat answer different questions:

- Graph/Self removes neural neighbor messages while retaining the common
  backbone, public inputs, global readout and H8 planning information.
- Graph/Flat compares the tested graph architecture with a globally informed
  MLP; architecture and parameterization also differ.
- Graph/H8 concerns learned correction beyond the same planner.
- Graph/MDL2 concerns overall operational performance.

The historical Graph/Flat cost estimate is **0.3264%**, with its interval
crossing zero. Self does not supersede Flat, and the latter must not be weakened
or omitted because it performs well.

Howard suggested five frozen-recipe Flat fits and evaluation on the same
120 replication worlds. That is an unexecuted suggestion, not an addition to
the closed package. It needs a separate complete bounded protocol and direct
authorization. Those 120 outcomes have already been inspected; they cannot
be called an untouched confirmation set for a subsequently selected recipe.
Completing this benchmark would address attribution, not itself improve Graph.

## Protocol and Selection History

The exact original M1 protocol is included unchanged in this PR. The saved
seed audit and direct packet/world comparisons support nonreuse of the
72 M1 worlds within the recorded predecessor lineage. M1 remains DEVELOPMENT.

The count now reported is five predecessor campaign designs and 14 configured
learned arms in the October Native lineage before Localv2. It excludes seed
replicates, recoveries and later diagnostics. Earlier project campaigns also
exist; an exhaustive project-wide selection-history count is still missing.
Do not interpret 14 as a global multiplicity adjustment or a fully resolved
manuscript disclosure.

## Later Negative Follow-Ups

The later relation, interaction, neighbor-correction, ranking, retention,
label-precision and PPO attempts occurred after old Graph selection.
They must not be counted as attempts made before selecting that recipe.

The PPO pilot was near-null against its zero-update proposal control and
worse than old Graph. More recent resource-competition and event-sampling
diagnostics did not support their fixed advancement rules. In the sampling
screen, both Graph and Self worsened; a wider gap caused by worse Self is
not Graph improvement. These observations do not demonstrate that every
possible architecture or training change must fail.

Historical local locators, not all published in this PR:
`specs/2026-10-17-gcn-direction-series/prior-attempts.md`,
`specs/2026-10-17-gcn-direction-series/terminal-readout.md`,
`reports/2026-10-17-d3-follow-up/readout.md`,
`reports/2026-10-17-d4-existing-data/readout.md`,
`specs/2026-10-17-gcn-joint-proposal/terminal-readout.json`,
`specs/2026-10-18-gcn-resource-competition/terminal-readout.json`, and
`specs/2026-10-18-gcn-state-sampling/terminal-readout.json`.

## Scope

The later Localv2 method uses native-return cost-correction learning with
Monte Carlo policy evaluation and improvement. It is not the earlier scalar
Value-TD target or the separate PPO proposer.

No panel establishes a 3% Graph-over-Self effect. Keep all original models,
comparators, unfavorable conditions and negative attempts. New numerical work,
manuscript claims, full source/data release and merging are separate decisions.
This report does not reopen closed runs or authorize fitting, model calls,
new labels, seeds, evaluation, budget expansion or changes to the original task.
