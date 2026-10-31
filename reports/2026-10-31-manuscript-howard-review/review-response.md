# Manuscript revision and request for Howard's guidance

Thanks, Howard. I have brought the results into the manuscript and separated
the claims more clearly. The useful result is not a large, general independent
GCN gain: the integrated controller improves on MDL2, learning improves H8's
cost, and the extra graph effect depends on the decision interface. The matched
bounded-message study has a clearer cost signal, but it also harms patient
service. Those adverse outcomes stay in the abstract, figures, and conclusion.

## Responses to the earlier reviews

Sources: [first review](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/pull/14#pullrequestreview-5568933844)
and [second review](https://github.com/ZhaoweiLi0722/GCN-RL-Paper-2026/pull/14#pullrequestreview-5595659234).

| Concern | Manuscript response | Remaining limitation |
| --- | --- | --- |
| Old three-block estimate was too prominent | Five independently trained pairs and 120 paired worlds lead the abstract/results; parallel Graph/H8, Self/H8, and Graph/Self block figure added | Four blocks are near ties. Block 0 dominates the graph point estimate; this does not prove a defective Self seed |
| Self is not the same as Flat | Equations and prose identify Self's linear global head and identity adjacency. Historical 72-world Graph/Flat result and patient harms remain in an appendix | No five-block Flat supplement was run; no claim of superiority to all non-graph models |
| Network information is already shared | Candidate construction, common H8, public features, raw skip, and global readout are described separately | Saturated information is a hypothesis, not an identified cause of the graph gap |
| Mechanism details needed correction | 36 facility links plus 20 hub links; two-hop reachability is not a complete direct graph or functional equivalence. Fixed flattened head is not arbitrary-size invariant | No evidence that typed physical interactions or a particular donor-recipient mechanism were learned |
| Patient harm and endpoint obligations | Losses, completions, pending service, transit expiry, block/condition tables and world-harm counts retained | Cost gain does not establish service equivalence or clinical safety |
| Planner/protocol identity | Current paired Monte Carlo residual method is distinguished from TD/CEM and DDPG; Native-H8 and structured H8 are not pooled | Historical methods cannot be counted as replications of the current controller |
| Recovery and evaluation selection | The cost-scale restore error happened before evaluation; original world identities were retained by the authorized evaluation-only recovery | Earlier development-driven recipe selection remains a limitation |
| Search count | Prior five campaign/fourteen arm count is explicitly scoped, not called exhaustive | Full-project selection count unresolved; intervals remain descriptive and unadjusted |
| Source accessibility | Selected six-pair primary fields, bounded-message comparisons, recent closeout readouts, source hashes, and standalone figure code accompany this revision | This is not a complete simulator/checkpoint release. A new reader cannot reproduce training from this package alone |
| Negative follow-ups | Precision, decision loss, auxiliary targets, conditional queries, service interventions, transfer tests, GCN/GATv2 recipient failure, and insufficient multi-resource support are retained | Completed negative screens and implementation failures remain distinct; no extra fit was run |
| Broader simulation uncertainty | Dedicated discussion separates stochastic worlds in fixed regimes, parameter uncertainty, distribution shift, and misspecification | No broad OOD or clinical generalization claim. Larger-network tests were gated off, not completed |

## Where I would value your help

Could you look especially at the contribution framing and the next research
question? I think the current evidence supports the planning-assisted learning
result and a conditional graph cost--service trade-off, rather than the original
goal of a large service-preserving graph increment. Do you see a stronger,
application-grounded role for graph processing that these tests are missing?

Your suggestion to examine RL under more simulation uncertainty seems important.
Which uncertainty dimensions would be meaningful for this manufacturing setting,
and should we prioritize frozen-policy robustness, retraining under uncertainty,
or correcting the action/target interface first? I have not started another
experiment just to widen a favorable graph gap.

This PR is a manuscript review request, not a request to merge older PRs or
approve a journal submission. Author declarations and human source verification
remain open. Only synthetic results and public literature are included; no
private correspondence, raw models, training archives, or unrelated history.
