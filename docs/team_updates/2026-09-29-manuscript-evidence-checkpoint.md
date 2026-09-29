# Manuscript evidence and next-decision checkpoint

P1 local handoff, September 29. **Share with caveats for coauthor review; not
submission-ready.** Routine M1/M2/G1/E1/C1/P1 packets are complete within their
stated boundaries. This does not mean methodology findings are fixed, domain
calibration is complete, or positive online RL has been found. No remote push,
PR, main merge, message sending or new scientific campaign is part of this work.

## Supported paper spine

The patient-indexed simulator and complete pretrained graph-aware residual
controller support an empirical systems study. The formal package comparison
remains favorable under crossed cost-inference sensitivity. Online endpoint
gains and isolated message-passing attribution remain unestablished. Describe
the demonstrated package, not a cleaner algorithm that was never run. A new
scenario alone cannot repair these attribution gaps.

## Claim-to-evidence matrix

| Claim | Evidence and allowed wording | Qualification / remaining work |
| --- | --- | --- |
| Formal controller beats MDL-2 in pooled cost | [Recovered rows/crossed audit](../../specs/2026-09-29-formal-crossed-audit/readout.md): -17.689540 million units, -0.658474%; crossed 95% interval [-19.805786, -15.354809] million | Modeled objective units, not dollars. Post-hoc sensitivity, not a new holdout test. Not every timing condition |
| Graph-aware package beats implemented flat package | Same audit: -8.603865 million units, -0.321357%; crossed interval [-11.284344, -6.087716] million | [G1](../../specs/2026-09-29-formal-graph-contract/readout.md): 613,286 vs 607,338 parameters, but differing features/heads/gates. Not isolated GCN value |
| Online DDPG adds endpoint value | Formal GCN final-minus-frozen +0.072903 million; crossed interval [-0.180322, +0.313172] million | Benefit unestablished, not equivalence or a conclusion about all DDPG implementations |
| Executed method is described reproducibly | [M1](../../specs/2026-09-29-formal-method-contract/readout.md): pinned code/configs, specimen-only residual, request replay, four-step relative online returns and fixed deployment | Description does not validate the reward mixture or its equivalence to evaluation cost |
| The null has a known single cause | [M2](../../specs/2026-09-29-formal-replay-contract/readout.md): 153/314 discontinuous cached windows, reward mixing, calibration-discount mismatch and wider teacher support | Demonstrated training-path limitations, not a causal explanation or guaranteed repair; cache-generation horizon remains unverified |
| Howard's planning/TD3 closes headroom | [Integration qualifications](2026-09-28-integrated-evidence-and-next-study.md) and [planner report](../../specs/2026-09-16-measurement-a-rollout-planner/results.md) preserve promising development findings | Latest planner rows/logs absent from merged tracked tree. Patient-level planner versus aggregate neural information is not matched merely by resampling latent health. No pooled cross-study effect or optimum fraction |
| Feedback helps after persistent change | [Queue evidence](../../specs/2026-09-29-service-queue-boundary/readout.md): finite synthetic feedback value; ordinary controls attain restricted full-progress optima | Known-law diagnostic, five decisions and declared closure. Not patient performance or online weight-update evidence |
| Hiding progress makes RL necessary | [C1](../../specs/2026-09-29-completion-count-comparator/readout.md): fixed event-count rule attains changed/nonbinding completion-view bounds | Coupled-queue gap is relative to this simple rule; censored-data ID-MPC and frozen history-policy comparisons remain absent |
| Staffing scenario is ready | [E1](../../specs/2026-09-29-qualified-support-contract/decision_contract.md): named setup-support staffing lever and availability-time dictionary | Literature motivates the question, not calibration. Twelve domain inputs remain null. No biological acceleration or assumed continuous execution |
| Clinical/economic deployment is justified | Historical patient-facing outcomes remain in the [frozen evidence map](../../experiments/evidence/patient_indexed_specimen_routing_publication_evidence_map.json) | Cost projection does not re-audit clinical margins. Indication/process-specific justification remains necessary; no deployment or monetary-savings claim |

Negative cost differences favor the first policy. Each percentage uses its
stated comparator's mean, not a selected avoidable-cost denominator. Only five
independent formal training seeds underlie the crossed analysis; many outcome
rows do not create more trained-seed replications.

## Reproducibility and provenance

[Crossed verification](../../reports/2026-09-29-formal-crossed-audit/verification.json)
independently checks all five cost contrasts by multiplicity bootstrap. Forty
projected CSVs contain 16,000 identity/cost rows. Two original summary hashes
and ten GCN-final historical CSV hashes match. The other thirty CSVs reconcile
with summaries and have current hashes, not independent historical byte proof.
Full patient trajectories/checkpoint tensors and the latest planner archive
remain separate availability issues.

Method/replay/graph readouts link historical source commits and saved JSON
audits. Regenerating them needs historical Git objects and the external
training/cache paths documented there. Do not describe the whole project as a
self-contained one-command replication package yet.

From the persistent integration worktree, these are non-training checks:

```bash
PY='/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python'
"$PY" -m unittest \
  tests.test_formal_method_contract tests.test_formal_replay_contract \
  tests.test_formal_graph_contract tests.test_formal_attribution_recovery \
  tests.test_crossed_audit_verification \
  tests.test_service_effort_mechanics tests.test_service_effort_decisions \
  tests.test_service_queue_network tests.test_service_queue_evidence \
  tests.test_service_queue_observation_contract tests.test_service_queue_completion_rule
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "$PY" -m compileall -q .
git diff --check
```

Optional reproduction uses **fresh** destinations, never archived report paths:

```bash
"$PY" -m evaluation.verify_formal_crossed_audit \
  --root reports/2026-09-29-formal-crossed-audit \
  --output /private/tmp/crossed-handoff-check-new.json
"$PY" -m evaluation.audit_service_queue_completion_rule \
  --output /private/tmp/count-rule-handoff-check-new
```

These consume existing observations, not new holdout outcomes. A failed source
lock is a mismatch to inspect, not a reason to remove protection. Focused tests
include synthetic unit fixtures, not research training. No TeX engine is
installed; PDF compilation and visual verification remain unperformed.

## Remaining decisions

| Dependency | Needed work; not assumed approval |
| --- | --- |
| Paper claim scope | Coauthors choose complete-package versus isolated graph/online claims. The introduction's condition-blind/temporal-encoder questions need a section-level evidence map or narrower framing before submission |
| Another learning campaign | Separately approve a corrected development protocol: target-consistent replay, true adjacency, multi-step discount, support, cache provenance and tests before training. Preserve negative outcomes; no reuse of closed holdout for selection |
| Isolated graph contribution | Prospectively match information transforms and proposal-conditioned gates; distinguish shared heads from message passing; hold physical edges fixed. Parameter count is insufficient |
| Stronger planner comparison | Obtain all retained raw rows/decision logs, source/config hashes and stop/resume history; reconcile actual shared observation and query budget before comparing |
| Operational scenario | Resolve the three E1 bundles: actual task/bookable lever; timestamped event/staff records; units, costs, persistence and closure. Close the channel if no causal leverage exists |
| Online-update attribution | Competent frozen history policy, adaptive rule and censored-data ID-MPC; equal information, declared compute/reset rules and independent future change episodes. Simple-rule gaps are insufficient |
| Submission | Clinical-margin rationale, sensitivity coverage, literature/novelty scope, legacy-result labeling and typeset PDF review. Significance and passing software tests do not guarantee peer-review acceptance |

## Short coauthor update (draft, not sent)

The formal controller gains still hold under the crossed cost audit, but we
should describe them as package-level gains, not isolated GCN or online-DDPG
effects. We found replay-target and graph/flat interface issues that need a
clean prospective design before another learning study. Ordinary feedback and
planning explain the small queue examples, so I would not train DDPG on them.
For a new scenario, the concrete question is whether qualified setup-support
staff can be booked and whether task/availability records let us learn the
changing response. We can discuss current-paper scope separately from that
follow-up study.

## Completion boundary

Validation at this checkpoint: 85 focused tests, full Python compilation and
diff checks pass. Fresh independent verification of 40 projected CSV hashes and
five crossed-bootstrap contrasts exactly matches the archived verifier output.
All new displayed formal/queue numbers reconcile to saved JSON. Twenty-eight
relative links, 47 unique TeX labels, 39 citation keys and comparator source/
output hashes pass. These are scoped checks, not a new full-manuscript scientific
or typesetting review.

This finishes the finite local queue, not the research program. Remaining work
depends on archives, domain inputs, claim-scope decisions or a new approved
protocol. Stop the continuation automation at this checkpoint; do not invent
more scenarios or request the same facts hourly. Original formal results and
Howard's reports remain preserved.
