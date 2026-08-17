# Routing-Primary Stage E Evidence Synthesis

Synthesis date: 2026-08-17

## Publication decision

Retain AFR-GCN-DDPG as the primary method. The five-seed routing-primary formal
holdout supports two claims: the graph controller improves on routing MDL-2,
and it improves on a parameter-matched flat DDPG controller. It does not
support a separate claim that the 100 online actor-critic episodes created the
gain. Howard's matched TD3 result is useful as a development-only backbone
ablation that independently corroborates graph and anchor value, but it has the
same frozen-pretrain-versus-final attribution limit.

No Stage C, C2, or C3 candidate passed its preregistered gate. Stage D was not
triggered, the formal holdout must not be reused, and open-ended algorithm
search is closed. The active work is manuscript integration and reproducibility
freeze.

## Formal routing-primary result

Differences are candidate minus comparator; negative total-cost differences
favor the candidate. The formal protocol used training seeds 10--14, four
routing scenarios, 100 paired replications per scenario and seed, and formal
holdout seed 91100000.

| Comparison | Cost difference | Relative difference | Two-level paired 95% interval | Completion difference | Patients-lost difference | Manufacturing-ineligibility difference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| AFR-GCN-DDPG minus MDL-2 | -17.690 million | -0.658% | [-19.361, -15.725] million | +0.001806 | -25.263 | -0.004705 |
| AFR-Flat-DDPG minus MDL-2 | -9.086 million | -0.338% | [-10.587, -7.510] million | +0.000636 | -8.888 | -0.003773 |
| AFR-GCN-DDPG minus AFR-Flat-DDPG | -8.604 million | -0.321% | [-11.065, -6.428] million | +0.001169 | -16.375 | -0.000932 |

All 20 GCN seed-by-scenario cells have lower mean cost than their paired MDL-2
anchor. The graph-versus-flat total-cost, completion-service, patient-loss, and
manufacturing-ineligibility intervals are all favorable.

## Cost interpretation

MDL-2 mean total cost is 2,686.444 million modeled weighted-objective units.
The additive GCN-minus-MDL-2 decomposition is:

| Additive component | MDL-2 share | GCN minus MDL-2 |
| --- | ---: | ---: |
| Base operating cost | 45.856% | -2.100 million |
| Patient-loss cost | 44.841% | -12.631 million |
| Expiry cost | 8.966% | -2.806 million |
| Urgency cost | 0.337% | -0.152 million |
| **Total** | **100.000%** | **-17.690 million** |

Specimen-transfer cost rises by 0.174 million, but it is already a
subcomponent of base operating cost. Other operating and shortage components
fall by 2.275 million, producing the net base-cost reduction of 2.100 million.
Adding specimen-transfer cost again would double count it.

The dominant source of value is lower patient-loss cost, followed by lower
expiry and base operating costs. These are modeled objective units, not dollars.

## Online attribution

| Evidence | Final/late minus baseline | Relative difference | 95% interval | Decision |
| --- | ---: | ---: | ---: | --- |
| Formal DDPG GCN final minus frozen point estimate | +72,903 | +0.00273% | Not stored in compact top-level summary | No online gain established |
| Development DDPG GCN episode 100 minus episode 75 | +68,613 | +0.00256% | [-344,648, +501,630] | Plateaued or inconclusive |
| Development TD3 GCN final minus frozen point estimate | +21,135 | +0.00080% | Audited as crossing zero | No online gain established |
| DDPG C3 structured candidate final minus frozen | +84,712 | +0.00416% | [-563,653, +833,767] | Advancement gate failed |
| DDPG C3 structured candidate minus unchanged control | -18,204 | -0.00089% | [-278,652, +271,812] | No regression; no gain established |

C3 verified that the negative result was not simply caused by a lack of legal
actions or action coverage: 58 of 156 audited states had an independently
validated lower-cost clinically noninferior action, 51 involved specimen
correction, and candidate training produced approximately 19% structured
selections with approximately 17% behaviorally distinct actions. Even so, the
final online effect remained negligible and uncertain.

The correct paper statement is therefore that graph-aware advantage-filtered
residual control creates value relative to the matched comparators, while the
incremental contribution of online DDPG or TD3 updates is not established at
the tested budget. The evidence points to offline advantage-filtered
pretraining/distillation as the main source of the observed gain.

## Transport timing

The Stage A frozen-policy sensitivity is asymmetric:

| Assumption | GCN minus MDL-2 | GCN minus flat | Interpretation |
| --- | ---: | ---: | --- |
| Specimen availability lead 0 | +0.633% | +0.255% | Primary advantage reverses |
| Finished-product return lead 1 | -1.050% | -0.303% | Primary advantage strengthens |

The manuscript must not claim general transport-timing robustness. It should
state that epoch-level specimen availability and continuous geographic travel
time are distinct model assumptions.

## TD3 development ablation

Howard's audited three-seed TD3 recovery produced 600 paired development
outcomes per comparison. Final GCN-TD3 was 0.735% lower cost than MDL-2 and
0.414% lower than matched flat TD3; both paired intervals were wholly
favorable. Frozen-pretrain GCN-TD3 was 0.736% lower than MDL-2, and final was
0.00080% worse than frozen. This supports TD3 as a corroborating backbone
ablation only. It is not formal holdout evidence and does not replace the DDPG
primary method.

## Claim boundary

Supported:

1. AFR-GCN-DDPG improves on routing MDL-2 under the formal routing-primary protocol.
2. AFR-GCN-DDPG improves on parameter-matched AFR-Flat-DDPG under the same protocol.
3. Patient-facing metrics improve alongside total cost in the formal comparison.
4. Matched TD3 development evidence corroborates graph and anchor value.

Not supported:

1. Online actor-critic updates independently created the observed gain.
2. TD3 or C3 has formal-holdout status.
3. The controller is robust to all transport-timing assumptions.
4. Objective-unit savings are realized monetary savings.
5. Routing itself is the treatment effect; routing is admissible for all compared policies.

## Reproduction

The source-of-truth claim map is
`experiments/evidence/patient_indexed_specimen_routing_publication_evidence_map.json`.
Generated CSVs are under `reports/patient_indexed_specimen_routing/`, and the
paper figures are `routing_primary_cost_effects.{png,pdf}` and
`routing_primary_cost_components.{png,pdf}`.

Run the portable rebuild with:

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/gcnrl-matplotlib \
python3 evaluation/build_patient_indexed_specimen_routing_publication_artifacts.py
```

Refreshing `formal/cost_component_summary.json` additionally requires the
immutable row-level formal result root and the explicit
`--raw-formal-final-root` option. No training or evaluation is launched by the
builder.
