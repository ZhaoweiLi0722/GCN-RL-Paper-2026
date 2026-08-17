# Stage F0 Online DDPG Identifiability Results

Status date: 2026-08-17

Status: completed read-only development diagnosis. The legal-action ranking
branch passed the Stage F0 design gate. Stage F1 training remains unauthorized
until a separate paired protocol is frozen and reviewed.

## 1. Audited evidence

- Six immutable unchanged-control replay buffers: GCN and matched flat, seeds
  50-52, with 5,200 online transitions per run and 31,200 total.
- GCN legal specimen-action manifold at pretrain, episode 25, and final.
- Nine prespecified states per seed, five distinct executed actions per state,
  and horizons 1, 4, and remaining episode.
- Three paired replications per state/action/checkpoint/horizon under fresh
  diagnostic CRNs.
- 1,215 manifold rows plus one header row.
- 28 selected immutable files rehashed before and after the audit.
- Formal holdout 91100000 was not used.

## 2. Replay target result

The hypothesized large short-to-long sign reversal among self-imitation samples
was not observed.

| GCN seed | Selected samples | Selected fraction | Negative behavior-remainder fraction |
|---:|---:|---:|---:|
| 50 | 673 | 12.94% | 0.74% |
| 51 | 571 | 10.98% | 1.05% |
| 52 | 776 | 14.92% | 0.39% |

Across pooled GCN replay, 99.31% of the 2,020 selected samples retained a
positive discounted behavior-trajectory remainder. General four-step versus
remainder correlation was weak (Pearson 0.163; Spearman 0.109), but the positive
one-step plus positive four-step self-imitation filter removed nearly all
opposite-sign cases. The temporal-sign branch therefore failed its 25% conflict
threshold.

The audit exactly reconstructed every persisted n-step reward and discount
multiplier within `1e-6`. This rules out replay serialization or n-step tail
assembly as an explanation.

## 3. Legal-action critic result

The critic-ranking branch passed in all three GCN seeds. Remaining-horizon
pairwise ranking accuracy moved toward random ordering during online updates:

| Seed | Pretrain | Episode 25 | Final | Final minus pretrain |
|---:|---:|---:|---:|---:|
| 50 | 0.5750 | 0.6000 | 0.4750 | -0.1000 |
| 51 | 0.5750 | 0.5125 | 0.4625 | -0.1125 |
| 52 | 0.6500 | 0.4250 | 0.5125 | -0.1375 |

At final, critic-versus-rollout Spearman correlation was 0.101, -0.012, and
-0.021 for seeds 50-52. The critic-selected action produced a material
improvement in 0.0%, 22.2%, and 55.6% of audited states respectively. The
straight-through gradient showed the same weak ordering: final pairwise
accuracy was 0.475, 0.450, and 0.550.

This was not caused by quantization collapsing the candidate set: every audited
state retained all five distinct executed integer-lot actions.

## 4. Headroom and behavioral link

At the frozen-pretrain checkpoint, the fraction of audited states with at least
one legal action improving cost by 1 million or more was 55.6%, 55.6%, and
77.8% for seeds 50-52. This agrees with the earlier independent 156-state audit,
which validated 51 specimen opportunities. The failure is therefore not an
absence of reachable local action headroom.

All three seeds met the legal-action ranking threshold. Seed 51 also had a
locked final-minus-frozen total-cost degradation of +715,787, providing the
required link between the diagnosed mechanism and behavioral deterioration.
Seeds 50 and 52 had favorable but small point estimates (-135,708 and -271,332),
consistent with the already inconclusive pooled online effect.

## 5. Decision

The Stage F0 gate passes through the legal-action ranking branch, while the
temporal-sign branch fails. The supported mechanism is:

> Online Bellman updates remain active but do not preserve within-state ranking
> of the executed legal specimen actions; the actor then follows an
> approximately uninformative straight-through gradient.

Exactly one Stage F1 candidate may be designed: a directly paired
finite-horizon counterfactual-advantage critic that ranks executed legal
specimen actions against MDL-2. It must not change the actor, exploration,
residual scale, gate, scenario distribution, model capacity, or 100-episode
budget.

This result does not prove that online DDPG is beneficial. It identifies a
reproducible failure mechanism and justifies one prospective paired development
test. Stage F1 must still improve final versus its tensor-matched frozen
pretraining checkpoint before any online-learning claim is considered.

## 6. Reproducibility hashes

- Tracked evidence snapshot:
  `experiments/evidence/patient_indexed_specimen_routing_ddpg_online_identifiability_f0/`.
- F0 config SHA256:
  `2b7d44556cf10bb0632b6ccb26ddea8d89b42bf3c6e060f6893d4194da23248c`.
- F0 summary SHA256:
  `1ed22e07109a8fa93e0130ee527cedffd7bb832c32a42cb22b9940c4a6cc4008`.
- F0 manifold rows SHA256:
  `980b547034ba8999bb233401a1b5ab01e0db9a21aced74fda02e049c75ff5094`.
- Stage C3 inventory SHA256:
  `f07f69c82ce37bdc5f687389f80c5860c2c9df4483db190cff9bded499001834`.
