# Routing-Primary DDPG Publication Evidence

This directory freezes the compact evidence used for the routing-primary
publication tables and figures. The formal files use five training seeds
(`10`--`14`), four routing scenarios, 100 paired formal-holdout replications
per seed and scenario, and formal holdout seed `91100000`.

## Locked result

- AFR-GCN-DDPG minus routing MDL-2: `-17,689,540` cost units
  (`-0.658474%`), with two-level paired 95% interval
  `[-19,361,227, -15,725,338]`.
- AFR-GCN-DDPG minus matched AFR-Flat-DDPG: `-8,603,865` cost
  units (`-0.321357%`), interval
  `[-11,064,605, -6,427,714]`.
- Formal final GCN minus frozen-pretrain GCN is a `+72,903` cost-unit
  point estimate (`+0.002732%`); the separate paired development checkpoint
  curve also crosses zero. Online DDPG gain is not established.
- Timing sensitivity is asymmetric: the GCN advantage reverses under specimen
  availability lead 0 and strengthens under finished-product return lead 1.

`formal/cost_component_summary.json` is regenerated from the immutable
row-level formal outputs by
`evaluation/build_patient_indexed_specimen_routing_publication_artifacts.py`.
Its audit records every source-file hash and verifies row counts, CRN keys,
scenario coverage, finiteness, and objective reconciliation.

Important accounting note: `base_cost` is additive to `total_cost`. Its
operating subcomponents include `specimen_transfer_cost`, so the transfer
subcomponent must not be added to `base_cost` a second time.

## Compact source hashes

| File | SHA256 |
| --- | --- |
| `formal/final_summary.json` | `7f762fd1ab16f95e908b56c3925a0f6bed2bf68f12fa6fd609c13dbb18b6f3a4` |
| `formal/pretrain_summary.json` | `eb8f6560bf4f060eb7609152c69073b60bfadc90f6cf18187079cd0a3d599856` |
| `sensitivity/lead0_summary.json` | `890a18ccf237053df2e47ac412a33d059c542097a02fd31eabf5449ba10416c9` |
| `sensitivity/return1_summary.json` | `ba3a7adc568c996ca0fbe48536871c9b09118a73dd6244c8b38ed738c71f6312` |
| `development/checkpoint_curve_summary.json` | `0016e3bb8fa779984de1a21015bcade6f377d72a231b7581adc14ffc5a652ecf` |

The formal result is the publication primary evidence. The checkpoint curve,
Howard's TD3 screen, and DDPG mechanism screens are development evidence and
must remain labeled as such.
