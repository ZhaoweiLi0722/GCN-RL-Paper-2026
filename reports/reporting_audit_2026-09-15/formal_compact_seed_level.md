# Crossed-design bootstrap audit (compact_seed_level_only)

- final_summary: `experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal/final_summary.json`
- pretrain_summary: `experiments/evidence/patient_indexed_specimen_routing_primary_ddpg/formal/pretrain_summary.json`
- Seed-cluster bootstrap on per-seed, per-scenario holdout cell means from the compact publication summary. World-level and two-way intervals require the row-level formal root and are not computed here.
- resamples: 20000, alpha: 0.05

## CRN sharing checks

- flat_residual_mdl2_network_ddpg_afd: anchor cell-mean max spread across seeds: `0`
- gcn_residual_mdl2_network_ddpg_afd: anchor cell-mean max spread across seeds: `0`

## flat_residual_mdl2_network_ddpg_afd_vs_anchor

Mean difference: `-9,085,675.2` (-0.338205%)
Per-seed means: `-6,440,093`, `-9,578,771`, `-11,387,025`, `-9,222,517`, `-8,799,970`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row_published | -10,587,379 | -7,509,749 | 3,077,630 | yes |
| seed_cluster | -10,436,712 | -7,553,063 | 2,883,650 | yes |
| seed_t_interval | -11,615,196 | -6,556,154 | 5,059,042 | yes |

## gcn_residual_mdl2_network_ddpg_afd_vs_anchor

Mean difference: `-17,689,540.2` (-0.658474%)
Per-seed means: `-19,307,776`, `-19,321,570`, `-18,599,793`, `-16,858,659`, `-14,359,904`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row_published | -19,361,227 | -15,725,338 | 3,635,889 | yes |
| seed_cluster | -19,171,697 | -15,851,988 | 3,319,709 | yes |
| seed_t_interval | -20,698,812 | -14,680,269 | 6,018,543 | yes |

## gcn_residual_mdl2_network_ddpg_afd_vs_flat_residual_mdl2_network_ddpg_afd

Mean difference: `-8,603,865.0` (-0.321357%)
Per-seed means: `-12,867,682`, `-9,742,799`, `-7,212,768`, `-7,636,142`, `-5,559,933`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row_published | -11,064,605 | -6,427,714 | 4,636,891 | yes |
| seed_cluster | -10,992,752 | -6,636,309 | 4,356,443 | yes |
| seed_t_interval | -12,604,492 | -4,603,238 | 8,001,255 | yes |

## flat_residual_mdl2_network_ddpg_afd_final_vs_frozen

Mean difference: `-139,378.2` (-0.005206%)
Per-seed means: `262,634`, `-599,979`, `-161,926`, `-446,746`, `249,125`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| seed_cluster | -451,075 | 172,318 | 623,393 | no |
| seed_t_interval | -699,485 | 420,728 | 1,120,213 | no |

## gcn_residual_mdl2_network_ddpg_afd_final_vs_frozen

Mean difference: `72,902.6` (+0.002732%)
Per-seed means: `87,362`, `121,657`, `66,352`, `176,746`, `-87,604`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| seed_cluster | -14,960 | 140,992 | 155,953 | no |
| seed_t_interval | -67,925 | 213,730 | 281,655 | no |
