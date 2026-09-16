# Crossed-design bootstrap audit (row_level)

- root: `results/patient_indexed_specimen_routing_stage_c_td3_development/evaluation/final`
- frozen_root: `results/patient_indexed_specimen_routing_stage_c_td3_development/evaluation/pretrain`
- resamples: 20000, alpha: 0.05

## CRN sharing checks

- gcn_residual_mdl2_network_td3_bc: anchor max spread across training seeds: `0`
- flat_residual_mdl2_network_td3_bc: anchor max spread across training seeds: `0`
- anchor max abs diff gcn_residual_mdl2_network_td3_bc vs flat_residual_mdl2_network_td3_bc: `0`

## gcn_residual_mdl2_network_td3_bc_vs_anchor

Mean difference: `-19,653,054.4` (-0.734800%)
Per-seed means: `-23,180,287`, `-18,472,624`, `-17,306,253`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -23,086,362 | -16,784,147 | 6,302,215 | yes |
| world_cluster | -21,530,955 | -17,785,365 | 3,745,590 | yes |
| seed_cluster | -23,180,287 | -17,306,253 | 5,874,034 | yes |
| two_way | -23,205,817 | -16,343,971 | 6,861,845 | yes |

Mean pairwise seed correlation of world-level differences: `0.731`; variance components (seed / world / residual): `9.2e+12` / `2.57e+14` / `9.5e+13`; analytic normal half-width `4,162,301`.

## flat_residual_mdl2_network_td3_bc_vs_anchor

Mean difference: `-8,611,323.8` (-0.321965%)
Per-seed means: `-9,465,137`, `-8,040,144`, `-8,328,690`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -9,976,487 | -7,281,285 | 2,695,202 | yes |
| world_cluster | -10,183,301 | -6,977,033 | 3,206,268 | yes |
| seed_cluster | -9,465,137 | -8,040,144 | 1,424,993 | yes |
| two_way | -10,454,045 | -6,746,331 | 3,707,714 | yes |

Mean pairwise seed correlation of world-level differences: `0.632`; variance components (seed / world / residual): `1.71e+11` / `1.36e+14` / `7.94e+13`; analytic normal half-width `1,828,710`.

## gcn_residual_mdl2_network_td3_bc_vs_flat_residual_mdl2_network_td3_bc

Mean difference: `-11,041,730.6` (-0.414168%)
Per-seed means: `-13,715,149`, `-10,432,480`, `-8,977,562`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -13,808,710 | -8,498,537 | 5,310,173 | yes |
| world_cluster | -12,930,855 | -9,140,272 | 3,790,583 | yes |
| seed_cluster | -13,715,149 | -8,977,562 | 4,737,587 | yes |
| two_way | -14,131,775 | -8,036,516 | 6,095,259 | yes |

Mean pairwise seed correlation of world-level differences: `0.503`; variance components (seed / world / residual): `4.99e+12` / `1.81e+14` / `1.79e+14`; analytic normal half-width `3,320,128`.

## gcn_residual_mdl2_network_td3_bc_final_vs_frozen

Mean difference: `21,135.5` (+0.000796%)
Per-seed means: `92,666`, `142,759`, `-172,018`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -325,162 | 352,145 | 677,307 | no |
| world_cluster | -258,214 | 312,816 | 571,029 | no |
| seed_cluster | -172,018 | 142,759 | 314,777 | no |
| two_way | -398,339 | 435,393 | 833,732 | no |

Mean pairwise seed correlation of world-level differences: `-0.034`; variance components (seed / world / residual): `0` / `0` / `1.48e+13`; analytic normal half-width `307,990`.

## flat_residual_mdl2_network_td3_bc_final_vs_frozen

Mean difference: `166,164.6` (+0.006233%)
Per-seed means: `356,169`, `-69,825`, `212,150`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -231,250 | 567,723 | 798,973 | no |
| world_cluster | -172,593 | 509,742 | 682,335 | no |
| seed_cluster | -69,825 | 356,169 | 425,994 | no |
| two_way | -329,674 | 668,956 | 998,630 | no |

Mean pairwise seed correlation of world-level differences: `0.000`; variance components (seed / world / residual): `0` / `0` / `1.9e+13`; analytic normal half-width `349,037`.
