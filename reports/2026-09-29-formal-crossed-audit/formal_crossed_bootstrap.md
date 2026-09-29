# Crossed-design bootstrap audit (row_level)

- root: `reports/2026-09-29-formal-crossed-audit/projection/formal_final`
- frozen_root: `reports/2026-09-29-formal-crossed-audit/projection/formal_pretrain`
- resamples: 20000, alpha: 0.05

## CRN sharing checks

- gcn_residual_mdl2_network_ddpg_afd: anchor max spread across training seeds: `0`
- flat_residual_mdl2_network_ddpg_afd: anchor max spread across training seeds: `0`
- anchor max abs diff gcn_residual_mdl2_network_ddpg_afd vs flat_residual_mdl2_network_ddpg_afd: `0`

## gcn_residual_mdl2_network_ddpg_afd_vs_anchor

Mean difference: `-17,689,540.2` (-0.658474%)
Per-seed means: `-19,307,776`, `-19,321,570`, `-18,599,793`, `-16,858,659`, `-14,359,904`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -19,388,283 | -15,724,865 | 3,663,418 | yes |
| world_cluster | -19,108,471 | -16,289,525 | 2,818,947 | yes |
| seed_cluster | -19,171,697 | -15,849,229 | 3,322,468 | yes |
| two_way | -19,805,786 | -15,354,809 | 4,450,977 | yes |

Mean pairwise seed correlation of world-level differences: `0.643`; variance components (seed / world / residual): `4.16e+12` / `2.21e+14` / `1.24e+14`; analytic normal half-width `2,358,068`.

## flat_residual_mdl2_network_ddpg_afd_vs_anchor

Mean difference: `-9,085,675.2` (-0.338205%)
Per-seed means: `-6,440,093`, `-9,578,771`, `-11,387,025`, `-9,222,517`, `-8,799,970`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -10,579,347 | -7,468,510 | 3,110,837 | yes |
| world_cluster | -10,238,912 | -7,938,467 | 2,300,445 | yes |
| seed_cluster | -10,436,712 | -7,553,063 | 2,883,650 | yes |
| two_way | -10,895,859 | -7,239,471 | 3,656,389 | yes |

Mean pairwise seed correlation of world-level differences: `0.614`; variance components (seed / world / residual): `2.92e+12` / `1.48e+14` / `9.54e+13`; analytic normal half-width `1,961,396`.

## gcn_residual_mdl2_network_ddpg_afd_vs_flat_residual_mdl2_network_ddpg_afd

Mean difference: `-8,603,865.0` (-0.321357%)
Per-seed means: `-12,867,682`, `-9,742,799`, `-7,212,768`, `-7,636,142`, `-5,559,933`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -11,122,136 | -6,397,598 | 4,724,538 | yes |
| world_cluster | -9,908,928 | -7,322,891 | 2,586,037 | yes |
| seed_cluster | -10,781,156 | -6,636,309 | 4,144,847 | yes |
| two_way | -11,284,344 | -6,087,716 | 5,196,628 | yes |

Mean pairwise seed correlation of world-level differences: `0.384`; variance components (seed / world / residual): `7.36e+12` / `1.36e+14` / `2.16e+14`; analytic normal half-width `2,716,335`.

## gcn_residual_mdl2_network_ddpg_afd_final_vs_frozen

Mean difference: `72,902.6` (+0.002732%)
Per-seed means: `87,362`, `121,657`, `66,352`, `176,746`, `-87,604`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -125,961 | 263,604 | 389,565 | no |
| world_cluster | -94,502 | 242,001 | 336,503 | no |
| seed_cluster | -14,960 | 140,992 | 155,953 | no |
| two_way | -180,322 | 313,172 | 493,493 | no |

Mean pairwise seed correlation of world-level differences: `-0.022`; variance components (seed / world / residual): `0` / `0` / `1.67e+13`; analytic normal half-width `178,919`.

## flat_residual_mdl2_network_ddpg_afd_final_vs_frozen

Mean difference: `-139,378.2` (-0.005206%)
Per-seed means: `262,634`, `-599,979`, `-161,926`, `-446,746`, `249,125`

| Interval | low | high | width | excludes 0 |
| --- | ---: | ---: | ---: | :--: |
| nested_seed_then_row | -498,012 | 226,523 | 724,535 | no |
| world_cluster | -355,696 | 77,527 | 433,223 | no |
| seed_cluster | -451,075 | 172,318 | 623,393 | no |
| two_way | -546,384 | 278,095 | 824,479 | no |

Mean pairwise seed correlation of world-level differences: `0.056`; variance components (seed / world / residual): `1.08e+11` / `1.11e+12` / `1.88e+13`; analytic normal half-width `360,078`.
