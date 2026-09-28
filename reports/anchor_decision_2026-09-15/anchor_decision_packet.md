# Anchor decision packet (fresh seeds 99.7M, 100 paired worlds per scenario)

Rule: formal paired_ci, z=1.96, margins completion 0.001 / mfg-ineligibility 0.001 / patients_lost 1.0; strict per-scenario = False in formal

## Versus executed MDL-2, pooled

| pool | arm | Δcost | wins | completion Δ (CI) | mfg-inelig Δpp (CI) | lost Δ (CI) | guardrails C/I/L |
|---|---|---:|---:|---:|---:|---:|:--:|
| pooled_all6 | mdl2_hedge_out130 | -3.00% [-3.18,-2.82] | 534/600 | +1.08pp [+0.96,+1.19] | -0.480 [-0.605,-0.354] | -45.5 [-53.4,-37.5] | ✓ ✓ ✓ |
| pooled_all6 | mdl2_hedge_out140 | -3.28% [-3.47,-3.08] | 546/600 | +1.28pp [+1.16,+1.40] | -0.653 [-0.803,-0.502] | -56.0 [-64.6,-47.4] | ✓ ✓ ✓ |
| pooled_all6 | mdl2_hedge_out150 | -3.38% [-3.59,-3.17] | 538/600 | +1.47pp [+1.35,+1.59] | -0.766 [-0.937,-0.594] | -65.2 [-74.4,-56.0] | ✓ ✓ ✓ |
| pooled_all6 | mdl_look003 | -3.65% [-3.83,-3.47] | 569/600 | +1.22pp [+1.10,+1.33] | -0.550 [-0.684,-0.417] | -50.7 [-58.8,-42.6] | ✓ ✓ ✓ |
| pooled_all6 | mdl_look004 | -3.62% [-3.88,-3.36] | 521/600 | +0.73pp [+0.58,+0.89] | -0.493 [-0.710,-0.277] | -13.0 [-24.8,-1.2] | ✓ ✓ ✓ |
| pooled_routing4 | mdl2_hedge_out130 | -3.41% [-3.69,-3.13] | 347/400 | +0.92pp [+0.76,+1.07] | +0.197 [+0.130,+0.264] | -21.2 [-30.0,-12.4] | ✓ ✗ ✓ |
| pooled_routing4 | mdl2_hedge_out140 | -3.54% [-3.83,-3.24] | 354/400 | +1.04pp [+0.88,+1.19] | +0.201 [+0.134,+0.268] | -23.2 [-31.8,-14.5] | ✓ ✗ ✓ |
| pooled_routing4 | mdl2_hedge_out150 | -3.59% [-3.90,-3.28] | 343/400 | +1.21pp [+1.05,+1.37] | +0.207 [+0.138,+0.276] | -28.3 [-37.2,-19.4] | ✓ ✗ ✓ |
| pooled_routing4 | mdl_look003 | -4.14% [-4.43,-3.85] | 372/400 | +1.03pp [+0.87,+1.18] | +0.203 [+0.132,+0.274] | -22.2 [-30.9,-13.4] | ✓ ✗ ✓ |
| pooled_routing4 | mdl_look004 | -3.22% [-3.55,-2.89] | 325/400 | +0.18pp [+0.00,+0.36] | +0.722 [+0.645,+0.800] | +43.0 [+33.1,+52.9] | ✓ ✗ ✗ |

## Versus executed MDL-2, per scenario (MDL-3 and ×1.4 only)

| scenario | arm | Δcost | completion Δpp | mfg-inelig Δpp (CI) | lost Δ (CI) | guardrails C/I/L |
|---|---|---:|---:|---:|---:|:--:|
| routing_nominal_history | mdl_look003 | -5.60% [-6.19,-5.01] | +1.58 | +0.211 [+0.091,+0.331] | -39.2 [-56.1,-22.3] | ✓ ✗ ✓ |
| routing_nominal_history | mdl2_hedge_out140 | -5.17% [-5.77,-4.57] | +1.67 | +0.155 [+0.040,+0.271] | -43.4 [-60.2,-26.5] | ✓ ✗ ✓ |
| routing_abrupt_regime_shift | mdl_look003 | -5.38% [-6.00,-4.76] | +1.36 | +0.302 [+0.192,+0.412] | -24.8 [-39.8,-9.8] | ✓ ✗ ✓ |
| routing_abrupt_regime_shift | mdl2_hedge_out140 | -5.05% [-5.67,-4.42] | +1.38 | +0.247 [+0.145,+0.349] | -27.8 [-42.8,-12.7] | ✓ ✗ ✓ |
| routing_regional_drift | mdl_look003 | -3.05% [-3.70,-2.39] | +0.27 | +0.480 [+0.340,+0.619] | +11.9 [-6.3,+30.1] | ✓ ✗ ✗ |
| routing_regional_drift | mdl2_hedge_out140 | -2.64% [-3.29,-2.00] | +0.30 | +0.460 [+0.324,+0.596] | +9.8 [-8.0,+27.5] | ✓ ✗ ✗ |
| routing_compound_regional_stress | mdl_look003 | -3.09% [-3.48,-2.72] | +0.90 | -0.180 [-0.343,-0.018] | -36.6 [-54.7,-18.5] | ✓ ✓ ✓ |
| routing_compound_regional_stress | mdl2_hedge_out140 | -2.07% [-2.48,-1.67] | +0.81 | -0.060 [-0.218,+0.097] | -31.3 [-49.3,-13.2] | ✓ ✓ ✓ |
| routing_contract_demand_drift | mdl_look003 | -3.67% [-4.06,-3.28] | +1.35 | -0.353 [-0.526,-0.179] | -46.5 [-65.9,-27.1] | ✓ ✓ ✓ |
| routing_contract_demand_drift | mdl2_hedge_out140 | -2.90% [-3.33,-2.48] | +1.35 | -0.286 [-0.456,-0.116] | -43.9 [-63.6,-24.1] | ✓ ✓ ✓ |
| routing_contract_demand_drift_severe | mdl_look003 | -2.82% [-2.98,-2.67] | +1.84 | -3.763 [-3.983,-3.543] | -169.1 [-179.3,-158.9] | ✓ ✓ ✓ |
| routing_contract_demand_drift_severe | mdl2_hedge_out140 | -3.06% [-3.23,-2.90] | +2.18 | -4.433 [-4.653,-4.212] | -199.5 [-209.8,-189.1] | ✓ ✓ ✓ |

## Head to head: MDL-3 minus ×1.4

| pool / scenario | Δcost | wins | completion Δpp | mfg-inelig Δpp | lost Δ |
|---|---:|---:|---:|---:|---:|
| all6 | -0.39% [-0.46,-0.31] | 387/600 | -0.06 | +0.102 | +5.3 |
| routing4 | -0.63% [-0.74,-0.52] | 285/400 | -0.01 | +0.003 | +1.0 |
| routing_nominal_history | -0.45% [-0.67,-0.24] | 67/100 | -0.09 | +0.056 | +4.2 |
| routing_abrupt_regime_shift | -0.34% [-0.61,-0.08] | 55/100 | -0.02 | +0.055 | +3.0 |
| routing_regional_drift | -0.41% [-0.59,-0.24] | 72/100 | -0.03 | +0.019 | +2.1 |
| routing_compound_regional_stress | -1.05% [-1.22,-0.88] | 91/100 | +0.09 | -0.120 | -5.4 |
| routing_contract_demand_drift | -0.79% [-0.93,-0.65] | 86/100 | -0.01 | -0.066 | -2.6 |
| routing_contract_demand_drift_severe | +0.25% [+0.19,+0.30] | 16/100 | -0.33 | +0.670 | +30.4 |
