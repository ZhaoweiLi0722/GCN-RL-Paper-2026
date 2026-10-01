# S1 saved-only readback

2026-10-01. Reader commit `c895fc6c476bf8dc729746a3dfa5799aeb4245d4`.
Machine-readable evidence: `readout.json`, SHA256
`ccd6186ee9bdea9a1b1baa75bedb7d5927a9debe8d99d8d53d0ed0d31bc152b0`.

## New supported answer

The original S1 failure was a resource-type contract error, not evidence of
poor initialization or failed RL. The additive reader preserves valid fractional
capacity/reagent transfers and replenishment. Patient/specimen counts remain
integer. Resource conservation permits only documented binary64 accounting
tolerance, never rounding or replacement of raw evidence.

All 39 saved episodes / 2,028 rows passed raw scalar, identity, action, role,
stream and file checks. All 606 source files matched the original preservation
manifest before and after reading. There were zero new environment calls,
optimizer updates, model forwards or checkpoint loads. Original sources,
results and the failed attempt were not altered or relaunched.

| Block | Recorded own-path reference agreement | Multi-class agreement | Six outcome differences versus R4 |
| --- | --- | --- | --- |
| 60 | 104/104 | 104/104 | All zero in each of two paired worlds |
| 61 | 104/104 | 104/104 | All zero in each of two paired worlds |
| 62 | 104/104 | 104/104 | All zero in each of two paired worlds |

The six outcomes are total cost, patient losses, completions, terminal active
patients, integrated waiting occupancy and expiry losses. The three blocks
meet the necessary own-path agreement and raw outcome conditions. These are
initialization/qualification data, not PPO results or independent confirmation.

Full qualification remains unresolved: scoring each saved initializer on the
R4-collected path was not executed. Matching observed actions/outcomes does not
substitute for the unchanged prespecified scoring procedure. No PPO/continued
BC training or final evaluation exists. The useful next step is recovering that
missing check from saved models, not recollecting demonstrations or changing the
reward. No claim of clinical noninferiority or RL benefit is made.

## Engineering verification

59 relevant tests passed in 2.078s; full repository compileall passed. Tests
include a complete invented 39-episode saved bundle, fractional resources,
integer patient rejection, preservation, legacy integer-data parity, role and
stream failure cases, and necessary conditions versus full qualification.

The first saved-only readback hit a file-size property-call bug before outcome
arithmetic. It is preserved in `engineering-failure-01.json`; a full invented
bundle regression was added and source committed before successful readback.
This was an offline tool correction, not a retry of the consumed experiment.

## Automation and next step

The prior S1 monitor had been deleted by its terminal-failure rule. One new
same-thread `gcn-rl` engineering heartbeat is ACTIVE every 30 minutes. It must
keep scientific failure separate from allowed engineering and saved-data work.
It is not a training process and depends on the desktop host being available.

Next authorized preparation: implement/test a qualification-only recovery
entrypoint, bind the four existing checkpoint inputs and current source/runtime,
and keep the old attempt closed. Its proposed scientific cap is 624 frozen
state scorings and 900 seconds, zero trajectories/optimizer updates. One user
approval question was issued. No approval or checkpoint-restoration success is
assumed. If preparation finishes before a reply, pause visibly with the exact
pending decision; do not delete the workflow or poll indefinitely.
