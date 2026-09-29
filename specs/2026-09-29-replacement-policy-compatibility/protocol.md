# R5: strict replacement-policy compatibility

2026-09-29. Zhaowei requested continued progress after R4. This finite packet
prepares R3 with explicit **R4 replacement** baselines; it does not silently
replace the F1 actors named in the historical R3 design. Preserve that design
and every historical output unchanged. No new simulation or learning.

## Fixed acceptance

1. Pin the final R4 archive manifest and verify all 24 payload files before
   and after inspection. Compare all archived `src/` files against the current
   implementations, including the environment, features and deployment path.
2. Require the policy's metadata, complete graph specification, every tensor
   key, shape, dtype and finiteness to match the constructed agent. Validate all
   modules before loading any. No partial transfer, threshold substitution or
   silent missing gate. Cross-check saved policy tensors against full state.
3. Load only actor/gate for inference. Use the existing agent implementation
   with CPU/MPS device selection and an empty one-entry replay allocation as
   the only runtime config overrides. Constructed critics/optimizers are unused;
   no saved critic/replay/optimizer is used for fitting or continuation value.
4. Select 16 equally spaced integer indices over each saved replay, irrespective
   of action/cost outcomes. Evaluate CPU policy-file, CPU full-state-module and
   MPS policy-file routes on these same archived observations. No reset/step,
   new CRN, reward label, patient scenario, optimizer call or exploration.
5. CPU/full-state outputs must agree exactly. CPU/MPS raw actor and gate
   scores use absolute tolerance 1e-5, normalized requests 1e-6, relative 0.
   Hard gate decisions and rounded requested specimen lots must agree exactly.
   Keep raw inputs/outputs, sampled time coordinates and gate threshold margins.
   Recheck parameter tensors and hashes after inference.

Static normalized requests and requested lots are **not actual patient routing**.
Operational feasibility, matched exogenous events, actor reachability and full
continuation identity still require the separate collector acceptance. Agreement
between two saved-module routes is not an independent simulator implementation.
No CPU/MPS bitwise or historical F1 output-parity claim is made.

Commit config/protocol and source before recording. A failed parity check is
reported with evidence, not repaired by loosening tolerance or selecting states.
Preserve the failed output. This packet does not authorize scientific collection
or automatic re-execution of any campaign. No automation, push or message.

## R3 replacement-baseline proposal

Retain the existing objective, scenario, actions, continuation, 12-state sample,
8+8 draws and caps (1,152 records / 37,596 steps / 3,600 seconds). Only baseline
provenance is proposed to change to all three R4 pretrain actors. Their prior
performance is unmeasured; R4's training success does not establish competence.
No historical results may be pooled with the new baseline results.

After this compatibility check, finish the strict real-collector/analysis and
fresh-stream collision checks. Request approval for that specifically bounded
data-only pilot before any new outcome collection. Do not train a critic/actor,
change reward weights, select favorable seeds, reopen Stage E or use holdout.
