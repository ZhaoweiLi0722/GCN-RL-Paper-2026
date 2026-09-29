# N4: actual collector and forward-agent engineering check

## Authority and stop boundary

Zhaowei's September 29 continuation follows the explicit N3 request for a
separate, default-off actual collector/agent integration and small engineering
verification. It authorizes this packet, not a scientific campaign. Howard's
approval is not asserted. Stage E, historical defaults, results, teachers and
checkpoints remain unchanged. No push, merge, optimizer update, performance
evaluation, scientific seed selection or formal holdout use is permitted.

Commit this protocol/config before implementation execution, and commit the
implementation before its recorded smoke. CPU float32 is explicitly selected
for these tiny mechanics fixtures; this is not a fallback from an MPS campaign.

## Fixed scope

- Use the real PatientConditionCapacityEnv with the attached three-facility
  routing fixture, unit-test seed 0 and at most four decisions per segment.
  This is an artificial software fixture, not a calibrated clinical scenario.
- Only explicitly enabled callers use the new collector. Do not register a
  new training algorithm or route historical agents through it.
- Preserve every raw base and patient-summary observation slot and time field.
  Both neural views receive the same physical links and observation-derived
  MDL-2 anchor. Hidden patient fields, environment RNG and snapshot digests
  remain collector metadata, never policy inputs.
- Reject unsupported overtime, procurement, central-hub and heterogeneous
  relation configurations. No silent feature dropping or relation collapsing.
- Reuse existing neural building blocks for small, frozen random-weight graph
  and flat actor/critic/gate forwards plus separate target copies. Use the same
  head form and declared information. Record actual parameter counts; this is
  not a parameter-matched architecture comparison or the historical agents.
- Record pre-gate proposal, gate output, submitted normalized request and actual
  execution receipts separately. Replay critic actions are submitted requests,
  not proposals and not invented inverse mappings of integer executions.
- The reward is absolute environment reward, scaled exactly once. No teacher,
  anchor-relative rollout or counterfactual labels are introduced.
- One record per actual step, full snapshot/RNG digest continuity, immutable
  records, all short n-step tails, endpoint-specific anchors and shared gamma^n
  targets. A separate cloned real environment must exactly reproduce each step.
  These hashes check current execution; they do not authenticate old artifacts.
- Distinguish collector cutoff (truncation) from an explicitly declared finite
  terminal fixture. Test both horizon mappings; neither establishes adequate
  clinical end-of-horizon accounting. Never continue stepping past env.done.

## Acceptance

Run graph/flat x physical/self-only x collector-cutoff/finite-terminal cases,
each with four decisions, and one additional environment-horizon truncation
case. Save all 36 step receipts, contracts, target values and actual parameter
counts. Checks must establish raw-observation reconstruction, equal numerical
input information, exact clone reward/observation/info/snapshot reproduction,
target-copy identity, unchanged weights, finite outputs, correct bootstrap
discounts and positive routing somewhere in this routing fixture. Zero updates
must be explicit. Include rejection tests for changed config/schema, unsupported
features, invalid actions, reset/state drift and boundary misuse. Re-run the
recorded smoke once and require byte-identical JSON. Run the existing safe
154-test suite, new focused tests, full compilation and diff checks.

Stop after the engineering readout. This does not fix historical evidence or
establish online gain, action smoothness, scenario headroom, clinical validity,
architecture parity, accelerator parity or a resumable production learner.
Scientific training still needs its own protocol, launch approval and
information-matched frozen/online/adaptive-rule/ID-MPC comparators.
