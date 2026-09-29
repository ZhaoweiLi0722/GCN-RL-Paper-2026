# N6: actual DDPG update and resume mechanics

## Authority and scope

Zhaowei requested the next step after N5 explicitly identified actual DDPG
updates and checkpoint/resume engineering. This permits bounded real optimizer
steps on invented numerical records, not environment training or performance
evaluation. No scientific scenario, teacher, old checkpoint, holdout, new
scientific seed or remote operation is used. Historical code/results and the
N4/N5 modules remain unchanged. Commit protocol/config before execution and
implementation before the recorded engineering check. Do not claim Howard's
approval or a performance gain.

Use N4 only for hash-locked feature dimensions/order, not outcomes. Use N5's
graph64/flat68 counts and the graph self-only control. Generate 12 explicitly
synthetic records with their own schema/reward IDs, gamma=1, positive reward
scale 1, and one-step windows. No patient reward relabeling. The fixture has
both terminal and collector-truncated endings. CPU float32 and unit seed 0 are
explicit, not a fallback from an MPS experiment.

## Update contract

- Clone the existing prospective forward network; keep that original API/default
  frozen. The opt-in kernel uses existing torch Adam and typed N1/N3 replay.
  No replacement training driver or algorithm registration.
- Critic: MSE between Q(current observed state, submitted request) and the
  existing detached endpoint target R + discount*Q_target(next deployed request).
  Actor: negative mean critic value of its current deployed request, after the
  critic step, with critic weights fixed for actor differentiation.
- Preserve the same bounded residual + fixed sigmoid gate + clipping numerical
  transform as N4. Explicitly differentiate the gate's proposal dependence in
  the actor loss, but never its parameters. Freeze target actor/critic/gate;
  only actor/critic targets receive Polyak updates, once per successful update.
- Adam learning rates, tau, replay size, batch size and step cap are in the
  engineering JSON. These are mechanics values, not selected hyperparameters.
  No teacher/ranking/paired auxiliary loss, regularizer, actor delay or exploration.
- Validate typed windows before insertion. Uniform sampling without replacement
  uses one owned NumPy RNG; retain exact window metadata, ring position and RNG
  state. Do not translate legacy numeric buffers by guessing provenance.
- Nonfinite loss/gradient or failed update must not leave partially advanced
  weights, optimizer moments, targets, counters or sampling RNG. Frozen mode
  refuses updates before consuming RNG. The capped kernel cannot keep training
  automatically after the declared number of successful updates.

## Checkpoint boundary

Support update-boundary kernel state only: actor/critic/gate and targets, both
Adam states, complete replay contents, cursor, owned sampling RNG, update count
and an exact semantic/model/optimizer manifest. Restore only into a compatible
initial prototype; reject changed reward/schema/gate/scales/settings or corrupt
payload before changing live state. Save new files without overwriting; load
only the tensor/primitive checkpoint format with torch weights_only=True.

The kernel has no environment, partial collector window, exploration process,
asynchronous sampler, GPU RNG, dropout or global random draws. Its checkpoint
must not be described as full live-episode resume. No fake unused noise/RNG
objects should be added merely to fit the historical checkpoint API.

## Recorded test matrix

For graph/physical, graph/self-only, flat/physical:

1. Make identical online/frozen initial forks; insert six synthetic windows.
2. Run six continuous updates, appending the remaining six windows after update
   3 to exercise replay wraparound. The frozen reference takes no update.
3. Separately run three updates, save a new checkpoint, and exit that logical
   kernel. In a fresh Python child process, restore and append the same six
   windows, then run the remaining three updates.
4. Compare every sampled index, loss, targets, final request and complete
   semantic state (including Adam moments, replay and RNG) exactly. Verify
   actor/critic really change, gate/reference stay fixed, bootstrap has no
   gradients and actor differentiation does not change critic parameters.

The recorded matrix has 36 successful kernel updates (72 Adam steps), counting
continuous and interrupted/restarted executions. Additional bounded unit tests
check gradients by finite difference, zero/partial/full Polyak endpoints, bad
replay/checkpoint rejection, nonfinite-update rollback and step caps. They are
software tests, not independent scientific replications or outcome tuning.
Report losses only as diagnostics; never call a lower fixture loss improved
manufacturing performance. Retain failures and disclose implementation fixes.

Run relevant regression tests and full compileall. Repeat the report under the
same source/runtime; require semantic state and diagnostic equality. Checkpoint
container bytes need not be portable across torch versions. Stop after the
engineering readout. N5's domain, headroom, baseline, full task/closure and
scientific launch requirements remain unresolved.
