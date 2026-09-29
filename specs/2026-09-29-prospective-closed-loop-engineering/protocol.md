# N7: bounded real-collector learning and full session resume

## Authority and boundary

Following N6, Zhaowei requested continuation of the named tiny real-collector
closed-loop and trajectory-resume verification. This permits the bounded mechanics
below, not performance evaluation, a new scientific scenario or reopening Stage E.
Commit this protocol/config before real checks, and source before the recorded
matrix. Preserve N4/N5/N6 and all historical source/results/defaults. No teacher,
old checkpoint, scientific CRN, coauthor sign-off, remote action or tuning.

Use N4's hash-locked routing fixture, seed 0, CPU float32. Its cost scale 1e-5,
gamma .9 and three-step windows are inherited mechanics values, not the proposed
N5 scientific objective. No clinical episode settlement is claimed. The new
wrapper must be explicitly enabled and cannot reset or run a second episode.

## Six finite cases

Each of three fixed architecture/boundary combinations has online and frozen
modes: graph64/physical with collector cutoff at 6 of horizon 8; graph64/self-only
with environment truncation at 6; flat68 with terminal mapping at 6. These are
software coverage cases, not a factorial performance comparison. The terminal
mapping tests masks, not the adequacy of patient terminal accounting.

Each case performs six actual decision steps continuously, then a separate
four-step prefix and two-step continuation in a fresh Python process. The frozen
kernel receives the same collection/scheduling code but takes no optimizer steps.
Online/frozen trajectories may diverge; only their own continuous/restarted paths
must match. No comparison of aggregate rewards between arms is permitted.

Before gate evaluation, add existing OU exploration to the actor's specimen
residual coordinates (three dimensions), then apply the same residual scale,
clipping and fixed gate. OU mu=0, theta=.15, sigma=.05, seed=0 are fixed mechanical
settings; all non-specimen coordinates retain zero residual scale. Deterministic
target/actor-loss actions remain noise-free. Log both deterministic and behavior
requests. A no-noise check must match N4 inference.

After each step, publish each newly mature three-step window exactly once; retain
the last two records as pending. At closure flush all shorter tails exactly once.
Insert windows into the N6 typed ring (capacity4), and make at most one DDPG update
after each decision once two windows exist. Thus online updates occur at steps
4,5,6 (three total); no updates occur in frozen mode. Interrupt after step4 with
two replay windows, two pending records and one online update already completed.
At closure the six emitted lengths must be [3,3,3,3,2,1]; ring position must be2.

One matrix has 72 primary environment steps plus 72 cloned verification steps,
18 successful kernel updates / 36 Adam steps and six fresh-process continuations.
Repeat once for deterministic evidence; bounded unit tests add mechanics steps.
No adaptive seed extension, tuning or performance plot. Positive physical routing
must be exercised, but this does not require noise/updates to change integer routing.

## Session checkpoint and failures

Include complete environment snapshot (identities, queues/transits, inventory,
histories, counters, RNG), collector cursor/closed flag/lineage token, pending
records, execution receipts, OU state and RNG, and the complete N6 kernel state.
Strict tensor/primitive encoding must preserve NumPy dtypes; weights_only=True
loading and checksum/manifest validation remain mandatory. Scratch clone state
is not retained: it is overwritten from the actual pre-step snapshot before
each verification step and never supplies the decision.

Restore on a private copy; verify a canonical environment snapshot round trip,
collector/environment/trace continuity, exact pending/replay reconstruction and
update counts. Independently replay OU draws to verify its saved state. Reject
incompatible config, wrong cursor/token, missing pending/replay rows, corrupted
environment/noise state and altered reward semantics without mutating live state.
Failed env/collector/update steps must roll back the whole session, not just Adam.
Checkpoints cannot replace existing files; no pickle of arbitrary class objects.

Require exact continuous/restarted receipts, raw observations, action requests,
rewards, environment identity/RNG snapshots, sample indices, losses, all network
and optimizer tensors, replay and pending state. Exercise continuation at step4
and refusal after closure; unit tests cover other boundaries and malformed loads.
Source hashes, raw diagnostics and checkpoint inventory are retained. Semantic
payload equality is required; binary checkpoint container bytes need not match.

## Stop condition

Run focused regressions and full compilation; report any failed check and fix
without hiding its artifacts. This closes only the declared single-episode CPU
mechanics scope. It does not cover distributed/asynchronous learning, GPU
determinism, clinical calibration or a competent pretrained baseline. New-task
costs/settlement, headroom, baseline quality and scientific locks remain open.
Do not launch a performance campaign or modify the manuscript's online-null claim.
