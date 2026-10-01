# Actor-only invented-task positive control

## Authorization and scope

Zhaowei's 2026-10-01 "continue" accepts the immediately preceding explicit
proposal: one fixed actor-only design, nine artificial fits, at most 1,152
optimizer calls and 30 numerical minutes. This is not permission for a patient
pilot, patient-data fitting, reward tuning, a search, remote operations or Stage
E reopening. No Howard approval is asserted. Existing P2 and failed calibration
evidence remain immutable. Only local commits and the previously authorized
versioned Dropbox-local preservation are included.

## Question and design

Can a minimal state-conditioned categorical actor learn a known action ranking
when the permanent reference prior and learned-critic interference are absent?
The closed calibration packet failed all nine ranking gates. This new positive
control changes head, prior and optimization jointly; it is not a causal
single-component ablation and must not be used to identify which change helped.

Reuse the preceding packet's invented two-node observations, six unique request
classes, exact conditional Q table and train/test time coordinates. The global
cue is deliberately informative; the negative cue needs reference request 0,
the positive cue needs request 0.75. Q = -0.25 - 2.75*(1-t) minus 0.25 for any
wrong action. No patient environment is instantiated or stepped. This is an
elementary known-answer interpolation check, not generalization to a new scenario.

The new unregistered actor has the existing graph/self-only/flat encoder and a
single linear six-logit head over normalized public state, anchor and reference.
Rows are bound to the canonical class keys of this fixed candidate bank. A
changed bank is rejected; no dynamic-bank or clinical deployment compatibility
is claimed. Raw request/observation receipts are unchanged. Physical adjacency
uses the original normalized operator; only neural context units are scaled.

The output head starts at zero except trainable reference bias 1e-4/32. There
is no fixed log prior added during forward. This preserves the initial greedy
reference only, NOT the historical 90% sampling law (here approximately 1/6 per
class). Freeze a same-start deep copy for each fit. No value head or critic is
created. No target labels enter the forward inputs. An unrestricted linear head
replaces the shared Tanh scalar scorer; the positive-control feature representation
is explicitly different from the historical candidate policy.

## Fixed execution

Config: experiments/configs/candidate_actor_positive_control_20261001.json.
Representations graph, self_only, flat; initialization seeds 101,102,103 in that
order; CPU float32 and one Torch thread. Encoder width16, logit gain32, fixed
divisors identical to the closed calibration packet. Graph/self parameter counts
match; flat is not parameter matched. The cue bypasses graph message passing, so
this test cannot establish any graph advantage. Encoder initialization preserves
the caller's RNG. Training is deterministic full enumeration, not sampled rollout.

Use the existing checked PPO clipped surrogate with exact old-policy-weighted
advantages Q - E_old[Q], entropy coefficient0.01, clip0.2, Adam lr0.0003, gradient
norm cap0.5, refresh old probabilities every4 calls. Uniform class enumeration
weights each advantage by K*pi_old. There is no GAE, learned baseline, replay,
reward change or value loss. This isolates actor learnability under exact labels;
it does not establish ordinary PPO learnability from noisy patient trajectories.

Each of9 fits has exactly128 calls unless terminal exception/time limit ends
the single attempt. Charge each call to a durable ledger BEFORE invoking Adam.
Global cap1,152 and1800 seconds from start of numerical setup through evaluation.
Do not refund calls, retry a failure, increase budgets, alter constants after
results, select an early checkpoint or launch additional variants. Stop on
nonfinite values, integrity failure or time exhaustion; save partial state.

Save initial/final weights, Adam, Python/NumPy/Torch CPU RNG states, old probability
table, exact labels, charged budgets, next step, effective config and runtime.
All nine final models must be sealed before constructing test-time inputs.
No continuation/resume is authorized. Failures remain reportable.

## Gates and interpretation

Before each fit verify initial logits equal the reference-only 1e-4 tie-break
within1e-7 and all greedy actions are reference; copies are tensor identical.
On all eight test contexts per fit, require100% greedy accuracy and minimum
winner logit margin >=0.1. All nine fits must pass. Original frozen accuracy is
50%. Report margins, exact expected invented return, frozen/updated choices,
probabilities, gradient norms, ratio clipping and parameter counts regardless
of success. No value gate applies because there is no learned value function.

Source/config/protocol and zero-optimizer unit tests are locally committed
before the only numerical run. Tests check contracts, loss/gradient algebra,
fixed support, no labels in forward, serialization/RNG and budget accounting
without optimization. Full repository compileall and relevant regressions pass.
The runner verifies old source/document locks, P2 payload/launcher, and the
previous calibration inventory before/after. Independently recompute scalar
metrics and gates from recorded logits/Q, reconcile charges/checkpoints/hash
seals and original immutability; no extra model forward or fitting for review.

Archive new evidence with every-member hashes and a new Dropbox-local copy.
Cloud synchronization and collaborator access remain separate, unverified facts.
Passing is only an engineering positive control; it permits a recommendation,
not automatic integration, another fit or patient launch. Failing closes this
attempt; further work needs a separately bounded decision.
