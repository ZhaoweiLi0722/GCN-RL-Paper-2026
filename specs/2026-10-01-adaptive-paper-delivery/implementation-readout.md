# Dynamic candidate implementation readout

## Latest execution-binding milestone: October 1, 20:05 UTC

Entry HEAD: 871d18b57cbeb36a7912459fc3f77321cb669672. The exclusive real
execution wrapper is implemented and fixture-tested. It requires a separately
committed explicit authorization bound to the exact frozen proposal. No such
authorization exists and no scientific launcher has been invoked.

Delivered this turn:

- `dynamic_candidate_execution.py` and `run_dynamic_candidate_pilot.py`: fixed
  worktree paths, committed source/config/protocol/runtime admission, exclusive
  one-attempt claim, owned patient-backend capability, local input preservation,
  and independently reconciled terminal budget/archive receipts.
- `dynamic_candidate_watchdog.py`: separately owned child process, inherited
  monotonic origin, initial setup charged to its phase, incremental ledger
  deadlines, bounded termination/reaping and terminal failure receipts. The
  final closure deadline includes archive-byte readback and terminal writing.
- Update/continuation kernels now check the active deadline around forward,
  backward and state restoration. Durable charges cannot be refunded by an
  exception; failure payloads remain evidence, not another attempt.
- `pilot-protocol.md`: one complete prospective S1 packet, including explicit
  graph/bias/prerequisite amendments, 22,224 environment calls, 1,920 optimizer
  calls, all phase caps and a six-hour global cap. It remains unapproved.

Parent validation: **155 tests passed in 36.086 seconds** across the 14 dynamic
model, update, resource, sequence, session, continuation, factory, verification,
backend, preparation, campaign, compute-deadline, execution and watchdog modules.
The focused wrapper/deadline suite also passed 47 tests in 1.205 seconds. Full
repository `python -m compileall -q .` passed. Numerical Adam/SGD steps and real
patient calls were forbidden by test sentinels. The watchdog tests owned only
harmless exit/sleep subprocesses, which were reaped. New patient builds/steps,
research trajectories and numerical optimizer steps are all zero.

McClintock (01a0f907-dd6a-70d1-a5f2-acd594f1ac16) delivered compute-boundary
callbacks and 14 tests, then was closed. The coordinator implemented the
wrapper/watchdog, integrated all changes and ran the combined suite. Harvey's
existing finite efficiency review remains applied; no repeated reviewer gate
or historical full audit was added. No research process was observed in the
entry host PID/PPID/command scan.

Next: commit this implementation, freeze its exact proposal from the clean
commit, and present the single S1 approval. Fixture success is not scientific
qualification or patient improvement. Restoration remains in-owner only;
scientific failure is terminal, without repair-and-retry or a cold restart.

## Previous integration milestone: October 1, 19:52 UTC

Entry HEAD: dd5044f97495a3131415f6e929753bf48fcb6110. D2's complete fixture
chain is now connected; D3's real execution binding is not complete. No new
science, numerical optimizer updates or patient-environment calls occurred.

New components:

- `dynamic_candidate_factory.py`: separate-owner model construction, declared
  parameter-count check, initialization, both reference-path and own-path
  action agreement, raw paired cost/patient qualification and same-start forks.
- `dynamic_candidate_verification.py`: independent saved-scalar/request/identity
  verification and block/world paired comparisons. It does not call the learner
  or environment to rederive outcomes.
- `dynamic_candidate_campaign.py`: all 33 serial scopes, disposable preflights,
  demonstration/initialization/qualification, same-start forks, continuation,
  nine-model test barrier, raw readback and local-only archive. It accepts only
  an explicitly marked invented backend, not the patient backend.
- `dynamic_candidate_backend.py`: prospective real-factory adapter with
  admission disabled by default, static input checks, declared split starts,
  non-refundable construction counts and RNG preservation. Not invoked on real
  environments during this work.
- `dynamic_candidate_specification.py` and `dynamic_candidate_preparation.py`:
  immutable-draft bindings, explicit graph/bias choices, six consumed R4 input
  hashes, algebraic parameter counts, scoped local seed inventory and a
  clean-commit freeze interface. The thin `prepare_dynamic_candidate_pilot`
  entrypoint always rejects `--run`.

The real-file static inspection verified six consumed input hashes and layout
JSON without loading checkpoint tensors or constructing a model/environment.
Proposed counts are actor 31,346 and critic 28,721, with no shared parameters.
The prospective streams had zero numeric collisions in 499 local declaration
files. This is not a final source freeze or coverage of missing external files.
An initial preparation error classified the repository's newline-only
`.gitkeep` as an unsupported config. The fix excludes only a non-symlinked
whitespace placeholder, and a regression test still rejects content-bearing
placeholders. This was engineering preparation, not a consumed science attempt.

Parent combined run: **129 tests passed in 35.773 seconds**. After the small
placeholder fix, all seven preparation tests passed again in 0.013 seconds.
Full repository compileall passed. Command:

```text
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache <project .venv>/bin/python -m unittest tests.test_dynamic_candidate_policy tests.test_dynamic_candidate_updates tests.test_dynamic_candidate_resources tests.test_dynamic_candidate_sequence tests.test_dynamic_candidate_session tests.test_dynamic_candidate_continuation tests.test_dynamic_candidate_factory tests.test_dynamic_candidate_verification tests.test_dynamic_candidate_backend tests.test_dynamic_candidate_preparation tests.test_dynamic_candidate_campaign
```

The complete dispatcher test uses a two-transition invented environment and
metadata-only Adam doubles; numerical Adam/SGD steps are forbidden. Its injected
raw reader is explicitly a fixture, not a patient identity verifier. The real
independent verifier has separate invented raw-identity/scalar tests. Thus this
milestone verifies engineering composition, not scientific qualification,
patient performance or a complete real-run acceptance.

Russell's advancement assignment and Harvey's read-only efficiency assignment
completed and were closed. The coordinator reviewed/integrated their work.
Harvey's three recommendations are applied: reuse the existing components,
resolve graph wording within the same packet, and keep possible wider
coordination outside this immediate critical path. No new reviewer gate.

Remaining concrete D3 work: bind exclusive claim and approved-packet admission
to the real backend and an outer deadline watchdog; cover before-backward
checks and terminal completion inside its phase deadline; freeze the final
source/runtime/input/seed packet. Current checkpoint restoration is in-owner,
not cold process restart, and failures remain terminal with no retry. Do not
represent the current fixture-only runner or preparation CLI as launch-ready.

The goal clarification is now in plan/workflow and in the active `gcn-rl`
automation, with exact local prompt readback recorded in
`automation-direction-update.json`. `manuscript-claims-draft.md` reuses the
existing evidence map without another outcome audit. The broader goal is
constrained cross-facility coordination; the restricted pilot remains a
mechanism test. Original reward, scenario, numerical budget and failed 1/9
artificial gate are unchanged. One prospective S1 approval remains required.

## Earlier module milestone

Status: D1 implemented; D2 additive integration in progress. No scientific
execution, artificial fitting or patient-environment call is authorized or
performed by this engineering work. Entry commit:
49d7ae8443b18c56b676dfb4092535af6c5484b4.

## Paper question and concrete blocker removed

The next paper-facing question remains whether simulator-trained RL improves
over its own competent frozen initializer and continued imitation, under the
same information and action support. The old fixed reference preference and
joint actor/value implementation cannot implement the proposed intervention.
The new code provides a variable-support request-conditioned actor with a
trainable reference preference and a separately parameterized value model.
This is a usable implementation, not evidence that the intervention improves
patient performance. Historical P2 and the failed 1/9 artificial acceptance
result are unchanged.

## Delivered code

- `src/models/dynamic_candidate_policy.py`: new unregistered model, explicit
  initialization seeds, private construction RNG, independent actor/critic
  parameters, public-state-only critic, zero final value output, no fixed
  reference log prior or output clamp, variable/singleton canonical support,
  exact original request metadata, and initialization/parameter manifests.
- `src/rl/dynamic_candidate_rollout.py`: type-specific receipt evaluation and
  replay verification without weakening old model guards.
- `src/rl/dynamic_candidate_ppo.py`: two-owner PPO transactions, fixed behavior
  returns/advantages, once-per-rollout normalization, separate gradient clips
  and Adam moments, two debits per minibatch, complete boundary restoration,
  atomic publication and terminal failure evidence. A failed partial payload
  is preservable evidence, not a valid retry checkpoint.
- `src/rl/dynamic_candidate_imitation.py`: actor-only initialization and
  continuation, split/lineage controls, frozen critic bytes, private samplers,
  independent optimizer and terminal failure handling.
- `src/rl/dynamic_candidate_resources.py`: proposed seed allocation, exact
  phase/owner partitions, separate trajectory/clone and actor/critic charges,
  fsync-backed append-only ledger, independent ledger reader, current/global/
  aggregate phase deadlines, and restoration that cannot erase later spend.
  Optional counterfactual execution is unsupported and fails closed.
- `src/rl/dynamic_candidate_sequence.py`: all 33 declared serial budget scopes,
  final artifact seal for all nine learned models before any test scope,
  file-byte verification on restoration, and terminal failure cursor.
- `src/rl/dynamic_candidate_session.py`: public-input collection with exact
  float64 requests, reference/greedy qualification, distinct full-MDL2 actions,
  finite-horizon liabilities, deterministic RNG/receipt restoration and a
  terminal failure latch. Partial failure evidence can be saved but cannot
  be restored as a retry. No patient simulator is imported by the test fixture.
- `src/rl/dynamic_candidate_continuation.py`: complete fixed-rollout collection
  connected to the two-owner PPO or actor-only BC kernel, external durable
  charges, interrupted collection/update boundary restoration, and terminal
  failure/interrupt handling. Existing closed-rollout semantics are reused.

Old model/collector/runner registrations and all old scientific source,
configs, result roots and checkpoints remain untouched. These adapters are
opt-in library components; there is no new scientific command-line entrypoint.

## Verification boundary

Final combined parent run: 78 tests passed in 3.558 seconds (18 model, 13 update,
14 resource, 7 sequence, 20 session, 6 continuation). Command:

```text
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache <project .venv>/bin/python -m unittest tests.test_dynamic_candidate_policy tests.test_dynamic_candidate_updates tests.test_dynamic_candidate_resources tests.test_dynamic_candidate_sequence tests.test_dynamic_candidate_session tests.test_dynamic_candidate_continuation
```

Adam/SGD numerical steps were forbidden by sentinels. The update/continuation
fixtures only advance mock optimizer metadata and explicitly verify unchanged
parameter bytes. Session/continuation fixtures instantiate a purpose-built
two-transition FakeEnv with invented costs, not a patient simulator. Resource/
sequence tests use fake clocks and artifact bytes. Both complete fake rollouts
and interruption/partial-failure paths passed. The separate 33-job fixture
validates the phase cursor, not an end-to-end scientific campaign. Full repository
`python -m compileall -q .` passed with the project .venv and specified cache
prefix. No historical numerical suite or full evidence inventory was rerun.

Copernicus (01a0f8d3-52ee-7801-8d2e-61ba1f5567c5) completed the model/session
assignment and was closed. The parent accepted the failure-latch correction,
integrated all six new test modules and owns the next orchestrator task. The
previous finite efficiency review remains applied; no extra release gate or
review chain was introduced at this partial D2 boundary.

## Remaining critical path

1. Add the actual orchestrator wiring for the enumerated draft: graph-only
   disposable preflights, demonstration initialization, paired reference/own-
   path qualification, same-start forks, fixed continuation, nine-artifact
   seal, evaluation, independent raw-cost/patient reader, and local archive.
   Reuse the established scalar verifier and archive implementation.
2. Complete the effective prospective packet: exact initialization preference,
   schema/parameter counts, source/input/runtime locks, historical seed
   collision inventory, watchdog and local-only preservation destination.
   Keep the original draft explicitly non-executable and unapproved.
3. Present one consolidated approval for the bounded simulator comparison and
   prospective replacement of the failed artificial prerequisite. Do not add
   another automatic toy learning campaign or use manuscript preparation to
   delay this decision.

Steps 1-2 are authorized local engineering. New fitting/patient calls remain
unapproved. No reward/scenario changes, architecture search, remote actions,
Dropbox export, formal holdout or Stage E reopening are part of this work.
The existing same-thread `gcn-rl` schedule remains the continuation mechanism;
it is not a research process or a guarantee of uninterrupted host uptime.
