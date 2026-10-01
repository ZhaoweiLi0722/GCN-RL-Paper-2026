# Dynamic candidate implementation readout

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
