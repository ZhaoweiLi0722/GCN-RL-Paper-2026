# Dynamic candidate integration review

2026-10-01. Existing source and zero-update mock tests support the request
contract below. The new artificial actor remains fixed-bank and is not ready
to control the patient environment. This is a completed interface audit, not
an integrated-model acceptance or authorization to train another architecture.

## What can be reused

`src/rl/routing_candidate_contract.py` constructs canonical classes from the
actual integer decoder request and unchanged non-specimen request groups.
Reference and anchor identity are explicit; aliases do not receive extra
probability mass. Option order cannot redefine an action class. Singleton
support is valid when all requests alias.

Chosen actions retain the original float64 submitted request, state token,
candidate hash and canonical class. Rounded class features are network inputs,
not replacement actions. Distinct requested transfers are not merged just
because both happen to execute zero in one observed state. Request equivalence
is not a certificate of equal full-policy returns or physical feasibility.

The public collector and rollout contracts retain raw observation provenance,
the selected behavior probability and its support, policy identity, RNG and
terminal/truncation semantics. No patient registry, future result or inferred
feasibility mask may be added. A payload whitelist alone cannot certify that
the caller obtained the public data at the decision time.

The specimen-only anchor candidate retains other groups from the reference.
It is not necessarily the complete MDL-2 action. The future experiment must
keep full MDL-2 as a separate controller, not rename the restricted candidate.

## Required new policy interface

Use one shared scalar scorer conditioned on public state and each current
request class, with explicit reference/anchor indicators. Variable class count
must not alter the meaning of shared output parameters. Any reference preference
used for initialization must be learnable, not an insurmountable constant.
An independent value model must not share optimizer-owned storage with the
actor. Its parameter count and information access must be disclosed.

This is a design recommendation, not an already implemented fitted model.
The historical shared candidate scorer supplies useful contract patterns, but
its numerical acceptance cannot be inherited by a new head. Likewise, the
successful fixed-bank artificial actor rejects bank changes by design and
cannot be wired to dynamic classes by padding or slot reinterpretation.

Before patient use the actual new model requires zero-update tests of request
permutation/alias invariance, variable cardinality, singleton behavior, exact
submission precision, arbitrary original request values, gradient isolation,
old-probability support, frozen-copy equality and full restore. It also needs a
prospective, separately budgeted initialization/qualification procedure. A
near-uniform toy actor is not a well-trained frozen-policy comparator.

## Verified tests and limits

The no-fitting regression suite includes:

- `tests.test_routing_candidate_contract`: aliases, singleton classes, exact
  half-grid/ULP boundaries, preserved original requests and rejection of hidden
  metadata or changes to other action groups.
- `tests.test_candidate_collection_boundary`: public-input parity, full MDL-2
  distinction, original action submission, reward/total/component reconciliation,
  terminal obligations and zero routing without false action-effect claims.
- `tests.test_candidate_policy_rollout`: shared legacy scorer permutation,
  old-support likelihoods, copied receipts, exact RNG restoration, terminal GAE
  and separation of different episodes. These are not tests of the new scorer.
- `tests.test_actor_positive_control` and `tests.test_sampled_return_control`:
  fixed-bank isolation, independent gradients, sampled scalar returns and
  recoverable artificial payloads, with real optimizer calls forbidden.

No patient rollout or patient-model optimizer update was run. No new dynamic
actor has been qualified. A2 source/mock review is complete; integrated patient
readiness remains blocked on the actual new policy and a separate protocol.
