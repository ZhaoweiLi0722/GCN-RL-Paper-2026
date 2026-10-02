# Cohort-Completion Objective: Prospective Design

Status: design preparation only. The latest user reply, "continue", advances
the objective investigation; it is not approval of unspecified new calls, a
terminal weight, changed scenario or another training attempt. The completed
baseline route remains closed. No full-lifecycle clinical benefit is modeled.

2026-10-02 implementation refinement: [protocol.md](protocol.md) and
[bound-and-settlement.md](bound-and-settlement.md) specify an8step patient bound
and a fixed11step economic tail. The resolution-stopped formula below is retained
as an earlier design option and a secondary decomposition; the proposed primary
cost now uses the same63step exposure for every arm, including post-resolution
holding. This avoids arm-dependent early accounting termination. No new science
or lifetime terminal valuation is thereby authorized.

## Question and Estimand

Can a graph-aware policy learn a better first-52-step coordination policy when
training also accounts for the consequences for patients still active at the
cutoff? Distinguish a changed objective from a new algorithm, graph architecture
or deployment-time adaptation. Retain every old-window metric for comparison.

The recommended endpoint is **modeled resolution of a fixed enrolled cohort**,
not an invented scalar terminal penalty. Include initial identities and every
patient enrolled during the original 52 transitions, including the final batch.
After the enrollment window closes, admit no new patients and continue the
existing aging, transport, eligibility, production and infusion dynamics for
that cohort. Loss in the simulator is not necessarily clinical death.
Closing inflow changes future congestion; results apply to a closed-cohort
objective, not to an indefinitely operating service with new referrals.

## One Declared Follow-Up Rule

Recommend the same fixed full MDL-2 follow-up rule for every arm, with resource
costs and actions recorded, no learning during follow-up and no access to hidden
patient risk information beyond its original interface. This estimates the
consequences of each first-window policy under a common continuation standard,
not each learned policy's performance after step 52. The prefix remains unchanged.

The follow-up wrapper must explicitly close enrollment and its demand generator,
not merely set the ordinary horizon to a larger number while allowing more
referrals. It must define public forecast/time inputs for the follow-up rule,
settle existing pipelines, and never reset patient age, material age, identities,
resource commitments or RNG. The time feature of the learned policy is never
used after 52 under this common-follow-up design. These are future engineering
requirements, not interfaces already implemented or tested.

## Objective Without New Penalty Weights

For the same prefix trajectory, define

`C_window = sum(t=0..51, original_cost_t)`

`C_cohort = C_window + sum(t=52..resolution-1, original_incremental_cost_t)`.

Use the existing primitive cost definitions initially; do not fit their weights
to make one arm win. Report future losses, completions, waiting and resolution
time separately. Preserve the scalar scale of `1e-9` if this direction receives
numerical authorization. Extend the objective window, not the meaning of a loss.

This formula alone does **not** settle the economic boundary. Explicitly resolve
post-window procurement, already paid in-transit orders, residual inventory/
capacity, holding costs and any net salvage before freezing a numeric packet.
The same settlement must apply to every arm. Do not double-charge sunk purchases,
credit unvalidated resale value or give free disposal because active count reaches
zero. Shared operating costs are network-level costs, not automatically separable
liabilities for individual patients. Missing domain calibration remains missing.

An eventual training implementation can append the recorded future cost once
to the last prefix reward, preserving prefix action receipts and attributing
returns across the prefix. This is a changed finite-window task with an observed
continuation charge, not policy-invariant potential shaping or an oracle reward.
The extended outcome contains later observations, just as ordinary returns do;
the actor's decision-time inputs must not gain future information.

## Direct Comparison, Not Another Fit Gate

Use a prospective old-window PPO versus cohort-objective PPO comparison, plus
same-initializer frozen and BC-CONTINUE controls; R4 and full MDL-2 give context.
Both active reward arms collect equal numbers of prefix trajectories and their
standardized follow-ups, under matched starting-world information. The old arm
logs but excludes future cost from its training target; the new arm includes it.
Match optimizer and trajectory budgets; do not compare newly trained models to
selected old checkpoints. Seal every model before a genuinely fresh test set.

Main outcomes: paired cohort cost and patient resolution, plus unchanged original
window cost/patient results. Report whether a new benefit is an RL increment over
frozen and BC, an objective-specific trade-off, or inherited R4 behavior. Success
and patient safeguards must be declared prospectively, not borrowed uncritically
from the old-window 1% screen. Do not treat a later loss as a successful clearance
or count test episodes as independent training seeds.

## Stopping, Budget and Scope Boundary

Determine a conservative resolution bound from the exact locked material shelf
life, manufacturing pipeline, finished shelf life and return contract before
setting a numerical cap. Waiting/transit material ages, manufacturing advances
and finished products resolve, but this source-level reasoning must be checked
against the future closed-enrollment wrapper. A cap reached with active patients
is unresolved censoring/failed completion, never a successful zero-liability tail.
Reserve time for remaining resource-account settlement and final archival.

One complete later packet must count prefix calls, follow-up calls, preflight,
clone/restore probes, loads, optimizer updates, test episodes, verification and
archives. It must specify per-arm and global caps, fresh streams, one attempt,
no automatic retries, and preserve the old results. No numbers are authorized
by this design document. Do not add fit-only gates or baseline searches before
the direct policy comparison. The 216 already viewed test episodes may motivate
design, but cannot serve as independent confirmation or training data.

## Decision Consequences

If this direction is adopted, its claim is first-window decision learning for
closed-cohort outcomes under a specified follow-up rule. A positive result would
not establish online deployment adaptation, general joint-resource control or
clinical readiness. A null result closes this particular reward-horizon change,
not all RL. If this cohort estimand does not match the intended service problem,
keep the finite-window objective and report censoring; do not change the task
solely because RL has not improved it.
