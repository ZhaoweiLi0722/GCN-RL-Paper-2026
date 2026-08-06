# Requirements

## Scientific Name and Scope

The mechanism is **patient-indexed, identity-preserving, pre-manufacturing
specimen routing**. It is not unrestricted specimen sharing.

One patient has one indivisible autologous specimen lot. Routing may move that
lot between qualified facilities before manufacturing. Pooling, splitting,
substitution, cross-patient consumption, in-production transfer, and finished
product reassignment are prohibited.

## Identity and Lifecycle

Every enrollment receives a stable `patient_id`; `specimen_id` is identical and
both fields are write-once. The identity record carries:

- collection/origin facility;
- current material facility, or no facility while in transit;
- manufacturing facility after production starts;
- transfer count;
- waiting, in-transit, in-production, finished-return, delivered, or lost status.

Each enrolled identity must occupy exactly one active lifecycle container or one
terminal state. Starting, routing, expiry, loss, completion, checkpointing, and
return delivery cannot create, delete, duplicate, or replace an identity.

Only `WAITING` specimens may route. The primary analysis permits at most one
direct route per patient. An identity with `transfer_count == 1` is ineligible
for all later routes, which rules out multi-hop, cycles, and ping-pong.

## Qualified Network and Integer Executor

`specimen_edges` is an explicit, nonempty list in every routing-enabled config.
An edge is treated as a qualified bidirectional relationship; no unlisted pair
may exchange patient material.

All GCN, flat, and MDL-2 actions use the same environment executor:

1. The first facility-net action block is scaled by
   `max_specimen_transfer`; negative values offer lots and positive values
   request lots.
2. Each signed facility request is rounded to an integer using half away from
   zero: `sign(x) * floor(abs(x) + 0.5)`.
3. Candidate donor/receiver pairs must have opposite requested signs and a
   configured specimen edge.
4. Pairs are processed by ascending configured edge priority, then ascending
   receiver index, then ascending donor index.
5. A donor's eligible patients are processed by lowest survival, greatest
   specimen age, earliest enrollment epoch, then lexicographic patient ID.
6. Every executed move contributes `-1` at its donor and `+1` at its receiver.
   Actual facility net flow and edge flow therefore conserve integer lots.
7. Unmatched inbound and outbound quantities are recorded separately. The
   headline blocked count is the larger unmatched side, so one failed movement
   is not double-counted.

The executor is the sole decoder of specimen movement. Learned heads and
heuristics propose facility pressure; they do not select or mutate patients.

## Timing, Aging, and Outcomes

Specimen routing has a dedicated lead time independent of reagent and
bioreactor transport.

- Primary setting: dispatch at epoch `t`, spend that epoch in transit, arrive
  before the epoch `t+1` production decision.
- Sensitivity: lead time zero, with arrival before production at epoch `t`.

While in transit, health and specimen age advance exactly once per epoch. The
specimen cannot manufacture before arrival. Material shelf-life expiry,
patient ineligibility, and transport loss are distinct terminal events.

Routing consumes no external RNG draw. A configured transport-loss event is
derived from SHA-256 over episode/scenario seed, patient ID, route epoch,
transfer count, and outcome label. Exogenous demand, health, disruption, and
training randomization streams therefore remain paired under CRN.

The optional viability function `exp(-0.15 * (age + transport_time))` is a
mechanics placeholder only. It is not clinically calibrated and cannot support
a clinical claim.

## Finished Product Return

Manufacturing may occur away from the collection site, but infusion remains
assigned to the same patient's collection/origin facility. The main experiment
uses `finished_product_return_lead_time_epochs = 0`, explicitly meaning immediate
modeled return after manufacturing. A one-epoch return-delay sensitivity is
required. The model never silently treats the manufacturing facility as the
patient's infusion site.

## Observation, Action, and Model Matching

When `include_specimen_routing_state` is enabled, each facility receives four
additional observation values:

1. specimens in transit to the facility;
2. mean survival of those in-transit patients;
3. transferred specimens currently waiting there;
4. waiting specimens still route-eligible there.

Routing and no-routing control scenarios expose the same state width, action
width, graph metadata, and specimen actor head. The no-routing control disables
execution only. This prevents architecture or parameter-count changes from
being confounded with the routing treatment.

The preregistered routing pilot uses:

- GCN hidden sizes `[256, 128, 64]` with the specimen edge head enabled;
- matched flat hidden sizes `[292, 212, 128]`;
- 563,397 GCN and 564,137 flat trainable actor-plus-critic parameters under the
  20-facility contract, a relative gap of approximately 0.13%;
- identical group scaling, centering, endpoint projection, action bounds,
  scenarios, seeds, online budget, teacher-generation protocol, and CRN holdout.

The full environment snapshot includes patient identities, every lifecycle
container, specimen and product-return transit queues, route history, inventories,
environment RNG, patient sequence, episode seed, clock, histories, arrays, and
cumulative metrics. Training state records `next_episode` and restores the
environment snapshot without advancing any stream.

## Diagnostics

Step `info`, episode CSV, evaluation rows, and provenance expose, as applicable:

- transferred patient IDs and route event records;
- origin, destination, distance, transport hours, lead epochs, and route cost;
- requested integer net, actual net, and edge flows;
- route count and blocked inbound/outbound requests;
- transport loss, transit expiry, and transit ineligibility;
- return assumption and return lead time;
- identity active/terminal counts;
- online update counts, finite losses, final actor drift from frozen pretrain,
  residual usage, and parameter counts.

`enable_specimen_routing` defaults to false. With routing state also disabled,
the environment must reproduce the frozen `ce9b627` no-routing trajectory hash.
