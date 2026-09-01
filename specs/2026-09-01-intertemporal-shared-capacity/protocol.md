# Development Protocol and Gates

This is a prospective development protocol. It authorizes implementation,
unit tests, smoke runs, and policy-free screens. It does not authorize formal
confirmation or policy training.

## Seed separation

- mechanics smoke: `99299000` family;
- state generation: `99300000` family;
- action discovery: `99400000` family;
- fresh validation: `99500000` family;
- held-out development scenario: `99600000` family.

Seed `91100000` and every prior formal, development, or confirmation family
remain forbidden. No seed family may serve more than one role.

## J1: mechanics and continuity

Required before scientific rows are generated:

- default-off trajectories are bit-identical to the parent branch;
- a current request has zero current-epoch production effect when lead time is
  positive;
- the request changes active capacity at exactly the declared maturity epoch;
- active capacity persists and decays according to the frozen recursion;
- the shared-budget projection never exceeds its cap and is continuous around
  the boundary;
- activation and active-capacity costs are finite, additive, and convex;
- snapshot round-trip reproduces the next transition exactly.

**Gate:** all contract tests and a finite smoke rollout pass. This is an
engineering gate, not evidence of policy value.

## J2: policy-free headroom and behavioral sensitivity

Before fixed-state rows, run one episode-level **J2a pre-screen** on the fresh
discovery and validation families. Tune the static, forecast-proportional, and
graph-smoothed budget fractions on discovery episodes, then compare the frozen
choices on validation episodes across all declared schedule variants. Continue
to the fixed-state J2 screen only if at least one state-dependent allocator:

- saves at least `0.005` of tuned-static cost in the seed-clustered pooled
  estimate;
- has positive mean saving in every schedule variant;
- is clinically noninferior in aggregate;
- preserves exact action-independent RNG end states.

J2a can close an obviously saturated channel cheaply, but it cannot authorize
training or substitute for the fixed-state criteria below.

Use fixed patient states spanning at least three scheduled-wave regimes and a
prespecified library of legal network allocations. For every state, select an
allocation on discovery streams and score it on fresh validation streams with
exact paired random numbers.

Necessary criteria:

1. At least 90% of local `0.02` request perturbations change future active
   capacity or cumulative commitment cost before projection tolerance. This
   rejects integer-lot collapse at the action interface.
2. At least 30% of states have a clinically noninferior validated improvement
   of at least `0.005` of comparator cost over the strongest eligible
   comparator.
3. Prospective state-dependent value over the best tuned static allocation is
   at least `0.005` of comparator cost.
4. At least 30% of selected allocations are interior to both the local caps
   and shared pool, and at least three distinct realized allocation vectors
   (shared-budget shares rounded to `0.001`) are selected.

The primary quantity is prospective realized cost, never an in-sample oracle.

## J3: fresh-stream label stability

Repeat the same states and action library with disjoint discovery and
validation streams.

- best-allocation agreement at least `0.70` on material-headroom states;
- pairwise cost-sign agreement at least `0.80` on material pairs;
- prospective value remains positive in every scheduled-wave regime.

## J4: observable-state value and ranking

Fit only development models. Hold out complete state-generation seeds and one
schedule variant. The model receives only decision-time flat or graph state.

Required criteria:

- pairwise ranking accuracy at least `0.70` in every held-out fold;
- selected actions capture at least `0.50` of oracle-minus-static headroom;
- selected actions beat the strongest forecast-based comparator by at least
  `0.005` of comparator cost on the held-out schedule;
- no fold is worse than the strongest comparator;
- graph-versus-flat parameter counts differ by at most 1%.

Exact action top-1 is reported but is not a gate for a smooth continuous
channel. Realized held-out value is primary.

## Stop rule

One mechanics correction is allowed for an implementation defect. No
scientific threshold, cost weight, schedule, or comparator may be changed
after J2 begins. A scientific failure closes this configuration. A redesigned
configuration must receive a new spec, config name, seed families, and output
root.

## Future algorithm stage

Only a fully green J1-J4 authorizes a separate preregistration. That later
protocol must compare tensor-identical forks:

- control: frozen neutral commitment head;
- treatment: online DDPG updates to the commitment head;
- both: identical base policy, exploration support, replay, environment, RNG,
  budget, checkpoints, and evaluation CRNs.

Primary attribution is treatment minus control and final minus pre-online.
The formal holdout remains untouched until the development decision is locked.
