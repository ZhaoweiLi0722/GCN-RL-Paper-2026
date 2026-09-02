# Frozen Protocol: Budget-Neutral Per-Facility Residual Headroom

Status: **awaiting Howard sign-off; no scientific seed may be consumed**.

## Question and estimand

The parent J2 result closed the global-parameter action library. This screen
does not revise J2 or its threshold. It estimates the value of a one-step,
per-facility continuous reallocation around J2's strongest global comparator,
with the shared capacity total held exactly fixed.

The primary estimand is the validation-stream relative cost saving of the
candidate selected per state on an independent discovery stream, compared
with the frozen graph-forecast action. The optimistic validation oracle is a
necessary upper-bound screen and is never reported as achievable policy
performance.

## Frozen environment and comparator

- Source scenarios and environment: the exact hashed J2a scenario config.
- State behavior, first-action baseline, and continuation policy: graph
  forecast, budget fraction 1.00, graph smoothing 0.25.
- Decision epochs: 4, 8, 12, 16, 24, 32.
- Lookahead: 12 steps, extending beyond the two-step overtime commitment.
- The specimen-routing, resource-routing, capacity-transfer, and replenishment
  action blocks are byte-identical across candidates.

## Candidate library

For each of 20 facilities, apply four local directions to the baseline
overtime allocation:

1. decrease the target by 5% of the shared budget and redistribute to other
   facilities in proportion to their remaining local headroom;
2. decrease by 10% under the same rule;
3. increase the target by 5% and withdraw from other facilities in proportion
   to their baseline allocations;
4. increase by 10% under the same rule.

Transfers are clipped only by the target and donor/receiver feasibility. The
realized allocation must remain within every local cap and preserve the
baseline total to `1e-8`. Together with the unchanged baseline, this yields 81
candidates per state. No candidate may inspect future environment outcomes.

## Reserved development data

These seeds were repository-searched before freezing and are disjoint from
all listed prior and formal families:

| Use | Reservation |
| --- | --- |
| State generation | 99700000-99700002 |
| Discovery worlds | 99800000 onward, 2 per state |
| Validation worlds | 99900000 onward, 5 per state |

There are 54 states. Expected rows are 8,748 discovery and 21,870 validation.
The formal routing holdout and all ranges 94,000,000-96,899,999 and
97,100,000-99,599,999 are forbidden.

## Gates fixed before execution

### R0: mechanics

- Every candidate preserves total requested capacity within `1e-8` and alters
  no non-overtime action slice.
- At least 90% of candidate-state directions realize a nonzero transfer.
- Every facility has at least one realized increase and decrease direction.
- Candidate arms finish each paired world at the same RNG state.

### R1: support upper bound

- Optimistic validation oracle saving is at least 0.5%.
- At least 30% of states have a clinically noninferior saving of at least 0.5%.
- Aggregate patient loss and completion service are clinically noninferior.

Failure closes this action support immediately: no observable-state model and
no DDPG experiment are justified.

### R2: prospective replicability

- Discovery-selected, validation-scored saving is at least 0.5%.
- The saving is positive in every scenario and clinically noninferior.
- At least 30% of states are materially and clinically improved.
- At least 70% of discovery-material selected actions still beat baseline on
  validation.
- At least 80% of discovery-material candidate-vs-baseline cost signs agree
  on validation.

Passing R0-R2 authorizes only a new, separately signed observable-state
ranking screen. DDPG remains unauthorized until a ranking model demonstrates
that the stable residual label is predictable from information available at
decision time. A failure is retained and reported; thresholds, candidates,
or seeds are not tuned after inspection.

## Interpretation for the manuscript

- Failure supports the current conclusion that the simulator is
  heuristic-saturated even after covering DDPG's local continuous allocation
  support. It should not be presented as evidence that DDPG is generally
  ineffective.
- Passing identifies unbanked action headroom but still does not establish an
  RL contribution. Predictability and then paired online attribution would
  remain separate claims.
- The current paper and its formal holdout remain unchanged regardless of this
  development screen.
