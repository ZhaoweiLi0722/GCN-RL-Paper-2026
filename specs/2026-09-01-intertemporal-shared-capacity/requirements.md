# Environment Requirements

## Clinical-operational interpretation

Facilities receive advance referral or collection schedules. A regional
staffing pool can be reserved continuously across clinics, but reservations
have a ramp lead time and cannot be moved instantaneously. Matured staffing
persists, incurs convex operating cost, and accumulates fatigue. The control
therefore trades a cost paid now against patient and shortage outcomes several
epochs later.

## Scheduled referral waves

The base environment gains a default-off scheduled-wave mechanism:

- `enable_scheduled_referral_waves`
- `scheduled_referral_clusters`
- `scheduled_referral_start_step`
- `scheduled_referral_block_length`
- `scheduled_referral_peak_multiplier`
- `scheduled_referral_repeat`

At epoch `t`, one declared cluster receives the peak demand multiplier. The
cluster changes only at block boundaries. The existing demand-forecast feature
must sum the expected demand over its configured horizon, including known
future referral blocks. Poisson arrivals and all other random mechanisms stay
unchanged.

The schedule is explicit configuration, not inferred from an evaluation seed.
Held-out scenarios change cluster order, block length, or forecast error rather
than silently changing environment code.

## Intertemporal overtime commitment

The existing overtime action block is reused only when both
`enable_overtime_control` and
`enable_intertemporal_overtime_commitment` are true. The new fields are:

- `overtime_commitment_lead_time`
- `overtime_commitment_persistence`
- `overtime_shared_budget_fraction`
- `weight_overtime_activation`

Raw actions still map continuously from `[-1, 1]` to per-facility requests.
Requests are proportionally projected onto a shared network budget when their
sum exceeds it. This projection is continuous and preserves relative
allocation.

A request submitted at epoch `t` cannot affect production before
`t + overtime_commitment_lead_time`. When it matures, active capacity follows

```
active[t+1] = persistence * active[t]
              + (1 - persistence) * matured_request[t]
```

so its effect carries across epochs. Active capacity, not the new request, is
the surge available to production. Overtime operating cost is charged on
active capacity. A separate activation cost is charged on the absolute change
from the prior request. Fatigue, when enabled, follows active utilization.

## Markov state and provenance

Flat and graph observations include, per facility:

- prior projected request;
- active overtime capacity;
- each pending commitment-pipeline slot;
- outstanding borrowed production capacity;
- static local headroom;
- remaining shared-budget fraction, repeated per node;
- fatigue when enabled.

Patient-environment snapshots include the active capacity and full commitment
pipeline. Exact save/restore must reproduce the next transition and reward.

## Comparator contract

The treatment channel has no MDL-2 action anchor during future learning. The
base routing/replenishment policy may remain frozen, but the new commitment
head starts neutral and is updated only by the attributed online arm.

Evaluation must nevertheless include strong external comparators:

1. no shared overtime;
2. a development-tuned static network allocation;
3. a forecast-proportional allocator using the same visible schedule;
4. a graph-smoothed forecast allocator that can move capacity only over the
   declared staffing network.

A learned policy must beat the strongest eligible comparator. Beating only
the zero-overtime arm is insufficient.
