# Exact Patient Bound And Fixed Economic Tail

Status: prospective source/JSON contract only, 2026-10-02. No patient trajectory,
raw outcome reanalysis, model load/forward, environment construction/step,
optimizer, simulation or test was performed for this memo. The latest locked-plan
amendment permits design, not another numerical attempt
([locked plan:2696-2719][locked]). This refines [design.md:24-59,85-101][design],
including the coordinator's prospective fixed-observation-horizon refinement. It does
not alter completed results or provide an execution authorization.

## Decision

Use **8 transitions as the conservative patient-resolution bound** after the
unchanged 52-transition enrollment window. Use **11 transitions as the fixed
economic observation tail for every arm**, including arms with no active
patients at the cutoff. The extra 3 transitions cover the maximum generic
resource-transfer delay. Total economic observation is therefore 63 transitions.

Run the existing full four-block MDL-2 rule on a declared closed-demand public
view while any cohort patient is active. At the first active-zero state, latch
new orders and all new transfers off permanently. Continue receiving previously
scheduled flows and charging the original primitive costs through the fixed
endpoint. Retain final resources physically in the terminal ledger; no liquidation,
salvage credit, cancellation refund, clinical valuation or invented disposal fee.
Record exact earlier patient-resolution time separately from economic termination.

This removes the accounting incentive to stop common overhead sooner by losing
patients sooner. It does not remove genuine differences in resource use, losses,
expiry or costs already present in the simulator. Patient safeguards remain
necessary in the integrated comparison, not a new review/fit gate.

## Frozen Evidence

All source references below point into the completed baseline's frozen tree,
not the coordinator's new implementation. Define:

- `R = results/dynamic_candidate_time_baseline_20261002/payload`.
- `F = R/locks/worktree`.
- `B60`, `B61`, `B62` are `R/binding/block{60,61,62}.json`.
- `P = F/src/env/patient_capacity_planning.py`;
  `C = F/src/env/capacity_planning.py`;
  `H = F/src/baselines/heuristics.py`.

`R/locks/execution.json:2-7` records execution commit
`45a5e16cd8e26364edaacb3aab885d7e4203621c` and frozen packet
`5279726f8fc069905ec1097fc961c13136c25241136a99383edae78bcd3c6300`
([execution receipt][execution]). These are read receipts, not a newly repeated
archive/hash audit.

The actual reference config path, relative to `F`, is
`results/frozen_baseline_rebuild_20260929/payload/training/frozen_baseline_rebuild_20260929/gcn_residual_mdl2_network_ddpg_afd/seed{60,61,62}/config.json`.
The frozen packet selects it at
`F/specs/2026-10-02-time-baseline-comparison/frozen.json:5186-5206`.
`F/src/rl/dynamic_candidate_backend.py:41-69,95-100` loads that runtime JSON and
uses it for both layout and episode construction ([backend][backend]).
`F/src/rl/experiment.py:38-59` uses its `env` block, not its historical
`multi_scenario_training.scenarios` field ([builder][builder]). In particular,
the latter's `routing_persistent_hotspot_cluster1` at [runtime config:605-616][runtime]
is not the executed scenario here. The recorded scenario is
`routing_nominal_history` ([B60:525-530][b60]).

A JSON-only comparison of the three `effective_scenario` objects found them
identical. Each binding records effective-environment digest
`39d5306cc3cbdcbcde2730c65366d105749f954ff1baed257d313a0ef2c39d21`
at line 519 ([B60][b60], [B61][b61], [B62][b62]). This compares config records,
not episode outcomes. Effective dataclass recording is explicit in
`F/src/rl/frozen_value_probe.py:70-77` ([scenario receipt source][scenario]).

| Effective JSON key under `effective_scenario` | Value | Frozen binding lines |
| --- | --- | --- |
| `base.episode_horizon`, `base.num_facilities` | `52`, `20` | B60:140,271 |
| `material_shelf_life` | `6` | B60:474 |
| `base.production_lead_time` | `3` | B60:278 |
| `finished_shelf_life` | `2` | B60:472 |
| `finished_product_return_lead_time_epochs` | `0` | B60:471 |
| `finished_product_return_assumption` | `return_to_collection_facility` | B60:470 |
| `specimen_routing_lead_time_epochs` | `1` | B60:496 |
| `max_specimen_transfers_per_patient` | `1` | B60:475 |
| `specimen_transit_loss_probability`, `enable_viability_hook` | `0.0`, `false` | B60:497,468 |
| `base.transfer_lead_time` | `3` | B60:458 |
| `base.transfer_lead_time_distance_thresholds` | `[500.0, 1500.0]` miles | B60:459-462 |
| `base.reagent_purchase_lead_time` | `0` | B60:280 |
| `base.enable_stochastic_procurement`, `base.reagent_lead_time_probabilities` | `false`, `[]` | B60:139,279 |
| Overtime control/commitment/fatigue; production throttle | all `false` | B60:134-137 |
| Initial specimens | `0` | B60:199 |
| Demand forecast horizon/source/error | `2`, `effective_rate`, `0.0` | B60:98-100 |
| Demand-history window; time-state enabled | `12`; `true` | B60:101,152 |
| `base.max_reagent_replenishment` | five facilities each at `40`, `115`, `50`, `105`, in order | B60:224-245 |
| `base.max_idle_bioreactors` | five facilities each at `9`, `24`, `11`, `22`, in order | B60:201-222 |
| Max specimen/reagent/reactor transfer request | `120`, `110`, `18` | B60:269,246,200 |

The runtime JSON independently states shelves/leads at lines 201-205,264,357,
508,537-541 ([runtime][runtime]). Read the effective records for defaulted fields;
do not substitute another config with longer procurement or return delays.

## Patient Bound

Let `S_t` denote state after `t` transitions. Prefix actions are `t=0..51`;
the cohort is every identity enrolled by `S_52`, including the last transition's
entrants. Enrollment happens after aging/production/infusion
([P:447-460,953-959][patient]); newly enrolled specimen age is zero
(`F/src/env/patient_condition.py:44-58`, [patient state][condition]). Thus the final
batch has received no service transition at `S_52`. Initial identities are also
included, although this exact config initializes zero specimens.

1. **Waiting and specimen transit share one age budget.** Each transition not
   started in manufacturing increments `specimen_age` once; reaching age 6 loses
   the patient. The two compartments use the same expiry test
   ([P:765-831][patient]). Routing changes location/status/count, not age
   (`F/src/env/specimen_routing.py:192-217`, [routing][routing]). Its one-step
   transit is therefore inside, not in addition to, the six-step shelf budget.
2. **Production starts before that transition's waiting expiry check.** An age-5
   specimen can start at the sixth opportunity; one that does not start resolves
   as lost on that transition ([P:317-354,833-843][patient]). At most five full
   waiting/transit transitions can precede a surviving start.
3. **The start transition counts as manufacturing transition one.** With lead 3,
   a new start advances to slot 2, next to slot 1, then to completion. Existing
   slot-2/slot-1 patients need at most two/one further transitions. Manufacturing
   does not wait for a later reagent delivery or another capacity allocation;
   eligibility loss can only shorten this duration ([P:845-878][patient]).
4. **There is no finished-product waiting extension in this config.** Return lead
   0 delivers on the manufacturing-completion transition, including products
   manufactured away from the collection facility. The scalar finished inventory
   is aged, replenished and fully consumed in that call
   ([P:921-944][patient]). Finished shelf 2 is not an extra mandatory two-step
   residence. Positive-return expiry/delivery is a different branch
   ([P:890-919][patient]); it is not active here.

Therefore, for a waiting/transit patient of valid age `a` at the boundary,
`B(a) <= (6 - a - 1) + 3 + 0 = 8 - a`. The uniform conservative maximum is
**`B_patient = (6 - 1) + 3 = 8`**, since `a=0` is possible. Patients never started
resolve within six transitions; patients already manufacturing resolve within two.
All valid cohort patients are therefore delivered or lost by `S_60`.

Off-by-one example, not a simulated trace: five waiting transitions `t=52..56`,
start at `t=57`, advance at `t=58`, complete/deliver at `t=59`; resolution state
is `S_60`. This is the worst permissible lifecycle bound, not a measured maximum
or a claim that MDL-2 attains it. Adding `6+3+2+1` would double-count both the
start boundary and residence that these frozen contracts do not impose.

Resolution means all cohort identities are in `DELIVERED` or `LOST`, not merely
empty waiting queues. The existing identity audit counts waiting, manufacturing,
specimen transit and product return ([P:1135-1220][patient]). A patient still
active at `S_60`, an invalid stage/age, or a new identity after closure is a
contract failure/unresolved censoring. Never set its liability to zero, extend
the bound automatically, classify loss as completion, or silently drop that arm.

## Fixed Economic Horizon

Generic reagent/capacity transfers use
`delay = min(3, max(1, 1 + count(distance > [500,1500])))` when geographic delays
apply; otherwise they use `transfer_lead_time=3`
([C:1693-1707][capacity]). Thus the integer maximum is **3**, not the patient
specimen lead 1, not manufacturing lead 3 reused as a different contract, and
not continuous transport hours rounded without the source's cap. It bounds all
actual-config edges without any new geometry computation or environment build.
The binding also records 190 physical capacity edges, distinct from the 36
resource/specimen edges ([B60:533-537][b60]); the GCN/hub representation is not an
additional physical transit stage.

New resource transfers enter `pipeline[delay-1]` after that transition's arrivals
were popped ([P:291-294,400-428][patient]; [C:1663-1691][capacity]). The action-stop
rule permits commitments on a transition that begins active and ends resolved;
the latest such transition is `t=59`. Conservatively reserve for a delay-3
transfer placed there: it arrives at the start of `t=62`, with its on-hand holding
charge assessed in that same transition. This does not assert that zero-demand
MDL-2 actually issues a nonzero transfer that late; the bound does not depend on
tightening its action behavior. There are no new transfers after `S_60` at the
latest.

Set **`H_economic = B_patient + max_generic_transfer_lead = 8 + 3 = 11`**.
Observe `t=52..62`, stopping economically at `S_63`, identically for every arm.
The final resource drain is guaranteed by this action-stop rule and the frozen
lead cap, not by empty patient queues alone. Procurement lead is zero, so no
extra procurement suffix is needed. A nonempty purchase pipeline or a longer
lead would contradict this admitted config and needs explicit re-derivation,
not an undocumented extension. This is a conservative common horizon, not a
claim that all eleven transitions are needed for patient resolution.

## Common Follow-Up Rule

**Prefix isolation.** Keep `episode_horizon=52`, original done flags, original
time normalization, prefix actions, costs, observations and RNG consumption
unchanged through all 52 transitions. Save the complete `S_52` receipt before
applying closure. The original final `_advance_clock` already samples and records
the next demand; that draw is not an enrolled patient and must not be enrolled
during follow-up ([C:1440-1449][capacity]; [P:457-460][patient]). Do not suppress that prefix
draw retroactively. Do not build/reset another world or reset any patient,
material, resource pipeline, identity counter or RNG to begin follow-up.

**Tail-only information.** After the prefix receipt, latch enrollment closed,
set tail current demand and forecast to 20 zeros, and expose zero prospective
mean-demand/rate-estimate vectors to MDL-2. Keep the immutable prefix config and
receipt intact; this is an explicit tail-only input overlay. The baseline uses
`heuristic_settings_for_policy("mdl2")`
(`F/src/rl/patient_replay_collector.py:67-71`, [producer][producer]): lookahead 2,
sharing on, order-up-to multiplier 1, forecast/history/patient-priority switches
off ([H:20-32,53-85,179-185][heuristics]). Its default workload is
`max(next_specimens,0) + 2*demand_rates` ([H:844-880][heuristics]). **Zero forecast
alone is insufficient.** Both `demand_rates` and any resolved
`demand_rate_estimates` in the tail heuristic view must be zero, without changing
prefix feature construction or relabeling this as fMDL-2.

Keep prefix demand/history/error arrays in the saved receipt, and retain factual
past history in the live environment. Only current demand, forecast and prospective
rate inputs become zero; the selected plain MDL-2 ignores history features.
In tail `_advance_clock`, increment the
clock, advance existing regional supplier-disruption state and sample supplier
availability under the original contracts; append zero demand and zero forecast
error to the tail histories. No new Poisson demand, demand-shock/referral or
forecast-noise draws; do not call the original clock and merely discard its new
referrals. Retain the cutoff supplier state for the first tail decision and do
not reseed. This changes post-cutoff RNG consumption prospectively; it does not
claim equivalence to the open-inflow continuation. Supplier logic and history
recording are at [C:1434-1449,1452-1471,1550-1575][capacity].
Advance this same tail clock after active zero as well; do not skip the remaining
economic transitions or supplier draws in an arm-dependent early exit.

Continue to report the frozen time coordinate `clip(t/52,0,1)=1` during the tail
([C:1517-1518][capacity]); use a separate tail counter for economic termination.
Never replace 52 with 63 before prefix collection. No learned-policy decisions,
learning, hidden risk types or future patient outcomes enter the tail rule.

**Actions.** At each tail pre-decision state with active cohort patients, use
the complete MDL-2 specimen/reagent/capacity/replenishment request, with the
same graph, limits, float32 calculation, deterministic ordering and environment
execution/projection semantics ([H:715-803,828-921][heuristics]). Do not add a
procurement optimizer, risk oracle, alternate controller or tuned order cap.
Permitting new replenishment here matters: removing it at `S_52` would impose
a different resource-deprivation policy and could cause artificial losses.

At the first active-zero state (including `S_52`), latch the normalized action
to **`[0]*60 + [-1]*20`** for all remaining tail transitions. All-zero action is
not idle: its replenishment block orders half the maximum
([P:300-307][patient]; [H:1419-1423][heuristics]). Do not wait until all eight
patient transitions have elapsed to stop new commitments. A transition that
starts active and ends inactive still pays for all actions it already executed;
no retroactive cancellation. Record resolution offset `r in [0,8]` separately.

## Cost And Stock Settlement

Use the exact primitive costs from each transition once, not `base_cost` plus
those same components a second time. Keep network-level resource costs at network
level; do not fabricate per-patient allocations. The fixed accounting contract is:

```text
C_window       = sum(t=0..51, original_cost_t)
C_tail_fixed   = sum(t=52..62, original_cost_t under the declared tail rule)
C_cohort_fixed = C_window + C_tail_fixed
C_tail_patient = sum(t=52..52+r-1, original_cost_t)  # empty if r=0
C_tail_post    = sum(t=52+r..62, original_cost_t)
C_tail_fixed   = C_tail_patient + C_tail_post
```

`C_cohort_fixed` refines design.md's resolution-stopped formula; report the latter
as `C_window + C_tail_patient`, not as the primary fixed-exposure objective.
For a later authorized cohort-objective training arm, append `-1e-9*C_tail_fixed`
once to the final prefix reward, keeping all original prefix cost/reward receipts
separate. Do not add both future components and their total. The old-objective
arm logs the same complete tail but excludes it from its training target.

| Item | Required treatment and frozen support |
| --- | --- |
| Already paid purchases | Original purchase cost is `42174 * sum(replenishment)` at order placement, not consumption or arrival ([B60:88-96][b60]; [C:1221-1233][capacity]). Keep prefix expenditure in `C_window`; never charge it again when consumed/received. Initial reagent endowment likewise gains no new imputed purchase bill. |
| Procurement in flight | Actual purchase lead is 0 and stochastic procurement is off. The no-pipeline branch adds new supply after current production, making it available for later production ([P:362-372][patient]; [C:1011-1019][capacity]). This is not a fresh uncharged order at `S_52`. A positive purchase queue is not expected under this config. If admitted in a future different contract, preserve its original paid status and receive it once using [C:1044-1060][capacity]; this memo's 11-step bound does not silently admit that change. |
| Reagent/capacity already transferring | Preserve every pipeline slot at closure and after active zero. Transfer charges arise from the sending transition's executed flows ([P:490-524][patient]); arrival adds no second purchase/transfer charge. The three-step suffix receives all due flows. Supplier unavailability constrains new purchases, not reception of already dispatched transfers ([P:291-307][patient]). |
| New tail replenishment | Permit the closed-demand MDL-2 quantity while active, with original supplier gating, facility limits and source ordering. Charge all executed quantity once, including any later unused quantity. Stop new orders after active zero; no refund for over-ordering and no bespoke on-order correction to plain MDL-2. |
| Holding | Per facility, charge `113.5*max(reagents-waiting,0)` and `14.4*max(idle_bioreactors-waiting,0)` on the post-transition on-hand state, every tail transition ([P:462-466][patient]; [C:1223-1227][capacity]). After active zero, waiting is zero, so all remaining on-hand stock/idle capacity is charged through `t=62`. No extra invented holding on resources still in transit or on occupied reactors. |
| Shortage and patient costs | Retain reagent shortage `86504.5`, reactor shortage `50273.6`, patient loss `500000`, material expiry/waste `100000`, urgency `25000`, with the existing exact count definitions ([B60:88-96,504-506][b60]; [P:462-488,534-541][patient]). Do not delete loss/expiry overlap or reinterpret loss as clinical death. |
| Transfers | Retain base specimen/reagent/reactor rates `600/200/500`, geographic cost scale `0.25`, time-cost scale `0.05` and original executed-flow formulas ([B60:88-96,142-145][b60]; [P:490-504][patient]; [C:1754-1779][capacity]). Do not reprice movement as straight-line distance times a newly chosen rate. |
| Residual resources | Retain final reagent quantities, idle/busy reactor slots and any pipelines in a terminal stock ledger. No credit, new charge, destruction, reset, donation or liquidation at `S_63`. Reactor acquisition/depreciation and reagent resale/disposal valuations are not supplied by these primitive contracts. |

**Existing clipping is not new disposal authority.** The frozen environment clips
on-hand reagents and idle reactors to facility maxima on arrival and bookkeeping
([C:487-500][capacity]; [P:443-445][patient]). The wrapper must not zero resources
or add any further clipping, and must not claim strict physical conservation if
the frozen clipping discards overflow. In the later resource ledger distinguish
due arrivals, admitted arrivals and source-implied clipped overflow; provide no
invented salvage/refund/disposal value for it. Retain the original purchase cost.
Changing this loss mechanism would be a separate environment amendment, not
part of the follow-up mixin.

The final receipt must distinguish `patient_resolved_at`, `economic_observed_to=63`,
unchanged enrolled identity set, delivered/lost counts and reasons, remaining
active compartments, all resource pipelines, residual on-hand resources and
primitive cost sums for window/tail/post-resolution. A complete receipt has
active count zero by `S_60` and no pending modeled flow at `S_63`; residual stock
need not be zero. These are implementation requirements, not checks run here.

## Estimand And Limits

The estimand is **first-52-step policy performance for a closed enrolled cohort,
under a common closed-demand MDL-2 continuation, with a common 63-transition
economic observation boundary and retained, unvalued terminal stock**. Equal
observation length removes arm-specific stopping-time exposure bias; it does
not force equal residual stocks or equal costs. Exact patient resolution, losses,
completions, waiting/turnaround and the old-window metrics remain separate outcomes.

An ongoing-stock service instead continues new referrals and congestion, needs
replenishment for future cohorts, and can use residual reagent/capacity after this
cohort resolves. Zeroing future demand changes that estimand. This memo does not
estimate steady-state service value, indefinite operating cost, lifecycle clinical
benefit, external financial NPV or a liquidation value. It also cannot make an
old saved policy gain appear through a newly chosen valuation.

The forced economic decision is **finite common observation, not complete lifetime
financial discharge**. The source supplies no reagent shelf life or reactor
retirement mechanism that would extinguish holding charges for retained resources.
With positive stock and positive holding rates, continuing those charges forever
has no finite total. Patient resolution plus pipeline drainage therefore cannot
prove zero future economic liability. Here future stock is explicitly retained
and unvalued, and beyond-63 holding is outside the named estimand, not waived via
free disposal. Full ongoing-stock/lifetime accounting would need a separately
justified horizon or terminal-value/resource-use model; inventing one is prohibited.

There is **no hard source/config blocker to this bounded contract**. Wrapper
correctness, prefix bit identity, identity conservation and fixed-tail budget
integration remain unverified engineering work owned by the coordinator. The
8/3/11 integers are source-derived bounds, not measured validation or execution
caps approved for a new experiment. Keep the existing efficiency advice: one
integrated packet and direct comparison, reuse completed work, no new reviewer,
baseline search or toy-fit gate. This memo owns only this file; no compile/test
run or other artifact edit is part of this source-only assignment.

[locked]: ../../docs/patient_indexed_specimen_routing_locked_execution_plan.md
[design]: design.md
[execution]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/execution.json
[b60]: ../../results/dynamic_candidate_time_baseline_20261002/payload/binding/block60.json
[b61]: ../../results/dynamic_candidate_time_baseline_20261002/payload/binding/block61.json
[b62]: ../../results/dynamic_candidate_time_baseline_20261002/payload/binding/block62.json
[runtime]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/results/frozen_baseline_rebuild_20260929/payload/training/frozen_baseline_rebuild_20260929/gcn_residual_mdl2_network_ddpg_afd/seed60/config.json
[backend]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/dynamic_candidate_backend.py
[builder]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/experiment.py
[scenario]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/frozen_value_probe.py
[patient]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/patient_capacity_planning.py
[capacity]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/capacity_planning.py
[condition]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/patient_condition.py
[routing]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/specimen_routing.py
[heuristics]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/baselines/heuristics.py
[producer]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/patient_replay_collector.py
