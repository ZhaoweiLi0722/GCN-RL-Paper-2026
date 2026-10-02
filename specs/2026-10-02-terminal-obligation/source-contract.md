# Terminal Obligations: Source Contract and Next Design

2026-10-02. Preparation only, after the completed null time-baseline comparison.
Read `AGENTS.md` and the latest locked-plan amendment, which closes an
unchanged-greedy baseline attempt and permits the reward-design pivot, not new
numerical scope ([plan:2659-2682][plan]). Only this file is authored. No raw
terminal-file analysis, simulation, model call, fit, weight search or execution
approval is supplied; the parent owns parsing, numerical findings and integration.
Source references below point to the completed attempt's frozen source copies;
the seven cited modules match the inspected checkout at
`e916c414c89bc334ad96e4cde48536535228791d`.

## Entry, Aging and Termination

The objective is 52 transitions, negative absolute step cost scaled once by
`1e-9`, gamma/lambda 1, zero terminal cost and zero terminal bootstrap
([factory:15-26][factory]). A learning-objective terminal is not patient resolution.

1. Reset enrolls initial waiting patients. Each step receives due resources and
   routed specimens, executes specimen routing, then starts production subject
   to waiting patients, idle reactors and reagents. A start is not a completion
   ([env:143-151,291-340][env]).
2. Remaining waiting and specimen-transit patients age once, with specimen expiry
   checked before ineligibility. Existing and newly started manufacturing
   patients also age once; ineligibility precedes manufacturing completion
   ([env:342-357,765-878][env]).
3. Resources are updated, then finished returns age and face expiry/eligibility
   checks before infusion. New completions are infused immediately at return
   lead zero, otherwise enter the return queue. Only afterwards are current
   demand arrivals enrolled; costs follow, then the clock advances
   ([env:359-466,534-567,880-959][env]).
4. On the last transition, pre-step `t=51` becomes `t=52` and `done=True` regardless
   of unresolved patients. New final-step arrivals have `enrollment_epoch=51`,
   `age=specimen_age=0`, and have received no routing/production opportunity or
   aging yet. Reset patients also have enrollment epoch zero, so epoch alone
   cannot distinguish them from first-step arrivals. Use actual age and identity
   provenance ([condition:44-58,124-149][condition]; [env:143-146,953-959][env];
   [base:1440-1450][base]). The next demand is sampled even at termination but
   is not another enrolled cohort. Boundary accounting records unresolved
   identities without charging them ([boundary:133-146,195-218][boundary]).

## What Can Be Present at Step 52

This is a source-level state map, not measured terminal counts or a claim that
every compartment is occupied. Parent inspection supplies the actual composition.

| State/location | Meaning and possible remaining obligation |
| --- | --- |
| `WAITING`, facility queue | Includes fresh final-step entrants and older survivors. Remaining routing, production and infusion compete for resources; specimen expiry and eligibility remain unresolved. |
| `IN_TRANSIT`, specimen transit | Routing has occurred. A record with `remaining_epochs=0` still awaits the next step's receipt before production; it is not delivered care. |
| `IN_PRODUCTION`, pipeline stages 1 onward | Start consumed reagent and occupies capacity; remaining manufacturing can complete or lose eligibility. Stage zero is not an active patient compartment. |
| `FINISHED`, product-return transit | Manufacturing completed, not yet infused; return timing, product expiry and eligibility remain. Return lead zero delivers immediately rather than leaving this queue. |
| `DELIVERED` / `LOST`, registry only | Resolved within the model and excluded from active locations. `LOST` includes expiry, transport loss and ineligibility; it is not synonymous with observed death. No post-infusion clinical outcome is modeled here. |

Sources: [env:746-878,880-951,1135-1220][env]; [identity:60-107,242-249][identity];
[condition:23-31,180-195][condition]. `terminal_active` is the unique identity
count across the first four rows, not waiting alone or a count of failures.

## Already Charged Versus Future Obligations

| Ledger | Already included through step 52 | Residual only, if the objective is extended |
| --- | --- | --- |
| Patient harm | `weight_patient_lost * patients_lost`; total loss includes expiry and transport loss. Waiting patients below the urgency threshold incur the per-step urgency charge. | Future loss/urgency events among unresolved patients, not a second charge for historical losses and not certain death for every survivor. |
| Material waste | `weight_expiry * material_wasted`, including waiting/transit/finished expiry, transport loss and manufacturing discard. | Future discard/disposal or other incremental material consequences, reconciled with purchases already paid. |
| Resources/operations | Purchases are charged on replenishment orders, including orders still in pipeline; idle inventory/capacity holding, queue-based reagent/reactor shortage and executed transfers are charged each step. Production subtracts reagent at start. `base_cost` is the sum of operating components, not an additional fixed charge. | Remaining operational use, new procurement/transfer and any outstanding commitments, net of defensible residual inventory value; do not repurchase consumed or already ordered reagent in a terminal estimate. |

Sources: [env:359-372,462-541,961-966][env]; [base:1205-1255][base]. Shortage
terms use end-of-step waiting minus available resources, after new enrollment;
they already price a form of queue pressure, but do not settle all later care.
`waiting_patient_steps` itself is a reported occupancy sum, not a separate cost
([metrics:162-188][metrics]). Do not add operating subcomponents to `base_cost`
again. Expiry losses are a subset of patient losses; patient harm plus material
waste can legitimately price different consequences of one event. Whether
either duplicates shortage or acquisition costs requires domain definitions,
not deleting a term because events overlap.

## Continuing Inflow and Right-Censoring

For each conserved episode, `enrolled = delivered + lost + active`. At equal
enrollment, `delta(active) = -delta(delivered) - delta(lost)`
([identity:60-107][identity]). Fewer early losses can therefore leave more active
survivors, even with more completions. Conversely, premature loss can shrink
active counts. Neither lower nor higher `terminal_active` alone proves good or
poor care; distinguish age, risk, stage, waiting and eventual outcomes.

Enrollment continues through the last step while observation stops at 52. Later
entrants have less follow-up, including zero treatment opportunities for the last
batch. Active outcomes are administratively right-censored, not zero future
burden. Simply lengthening the horizon while retaining inflow creates another
last cohort and repeats the boundary problem. This is a declared finite-window
objective limitation, not proof of a coding defect or the cause of PPO's null.

## Two Different Designs

**Calibrated terminal liability:** keep the observation window and add a common
`L_T(s_T)` once to its accumulated cost. It should estimate only remaining burden
under a named continuation policy and demand/resource regime. Stage, patient
age/risk, material age, residual lead times and shared resource constraints matter;
a separable per-patient sum is an assumption, not guaranteed by the simulator.
Calibration, uncertainty and cost overlap remain unresolved. A nonzero terminal
liability changes the objective, unlike terminal-zero potential shaping
([reward memo:39-52,54-82][memo]). It is an estimate, not observed follow-up.

**Fixed-enrollment-window follow-up to resolution:** retain initial patients and
all admissions during a declared window, then close enrollment and follow those
identities until each reaches modeled infusion or loss. Continue resource,
aging and transport dynamics and count incremental costs/outcomes; do not reset
survivors or repeat sunk charges. Stopping is outcome-based, not an arbitrary
extra number of epochs. A computational stop with unresolved identities is
censoring, not completed follow-up. Closing inflow changes congestion: this is a
closed-cohort estimand, not continuous-service performance. Tracking a fixed cohort
while continuing background referrals is a different design requiring explicit
competition and cost attribution. Neither design has been executed here.

## Decisive Next Design

**Recommend fixed-enrollment-window follow-up to modeled resolution as the next
objective-design direction, ahead of inventing a terminal penalty.** Retain the
existing 52-step enrollment window, including its final arrivals, and specify a
closed-cohort drain phase. Preserve the original 52-step report separately; pair
cohort completion/loss and time-to-resolution with incremental resource costs.
Resolution means infusion/loss in this simulator, not lifetime clinical benefit.
This refines the earlier liability-first recommendation ([readout:50-54][readout]);
it does not amend the locked objective or authorize implementation/execution.

The missing domain inputs are:

- Intended care endpoint and cohort eligibility: infusion versus later outcomes,
  meaning of ineligibility/expiry as final loss, and whether recollection or rescue
  care should exist. Clinical risk/eligibility and transport/storage assumptions
  need justification, not inferred economic coefficients.
- Whether closed enrollment is the intended service question; post-window supply,
  staffing/capacity, procurement and residual inventory/commitment treatment.
- Economic perspective and units: what shortage, urgency, patient loss and waste
  each value; incremental versus sunk costs, residual credits and time valuation.
- A prospectively specified continuation rule and its public time/observation
  semantics beyond 52. A common fixed rule isolates inherited terminal burden;
  continuing each controller instead evaluates a longer controller package.

**No retrospective RL increment:** if PPO and frozen have identical final patient
states and actions, any common patient-state terminal valuation adds the same
amount. More generally, with equal original costs and equal inputs to a common
`L_T`, `delta(C + L_T) = 0`. A resource-aware valuation additionally needs equality
of its resource/pipeline inputs; equal active counts alone are insufficient.
The existing readout reports identical actions and raw outcomes
([readout:5-22][readout]); the parent owns identity/full-state confirmation.
Policy-specific terminal functions or different future actions do not recover a
historical increment: they introduce a different estimand. Any possible future
benefit from retraining or longer follow-up remains unmeasured.

No coefficient, extension horizon, training approval, extra audit gate or new
simulation result is selected. Only documentation was changed; source tests,
compileall, raw-file diagnostics, readout, Live/workflow and all integration remain
parent-owned. No source, result or locked file, commit or automation was changed.

[plan]: ../../docs/patient_indexed_specimen_routing_locked_execution_plan.md
[readout]: ../../reports/2026-10-02-time-baseline-comparison/readout.md
[memo]: ../2026-10-02-time-baseline-comparison/reward-pivot.md
[env]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/patient_capacity_planning.py
[base]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/capacity_planning.py
[condition]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/env/patient_condition.py
[factory]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/time_baseline_factory.py
[boundary]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/candidate_collection_boundary.py
[identity]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/candidate_pilot_verification.py
[metrics]: ../../results/dynamic_candidate_time_baseline_20261002/payload/locks/worktree/src/rl/dynamic_candidate_verification.py
