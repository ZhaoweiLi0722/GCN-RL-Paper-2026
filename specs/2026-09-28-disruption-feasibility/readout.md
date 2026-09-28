# Disruption feasibility: first bounded pilot

Completed 2026-09-28. Local development only. No neural model was trained,
no formal holdout used, and no source/evidence in earlier campaigns changed.

## What was integrated

Merge commit `9139798` brings Howard's `e2b-run` (`461dbe4`) together with
our audited local residual screen and P0 information-contract work (`328806d`).
The integrated evidence ledger is
`docs/team_updates/2026-09-28-integrated-evidence-and-next-study.md`.
The manuscript discussion now distinguishes local first-action failures,
sequential control, and online weight updates. Howard's original reports are
preserved, not rewritten to imply independent verification or new approval.

The user explicitly chose **local only**: no push, new PR or main merge.
Persistent worktree: `worktrees/september-research-integration` under the
project directory, not a temporary directory.

## Results

Execution commit `e16ec36c42d1e87d6ef6dca0c2facf8587bd1bec`.
Exactly 192 episodes / 9,984 steps: four cells, four fixed rules, four
discovery and eight independent validation worlds. Worlds are paired across
rules and cells, so 192 episodes are NOT 192 independent statistical units.
Lower cost is negative below; all values are validation changes vs MDL-2.

| Fixed rule | No change | Capacity outage | Procurement-delay change |
| --- | ---: | ---: | ---: |
| MDL-2 | 0% | 0% | 0% |
| MDL-3 | +2.53265% | +1.79313% | +2.22805% |
| Inventory-position/lead-aware MDL-2 | -0.02011% | -0.01248% | +0.80301% |
| Same rule, 1.2 safety multiplier | +0.28738% | +0.29302% | -0.08848% |

The two no-change cells are bit-identical by construction and verification;
they are not separate replications. Discovery selected the plain lead-aware
rule for no-change/capacity cells, and the safety-multiplier rule for the
delay-change cell. Each selected rule lowered validation cost in 5/8 worlds.
No inferential or clinical-noninferiority conclusion is claimed.

Selected rule versus MDL-2, validation mean clinical differences:

| Cell | Completion change (pp) | Manufacturing ineligibility change (pp) | Lost patients per episode |
| --- | ---: | ---: | ---: |
| No change | +0.01868 | -0.01448 | +0.125 |
| Capacity outage | +0.01633 | -0.00772 | +0.250 |
| Procurement-delay change | -0.05856 | +0.02726 | +1.750 |

The delay-cell cost reduction comes with slightly adverse patient endpoints.
It is not a demonstrated safe operational gain. The capacity outage removes
10 of 210 reactor units from usable stock, not half the network: the frozen
floor(0.5 * 5) rule quarantines two units at each of five sites.

## Mechanics findings and scope

The first smoke failed before any discovery/validation run. At decision 1,
the legacy transfer-arrival handler clipped 5.6000005 idle reactor units at
zero-based site 3. A direct source trace located this in
`CapacityPlanningEnv._receive_transfer_arrivals`; a unit fixture reproduces
the loss when arrivals exceed the local idle-stock ceiling. This is an
inventory-accounting/modeling issue, not evidence that RL training failed.
Its incidence and effect on the earlier formal comparison are **unknown**;
do not claim historical conclusions either invalidated or unaffected by it.

The new synthetic pilot uses a common nonbinding idle-stock ceiling of 210
at every site, frozen before its scientific seeds were run. This eliminates
disposal by clipping but relaxes local storage constraints. It is not an
innocuous patch or a test of the original formal environment. Historical
source and results remain intact. An operationally meaningful overflow rule
and sensitivity assessment are needed before promoting this setup.

The failed smoke, its original composed config, and the successful mechanics
rerun are retained under `smoke/` and `smoke-r2/`. No scientific retry or
post-result tuning occurred. The full pilot's maximum physical-capacity
accounting error was 2.84e-14; paired arrivals/RNG and identical-control checks
passed. Rewards and logged costs reconcile.

At horizon 52, selected policies still have approximately 417--437 waiting
patients and 129--131 patients in production, plus transfers. The delay case
has about 117.26 reagent units on order versus 8.63 in the no-change case.
These terminal liabilities are recorded rather than assigned a new salvage
value after the result. Endpoint differences cannot establish complete
patient outcomes. Eight validation worlds and illustrative uncalibrated
parameters are further limitations.

## Decision and next work

Do not start another DDPG campaign from these results. This four-rule library
does not bound the potential value of a sequential policy, so its small gains
are also **not proof that either scenario lacks RL opportunity**.

1. Resolve idle-stock overflow semantics and define a terminal settlement
   rule. Audit the original model's clipping incidence separately, preserving
   its existing reported outputs and avoiding formal model selection.
2. Obtain Howard's raw planner episode rows, decision logs, execution hashes
   and observation contract. The integration test shows that equal neural
   observations do not fix the per-patient inputs read by his sampler.
   This affects the information-matching claim, not a quantified cost effect.
3. For the procurement scenario, compare public-history lead estimation plus
   an inventory-position rule, and a repeated-decision planner using the same
   observations. For capacity, distinguish announced unavailability from an
   unknown conversion response to a controllable commitment. Keep these
   mechanisms separate and calibrate them before a larger sweep.
4. Only if repeated-decision improvement is clinically acceptable and
   independently replicable beyond the strongest challenger, freeze a
   belief-adaptation x online-weight-update design. The key comparator is a
   tensor-matched frozen history-aware policy, not just MDL-2.

The paper's defensible current story is still graph/AFD policy gains with
unestablished online actor-critic increments. Better simulator validity,
information parity, crossed inference and strong comparators improve the
paper even if the eventual online result remains negative. The new pilot
numbers are not additions to the formal performance table.

## Reproduction and audit

Raw files: `reports/2026-09-28-disruption-feasibility/pilot/` contains
`episodes.jsonl`, `steps.jsonl`, `execution.json`, `summary.json`, `status.json`
and `inventory.json`. The runner refuses to overwrite an existing output.

```bash
# Read-only; no new simulation, model loading or training:
python3 specs/2026-09-28-disruption-feasibility/audit.py
```

The standard-library audit checks all nine source and five output hashes,
episode/step Cartesian products, finite metrics, cost totals, conserved
stock, paired fingerprints, discovery-only selection and independently
recomputed paired cost changes. It does not independently replay clinical
dynamics. The earlier 17-hash residual audit and 9-hash P0 audit also passed
on the merged tree. A final regression/compilation record accompanies the
integration commit: 162 focused tests passed across 13 test modules, full
`python -m compileall -q .` passed, and `git diff --check` was clean.
The manuscript change is source-only; no new PDF was built in this turn.
