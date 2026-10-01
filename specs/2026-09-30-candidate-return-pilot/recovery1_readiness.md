# P1 recovery readiness and graph contract decision

Updated 2026-10-01T00:55Z (September 30 local time). Engineering changes are
complete; **no recovery scientific attempt has started**. This is not another
negative RL result. The remaining issue is an explicit graph-contract decision.

## Authorization and preserved failure

Zhaowei replied "continue" after the concrete P1-R1 request: repair interface
compatibility and cross-process timing, then freeze and run at most one new
bounded attempt only if the original scientific design remains unchanged.
That permits engineering work, not silently changing the graph representation.
No Howard approval is claimed. The original P1 authorization, protocol, proposal,
execution packet, failed tree and archive remain byte-identical.

The first P1 attempt remains terminal: one environment construction/reset,
zero env.step calls, zero optimizer steps, zero completed trajectories. This
turn added zero real environment constructions, simulation steps or scientific
updates. Artificial tests include tiny numerical optimizer transactions only.

## Verified mismatch

The static inspector reads all three locked R4 configs and uses the factory's
existing graph-ablation, coordinate and edge-normalization helpers. It never
constructs an environment or loads a policy. All three produce the same result:

| Item | R4 configuration and static resolution | Current candidate producer |
| --- | --- | --- |
| Public raw observation |20 facilities,561 values |20 facilities,561 values |
| Action |80 facility-net coordinates |80 facility-net coordinates |
| Physical specimen edges |36 |One shared adjacency required |
| Physical reagent edges |36 |Must equal specimen edges |
| Physical capacity edges |190 |Must equal specimen edges |
| Physical information edges |36 |Must equal specimen edges |
| R4 graph hub |21 graph nodes;20 capacity spokes |No hub layout accepted |

Physical capacity-sharing edges and the reference model's hub/spoke graph are
different representations of capacity sharing, not duplicate edge counts.
The hub does not add a node to the raw561-value observation. Removing only the
hub guard would expose the next error: heterogeneous relations. Combining the
edge sets into one union would instead make the candidate graph complete,
losing the geographical routing structure. Neither is an interface-only fix
under the original protocol's explicit stop rule.

This is static metadata evidence, not a realized-topology or simulator-parity
claim. The original producer and all environment/model dynamics are unchanged.
Readiness now rejects this input before source freeze, exclusive claim,
reference loading or environment construction. The backend also has the veto.

## Engineering verification

- Replaced cross-process comparisons of Python3.9 Mac `time.monotonic()` with
  explicit POSIX `clock_gettime(CLOCK_MONOTONIC)`. Ledger, supervisor, parent
  claim and runtime receipt bind `posix-clock-monotonic-v1`. No fallback to a
  process-local clock. Unsupported runtimes fail before launch.
- Old ledgers remain readable for independent arithmetic/history checks but
  cannot masquerade as live shared-clock receipts. Injected test clocks are
  likewise not accepted for live process supervision.
- Tests cover a fresh child's clock bracketed by parent readings, an older
  parent supervising a new one-second scope, correct timeout/reaping, successful
  completion, and missing/legacy clock rejection. No research child was launched.
-39 targeted tests passed in3.655s; full288-test relevant regression passed
  in53.259s. Full-repository compileall and git diff checks passed. Real patient
  constructors/reset/step and policy loading are forbidden in the new tests.
- Static readiness exits1 as intended: incompatible topology, not a test crash.
  Seven R4 hashes match;890 external-to-P1 historical JSON/JSONL files have no
  stream collision. This scan excludes the original P1 root and does NOT make
  its already initialized preflight ordinal0 fresh again. All demonstration,
  qualification, training and test streams remain unconsumed by P1. Any recovery
  packet must explicitly account for that distinction before launch.
- Rechecked all15 failed-tree/archive members,11 original/R4 locks, and890
  prior audited/excluded-file hashes. Both Dropbox-local copies still match.
  Cloud sync and Howard access remain unverified.
- Host process scan returned only87683/87686 (inspection shell/filter), no
  P1 runner or tests. No automation was recreated; other tasks were untouched.

Evidence: `reports/2026-09-30-candidate-pilot-integration/recovery1-readiness.json`
and `recovery1-preservation.json`. Code: `candidate_pilot_compatibility.py`,
`research_clock.py`, and the existing campaign/execution/watchdog boundaries.
These are engineering receipts, not a new effective execution packet.

## Proposed decision not yet approved

**Recommendation: explicitly narrow the new candidate's message graph to the
36 specimen-routing links, without changing the simulator or R4.** This is a
small scientific clarification/amendment, not a silent compatibility patch:

- Keep20 public facility nodes and all561 numerical observations,80 requested
  action coordinates, unchanged R4/MDL-2/options, integer execution, reward,
  objective, horizon and data split. The real190-edge capacity network and R4's
  hub-aware inference remain intact.
- Use specimen adjacency only in the new graph message operator. Every new
  representation receives the same raw observations, declared specimen links,
  candidate/reference information and class features. Preserve widths16/32 and
  original parameter counts, which must be rechecked rather than assumed.
- Call it a specimen-routing graph ablation, not full multi-relation information
  modeling. Shared R4 features still contain a GCN; flat remains parameter
  unmatched. Frozen/PPO/continued-BC same-start attribution remains primary.
- Keep the95% qualification gate,32 continuation episodes/model,396 evaluations,
  all subcaps and total52,728 simulator calls/4,608 updates/6h. Preserve the old
  failure; a new directory/packet must be frozen before one new attempt. No
  automatic retry or follow-on and no new reward/scenario/algorithm search.

A relation-specific model is an alternative, but changes input channels/model
capacity and needs a larger matched-control redesign. Do not implement that
opportunistically. The present evidence does not favor either for performance.

The next required decision is approval of the explicit specimen-only candidate
graph amendment and the single bounded recovery under the remaining original
conditions. Until then: no interface bypass, model fit, scientific preflight,
new execution packet, remote action, holdout use or Stage E reopening.
