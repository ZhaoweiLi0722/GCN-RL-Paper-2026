# Results-Led GCN+RL Algorithm Selection

2026-10-06. Entry commit: 6b39efa3f45c4b5baf5ae853cd9d36de80983b36.
Status: proposed design, NOT frozen, launch-ready or scientifically authorized.
No new training, model inference, environment or optimizer execution occurred.

## Direction And Correction

Zhaowei requested an outcome-driven search for the best supported GCN+RL
configuration, including alternatives beyond DDPG and TD3. Keep the question
and comparators fixed while improving methods. Prior implementation effort or
algorithm reputation must not determine the winner.

Exploring TD-value-MPC did not prove it beat GCN-DDPG. Stronger historical
evidence for a DDPG controller package did not prove DDPG was the best algorithm.
Withdraw both as cross-family rankings; preserve their actual scoped results.
Self-only means identity neural adjacency, not dev-only, an isolated facility,
or the historical flat MLP. Physical network links remain unchanged.

## Existing Evidence Is Not A Common Leaderboard

| Record | Supported finding | Important limit |
| --- | --- | --- |
| E1: formal routing | GCN-DDPG package costs 0.658474% less than MDL-2 and 0.321357% less than flat. | No isolated message-passing, incremental online-update or cross-family superiority claim. |
| E2: earlier five-seed replenishment | GCN-DDPG costs 0.015112% less than matched GCN-TD3 in that study. | Small task-specific effect, not general DDPG dominance. |
| E3: TD3 routing development | Positive graph/anchor comparisons; final-versus-frozen increment unestablished. | Not a direct victory over separately evaluated DDPG. |
| E4: five-block value-MPC | Persistent graph/flat absolute-cost screen passed; graph/plain-MPC primary failed. | Not broad planner superiority or an isolated graph-edge effect. |
| E5: native-return value-MPC | Fast H8 mean paired saving 2.470%, favorable descriptive absolute-cost interval; persistent primary failed. | Exploratory condition-specific signal, not a DDPG/TD3 comparison. |
| E6: fresh conditional value-MPC | Fast H8 mean paired saving -1.445%; primary and separate graph/TD advantages unestablished. | Changed warmup, not exact E5 replication or proof DDPG wins. |

Tasks, denominators, training recipes and/or worlds differ. Do not sort these
percentages into a leaderboard. This scoped review identified no matched
five-family comparison. Literature motivates candidates, not application winners.

## Proposed Fixed Benchmark

Recommendation, not a silently adopted scientific contract: use the existing
synthetic support-hour task so the completed H8/H16 and value-MPC controllers
can supply strong, stable references. Preserve reward, physics, original three
conditions, public information and feasible support actions. Keep common MDL-2
routing/purchasing operations in EVERY arm. This studies support-hour allocation,
not the original full-routing controller claim; keep that evidence separate.

- Fixed primary reference: plain H8, including its heuristic terminal cost.
- Fixed secondary compute reference: plain H16, including its higher query cost.
- Five candidate families: GCN-DDPG, GCN-TD3, GCN-SAC, GCN-PPO, GCN TD-value-MPC.
- Keep a prospectively specified previous-configuration reference on the SAME
  worlds to measure iteration. Never select its best old test seed/checkpoint.
- Do not drop an inconvenient comparator or preselect a winning family.

The actor-critic arms must share public feature construction, graph definition,
action decoding/feasibility and access to any common warm-start data. Comparing
AFD-pretrained residual DDPG to scratch SAC/PPO is not a backbone-only test.
Extra critics and PPO on-policy sampling remain real algorithm differences.
Value-MPC/actor-critic is a complete-controller contrast, not a backbone-only
ablation. Planning and native clone data must be included in total resource cost.

## Finite Selection Rules

1. Predeclare finite recipe/tuning allowances for every family. No unlimited
   search until a positive result appears, automatic expansion or hidden failures.
2. Use fresh training/development streams and paired within-stage evaluation
   worlds. Do not use the old formal holdout to tune or select candidates.
3. Compare all five families in development; carry at most two finalists to
   fresh-seed confirmation using a rule fixed before development outcomes.
   Preserve failed/eliminated arms. Do not rank on final-test outcomes then tune.
4. Match native-interaction and tuning allowances, not identical optimizer counts
   or rollout histories. Charge branch transitions, warm-start production and
   model queries; report wall time, memory, latency and all computation separately.
5. Fix full cost and simulated patient loss. Recommended default: minimize cost
   subject to no allowed patient-loss deterioration. Statistical rules, margins
   and practically meaningful cost thresholds must be specified before launch.
   This is not clinical noninferiority. Retain Pareto alternatives when objectives
   trade off, rather than inventing one universally best method.
6. Prospectively include same-algorithm self-only and applicable frozen/no-RL
   controls for finalists. Retrain matched ablations; do not corrupt a trained
   GCN adjacency. A random frozen actor is not a competent operational baseline.
7. Seal final models before tests; zero test updates or checkpoint selection.
   Specify paired inference, training-seed uncertainty and multiplicity handling
   in the full protocol. An inconclusive ranking remains inconclusive.

Best means best supported within the task, candidate set and resource budget,
not globally optimal. A package win does not prove separate GCN and RL increments.
Synthetic-only and missing E1 calibration limits remain unchanged.

## Concrete Remaining Work

Source inspection confirms generic GCN-SAC/PPO classes and registry entries.
They learn from scratch with the general facility-action interface. They are NOT
ready matched support-hour integrations. The completed value runner has no actor.

Required work is one shared public-feature/support-action adapter and bounded
multi-family runner: preserve pre-projection stochastic action/log-probability
bookkeeping, use consistent off-policy reward/replay meanings, separate selection
stages, seal models and account for all scientific calls. Reuse existing learners
and checks; test changed interfaces and required entry paths only. No extra
detached toy/audit campaign.

Then present ONE complete numerical package: exact recipes, development/finalist
allocations, seeds, native/clone/planning/forward/update counts, phase/owner time,
memory/disk/file caps, failure handling and handoff. No defensible wall-time
estimate exists yet: the previous 9.135-hour value-MPC run is not an actor-critic
benchmark. Do not recycle its caps or call this launch-ready. One whole-package
approval is required before science; no redundant routine per-phase approvals.

## Sources

- E1: specs/2026-09-29-formal-crossed-audit/readout.md and specs/2026-09-29-formal-graph-contract/readout.md.
- E2/E3: paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks/main.tex, formal_drift_pairs and routing-primary backbone discussion.
- E4: specs/2026-10-04-value-mpc-comparison/recovery1/terminal-readout.md.
- E5: specs/2026-10-05-native-return-value/terminal-readout.md.
- E6: specs/2026-10-06-conditional-value-confirmation/protocol.md and terminal-readout.md.
- Interfaces: src/rl/agents.py; src/models/gcn_sac.py; src/models/gcn_ppo.py; src/rl/capacity_confirmation_runner.py; experiments/configs/capacity_confirmation_20261006.json.
- TD3: https://proceedings.mlr.press/v80/fujimoto18a.html.
- SAC: https://proceedings.mlr.press/v80/haarnoja18b.html.
- PPO: https://arxiv.org/abs/1707.06347.

Sources were inspected by the assistant. No new human verification, registration,
statistical power, numerical approval or completed comparison is claimed.
