# Fresh-Seed Conditional GCN Value-MPC Study

Status: prospective engineering under Zhaowei's `全面的推进`; the complete
numerical execution decision below is separate and not inferred from direction.
One attempt; no scientific calls before committed package approval and locks.

## Decision And Evidence

Use GCN TD value-guided MPC as the candidate research architecture, not DDPG,
TD3 or an algorithm search. The completed native-return comparison failed its
persistent-change primary screen. Fast-fluctuation cost improvements versus H8
and historical frozen GCN were secondary, selected after seeing prior outcomes.
This new study prospectively focuses on that condition on fresh data, while
reporting stable and persistent conditions and every negative result. It does
not retroactively turn the previous primary failure into a success.

Questions: does the candidate improve fast-condition cost against H8 and its
own newly trained frozen initializer? Does graph message passing add value under
a shared-data recipe? Does TD differ from same-data MC policy evaluation? Is
benefit conditional on volatility, rather than inferred from significance in
one subgroup and nonsignificance in another? What labor/patient/compute trade-offs
accompany any improvement?

## Fixed Recipe

Five fresh model seeds and disjoint warmup/reference/test tapes; no seed or
checkpoint selection. Per block:

1. Create full-graph and self-only models from identical trainable tensors.
   Both use the existing 31 public features, width32, same two graph layers,
   pooling/head and3169 parameters. Self-only replaces normalized graph adjacency
   with identity. Its checkpoint must retain identity. This is not the historical
   differently-deep flat MLP control or a search over graph structures.
2. Collect48 shared plain-H8 worlds (16 per condition). Both learners train on
   exactly those records,32 observed-TD8 updates/world,1536/model. Sampling is
   paired. Loss is float64 with float32 parameters in both arms. This standardizes
   data for attribution but changes the historical24plain+24own-continuation
   recipe. It is NOT an exact replication of the historical ancestor training.
3. Seal both warm models. Freeze the full-graph model as the common native-tail
   parent. Fork graph TD and graph MC from graph warm weights, self-only TD from
   self-only warm weights. Reset Adam, sampling state and new-update counts.
   Own initialization hash and common tail-policy hash are separate bindings.
4. Collect24 new shared plain-H8 worlds (eight/condition). At index i, roots are
   i and i+24; candidate `(i + 8 * root_slot) % 16`. Clone the complete native
   state/RNG, execute the candidate for8 steps, then the frozen common graph
   parent to48 and unchanged zero-support settlement through64. Preserve the
   existing native collector, lag, receipts and fixed physics. No forecast-tail
   label collection is needed. All three learners share the same two branches
   and sampling indices. TD8 and MC targets/scale/hyperparameters are unchanged.
   Each gets32 updates/world,768/model. Branches are dependent continuations,
   not additional independent worlds. Native labels are synthetic, not clinical.
5. Seal all15 final models plus10 warm models before opening any test world.
   Five legacy GCN models are bound only as historical evaluation references;
   they never initialize or label the new learners.

Evaluate60 new worlds (four/condition/block) across seven frozen roles:
plain_h8, fresh_frozen, existing_frozen, graph_td, self_only_td, graph_mc,
plain_h16. Each trajectory has48 control plus16 settlement steps; test updates0.
Total240 warmup +120 label-reference +420 evaluation =780 main trajectories,
plus240 partial branches. Training datasets are shared within blocks, independent
across fresh blocks. Test tapes pair roles, not necessarily different conditions.

Reward weights, physical conditions, public inputs, candidate support, TD8,
gamma1,lr0.0003,batch64,32updates/world,gradient cap5 and scale1000000 stay fixed.
No new literature parameter ranges or sensitivity scenarios are mixed into this
confirmation package. Literature informs interpretation, not retrospective tuning.

## Complete Nonrefundable Budget

| Resource | Cap |
|---|---:|
| Main trajectories / partial clones | 780 /240 |
| Native steps: main + branch | 49920 +9720 =59640 |
| Native operations incl. main construction/reset, excluding clones | 61200 |
| Value updates: warm + native-tail | 15360 +11520 =26880 |
| Actor updates / test updates | 0 /0 |
| Optimizer example presentations | 1720320 |
| Module forwards incl. bootstrap/parent/evaluation | 46220 |
| Prediction steps: main + native-parent | 15482880 +1574400 =17057280 |
| Filter hypothesis transitions | 5964000 |
| Native-parent planning decisions | 4100 |
| Warm /final seals; legacy bindings | 10 /15;5 |
| Wall time incl. I/O, analysis, archival, failure reserve | 86400s (24h) |
| Disk raw + archive; file cap | 6GiB +6GiB;12000 |
| RSS /threads /workers | 4GiB /4 /1 |

Time owners: admission600s; shared warmup reference18000s; native references
including all branches25200s; all fits and seals7200s; evaluation24000s;
analysis/archive4800s; failure preservation6600s. No phase borrowing/refunds.
Per-job: warmup180s, reference with branches900s, H8eval180s, H16eval300s,
fit60s,seal60s. Full query chunks reserved before dispatch remain charged on
partial failure. Clones separately counted. TD/MC same data/update count is not
equal forward compute; H16 consumes twice the planning queries per decision.

Recent measured H8/H16 worlds were approximately35/66s and old label collection
included extra forecast work absent here. A rough9-12h scientific runtime is
plausible, not a guarantee; the full24h cap includes I/O and preservation.

## Prespecified Readout

Candidate graph_td. Joint primary development screen: in fast fluctuation,
positive absolute cost savings in all five blocks and descriptive95 lower bound
above0 against BOTH plain_h8 and fresh_frozen, with nonpositive mean extra
patient losses against those two controls in that condition. This is a fixed
development criterion, not a powered clinical or multiplicity-adjusted trial.
Report magnitude and uncertainty even when the screen fails. No requirement
that RL win everywhere; all stable/persistent and individual-world harms remain.

Use existing five-block, four-world-per-condition,2000-draw two-level descriptive
bootstrap. All21 role pairs x3conditions x5blocks, full component costs,
losses/deliveries, requests/committed/applied hours, latency and all compute.
Secondary graph_td/self_only_td isolates the use of neighbor messages within
this fixed common-parent, shared-data training pipeline, not universal GCN value.
Graph_td/graph_mc compares TD/MC estimators; MC itself is policy evaluation.
Fresh_frozen/legacy combines recipe and historical differences, not an isolated
causal training effect. Legacy contrasts are not new independent training evidence.

For both primary comparators, explicitly compute training-block differences in
mean cost savings: fast minus stable and fast minus persistent. Bootstrap blocks
jointly but sample the distinct condition worlds separately. Do not treat seeds
with the same replicate index in different conditions as paired patient worlds.
These interaction intervals are descriptive; subgroup significance alone does
not establish a volatility interaction. No extra model calls for diagnostics.

Read all780 raw main trajectories and240 branch raw files, reconcile costs,
patient conservation/settlement, actions, actual labor and test model/zero-update
bindings. Retain all outputs and failure evidence. E1 remains uncalibrated;
no real-patient data, deployment-time learning, clinical safety, full TD-MPC,
DDPG/TD3 superiority or guaranteed publication is claimed.

## Execution And Handoff

The numerical decision covers the whole single package. After its explicit
approval, commit implementation, literal authority, source/runtime/input/stream
locks; launch once in results/capacity_confirmation_20261006. The first counted
world is the scientific preflight. Necessary mock/zero-update tests are not a
new learning campaign. No redundant stage/start approval after the bound packet.

Terminal failure consumes the attempt: preserve, report and do not auto-retry,
extend samples/epochs, select checkpoints, change reward, reopen StageE or launch
a follow-on. Complete result readout, truthful manuscript update, local commit
and single archive on success. Continue the user's additive Dropbox artifact
authorization after a complete boundary, never overwrite different files;
verify destination bytes, distinguish local copy/cloud sync/collaborator access.
No implicit push/merge/public links/permission changes/external messages or
approval on Howard's behalf. Reuse the same gcn-rl monitor only after launch;
pause visibly at the terminal handoff, not an infinite research campaign.
