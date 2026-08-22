# Online DDPG Problem Redefinition Review

Status date: 2026-08-18

## Technical summary

The low-cost redesign screen does **not** support moving the current routing
problem onto its continuous reagent or capacity controls. Those actions execute
reliably, but they do not provide independently replicated, clinically
noninferior cost headroom. The usable local headroom remains concentrated in
patient-indexed specimen routing, whose integer-lot execution is fundamentally
discrete.

This result does not invalidate AFR-GCN-DDPG or its formal endpoint result.
AFR-GCN-DDPG remains the primary method because it is better than routing MDL-2
and parameter-matched flat DDPG on the locked formal holdout. It does narrow the
mechanistic claim: the demonstrated gain should be attributed to graph-aware,
advantage-filtered offline pretraining and anchored residual control, not to an
incremental benefit from the tested online DDPG updates.

No further continuous-control audit, online training, or formal confirmation is
authorized. Making online DDPG scientifically central would require adding a
genuinely continuous operational decision to the simulator, which is a new
modeling study rather than a repair of the current paper.

## Continuous controls execute but lack safe headroom

The corrected H0 screen used persistent-hotspot assignments; H1 repeated the
same frozen protocol on abrupt shift, regional drift, and compound stress.
Each screen covered 27 states: nine fixed decision points for each of three GCN
pretraining seeds. At every state it compared MDL-2 with twelve fixed residual
actions under three discovery and five independent validation replications over
the remaining episode.

| Action group | Persistent execution | Persistent stable best | Persistent validated opportunity | Persistent prospective success | Nonstationary execution | Nonstationary stable best | Nonstationary validated opportunity | Nonstationary prospective success |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Specimen transfer | 88.9% | 29.6% | 55.6% | 40.7% | 88.9% | 14.8% | 55.6% | 37.0% |
| Reagent transfer | 100.0% | 0.0% | 3.7% | 0.0% | 100.0% | 0.0% | 3.7% | 0.0% |
| Reagent plus capacity | 100.0% | 0.0% | 3.7% | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |

`Execution` is the fraction of non-anchor inputs that changed the corresponding
physical flow. `Validated opportunity` is the fraction of states with at least
one validation action saving at least 1 million modeled objective units while
not reducing completion service, increasing patients lost, or increasing
manufacturing ineligibility. `Prospective success` applies the same validation
test to the action selected using discovery replications only. `Stable best`
additionally requires the same non-anchor action to be best and clinically
noninferior in both streams.

The combined reagent/capacity ordering was reproducible: discovery-validation
best-action agreement was 92.6% in both screens, with pairwise sign agreement
of 94.8% and 94.7%. That apparent learnability is not useful because almost no
large cost improvement survived the clinical guardrails. Reagent-only actions
were clinically safer at smaller thresholds but were too small and unstable at
the prespecified 1-million threshold. A more sophisticated critic cannot create
an objective opportunity that the environment does not expose.

## The action-space mismatch is now the primary diagnosis

Three observations jointly explain why online DDPG has not added endpoint
value in the current formulation:

1. The formal performance gain already exists at the frozen, offline-pretrained
   policy; final online updates do not improve it reliably.
2. The clinically useful local alternatives are mostly specimen-routing
   choices, but those choices are projected and quantized into integer patient
   lots.
3. The resource actions that preserve continuous geometry either have weak
   objective leverage or exchange cost for clinical harm.

This is a problem-definition mismatch, not simply a learning-rate or episode
budget problem. More online episodes, larger actor updates, distributional
critics, or uncertainty penalties still optimize the same action surface. They
may change estimates without supplying a stable, safe policy-improvement
direction.

## Redesign candidates and decisions

| Candidate | Scientific value in the current simulator | Decision |
| --- | --- | --- |
| Direct DDPG on reagent residuals | Continuous and executable, but validated material opportunity is 1/27 states and prospective success is 0/27 in both screens. | Reject for this paper. |
| Direct DDPG on reagent plus capacity | Action ordering is stable, but clinically noninferior material opportunity is 1/27 persistent states and 0/27 nonstationary states. | Reject for this paper. |
| Constrained or risk-sensitive DDPG | Better constraint handling cannot recover the cost headroom already removed by the clinical guardrails. | Do not train without a new positive headroom screen. |
| Distributional or uncertainty-aware critic | Could represent stochastic returns better, but does not solve absent continuous control leverage. | Methodologically unmotivated now. |
| Proto-action or hybrid discrete-continuous controller | Better matches specimen execution, but the continuous branch has no demonstrated role and the corrected specimen screen still has unstable top actions. | A separate method study, not a current-paper extension. |
| Add continuous production, procurement, overtime, or staffing decisions | Could create the smooth, persistent control authority that DDPG requires. It changes simulator assumptions, data needs, comparators, and validation. | Only credible future DDPG route; treat as a new project. |

The corrected 27-state specimen comparator is informative but not a recovered
Stage G1 result. It shows substantial headroom and only 59.3% discovery-
validation best-action agreement in both scenario screens, below the former
70% gate. A full 156-state legal-action recovery was deliberately not launched
because the cheaper prospective screen already failed to authorize escalation.

## Scenario reconstruction correction

The original F0, G0, and G1 post-hoc mechanism audits passed each run's stored
`env` directly to `build_env`. Multiscenario training snapshots keep the
pretraining reference environment there (`routing_nominal_history`) and store
the actual online assignment under `multi_scenario_training.scenarios`.
Consequently, those audits are reproducible nominal-history diagnostics, but
their persistent-hotspot scenario labels were incorrect.

This defect does **not** affect Stage E formal endpoint evaluation, Stage F1
training, or Stage F1 checkpoint-curve evaluation, which reconstruct scenarios
from the benchmark plan. It does affect scenario-specific mechanistic claims in
F0/G0/G1. Their files and hashes remain immutable; correction notices mark
their interpretation as superseded. The H0/H1 implementation explicitly
reconstructs the benchmark-plan scenario and asserts that every output row's
built scenario matches its declaration.

## Publication recommendation

For the current EAAI manuscript:

1. Retain AFR-GCN-DDPG as the proposed primary method and report the formal
   graph-versus-MDL-2 and graph-versus-flat advantages.
2. Describe the deployed controller as an offline-pretrained, MDL-2-anchored
   graph residual policy implemented with a DDPG backbone.
3. State explicitly that the tested online actor-critic phase did not establish
   incremental endpoint improvement.
4. Present F1 and the corrected action-geometry screens as negative attribution
   ablations and limitations, not as failed attempts hidden from the narrative.
5. Do not reopen the formal holdout or start another tuning campaign for this
   paper.

A concise claim is:

> The graph-aware advantage-filtered residual controller improved the locked
> routing baselines, while prospective attribution experiments did not
> establish an additional endpoint benefit from the tested online DDPG updates.
> Frozen action-geometry audits further showed that the remaining clinically
> useful control headroom is concentrated in discrete specimen-routing choices
> rather than the simulator's continuous resource-transfer channels.

## Evidence and reproducibility

- Corrected H0 summary SHA256:
  `f533452b92d169796bceaa6966ad7925c67b59de5cfa6edf39ef6d29ec025f74`
- Corrected H0 rows SHA256:
  `4ce85c796da6021f32e09f5014c11e738b027cd05b04c02f84ba9f3866ac0121`
- H1 summary SHA256:
  `0dbb140148ae96755a94c24f2e74e2a1d644b6b909bbf78ffc31df370bef401e`
- H1 rows SHA256:
  `1529a0242e6a3d0ba4464226f2dad0c9a1ada652b625574a2c55f4a737b3d046`
- Curated evidence:
  `experiments/evidence/patient_indexed_specimen_routing_ddpg_continuous_control_redefinition/`
- Audit implementation:
  `evaluation/audit_ddpg_continuous_control_headroom.py`

The screens are descriptive and prospective development diagnostics. They do
not estimate a causal treatment effect, prove that all possible continuous
actions are useless, or establish performance under newly modeled operational
decisions.
