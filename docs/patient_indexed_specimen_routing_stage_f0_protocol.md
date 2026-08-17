# Stage F0 Online DDPG Identifiability Protocol

Status date: 2026-08-17

Status: locked read-only development diagnosis. This protocol does not
authorize training, checkpoint selection, formal confirmation, or formal
holdout access.

## 1. Objective

The publication claim would be stronger if the online DDPG update improved on
its own tensor-matched frozen AFR pretraining checkpoint. Existing campaigns
show that the actor and critic update, the policy drifts, and specimen routing
changes, but final-minus-frozen performance is not established. Stage F0 asks
whether that negative result is caused by one identifiable mismatch that can
support one prespecified correction.

The target claim is deliberately narrow:

> Online DDPG learns a stable improvement over its own frozen pretraining
> policy on the executed integer patient-lot specimen-routing manifold.

Final versus MDL-2 cannot establish this claim because most of that effect is
already present at the frozen pretraining checkpoint.

## 2. Existing evidence and exclusions

The immutable Stage C3 headroom audit evaluated 156 frozen-policy states and
independently validated 58 lower-cost clinically noninferior local actions,
including 51 specimen corrections. Mean validated single-decision improvement
was approximately 4.10 million cost units. Structured exploration also covered
all five legal options at the intended rate. Reachable action headroom and
behavioral coverage therefore exist.

Stage F0 must not repeat the following as a sole explanation or intervention:

- additional online episodes;
- TD3 in place of DDPG;
- teacher-support filtering;
- online-boundary critic reset or actor warmup;
- persistent-hotspot exposure alone;
- wider structured exploration alone; or
- broad hyperparameter search.

All of these have already been tested in bounded development campaigns without
passing an online-attribution gate.

## 3. Locked diagnostic questions

### 3.1 Replay target consistency

For each unchanged control run and each GCN/flat seed 50-52, reconstruct from
the immutable final replay buffer:

1. exact one-step behavior-minus-MDL-2 reward;
2. the persisted four-step discounted training target; and
3. the discounted remainder of one-step relative rewards in the same behavior
   episode.

The third quantity is not a causal value of the first action because future
terms follow the realized behavior trajectory. It is used only to test temporal
sign consistency. The audit also checks the exact reconstruction of every
persisted four-step target and discount multiplier.

### 3.2 Legal-action critic ranking

For GCN seeds 50-52, reproduce nine prespecified states from a fresh
frozen-pretrain trajectory in each locked persistent-hotspot scenario. At
pretrain, episode 25, and final, compare MDL-2 with specimen corrections
`+/-0.05` and `+/-0.10`.

At each state and checkpoint:

- obtain critic values after the same integer patient-lot quantization used in
  training;
- obtain the critic action gradient through the actor's straight-through
  estimator;
- evaluate every legal action with paired CRNs at horizons 1, 4, and the
  remaining episode; and
- follow the checkpoint actor being audited after the first candidate action.

The audit reports within-state pairwise ranking, top-action accuracy, gradient
direction accuracy, quantized action distinctness, and remaining-horizon
headroom by seed and checkpoint.

## 4. Immutable inputs and fresh streams

- Stage C3 artifact inventory SHA256:
  `f07f69c82ce37bdc5f687389f80c5860c2c9df4483db190cff9bded499001834`.
- Stage C3 comparison SHA256:
  `30d1a39785a8e4bc9f17d68e6cf615f4e29ab5b2a64284e1105601bc87fecda4`.
- Training seeds: 50, 51, and 52.
- Fresh trajectory seed base: 95900000.
- Fresh paired-rollout seed base: 96200000.
- Formal holdout 91100000 and every prior development stream through 95850000
  are forbidden.

The full artifact inventory is checked before use, every selected input is
hashed, and all selected hashes are checked again after the audit.

## 5. Prespecified F0 gate

Stage F1 design is justified only if all conditions hold:

1. At least two primary GCN seeds have either of the following reproducible
   mechanism branches:
   - at least 25% negative behavior-trajectory remainder among the positive
     samples selected for self-imitation; or
   - final remaining-horizon critic pairwise accuracy at most 0.60 or gradient
     top-action accuracy at most 0.40.
2. The mechanism-branch seeds retain at least 10% states with a legal action improving cost by
   at least 1 million.
3. At least one mechanism-branch seed is also worse at final than frozen
   pretraining in the locked Stage C3 evaluation.
4. The correction is fixed before any new training result is viewed.

A smoke run cannot pass this gate. A passing full audit authorizes protocol
review only, not execution.

## 6. Only eligible Stage F1 correction

If F0 passes, the single eligible candidate is a directly paired
finite-horizon counterfactual-advantage critic on executed legal specimen
actions versus MDL-2. The actor, deterministic DDPG update, exploration, gate,
residual scale, scenario distribution, model capacity, seeds per arm, and
100-episode budget must remain unchanged.

If F0 fails, no further online DDPG training experiment is justified by this
mechanism. The current manuscript remains valid with transparent
negative/inconclusive online attribution.

## 7. Reproduction

```bash
PYTHONPATH=. python -m evaluation.audit_ddpg_online_identifiability \
  --config experiments/configs/patient_indexed_specimen_routing_ddpg_online_identifiability_f0.json \
  --artifact-root /path/to/repository-with-full-stage-c3-results \
  --component all
```

Use `--smoke` with a fresh `--output-root` for the one-seed, one-state preflight.
