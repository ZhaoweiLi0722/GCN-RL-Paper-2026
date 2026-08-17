# Stage F1 Paired Online DDPG Protocol

Status date: 2026-08-17

Status: locked prospective development protocol. The user authorized Stage F1
development execution on 2026-08-17, conditional on all code, config, hash,
smoke, and host MPS gates passing. It does not authorize formal holdout access
or automatic formal confirmation.

## 1. Question

Stage F0 found a reproducible failure of the online critic to rank executed
integer patient-lot specimen actions. It did not find corrupted n-step replay
targets or a large short-to-long reward-sign reversal. Stage F1 therefore asks
one narrow question:

> When the DDPG critic receives a direct paired finite-horizon target for the
> executed specimen action relative to MDL-2, do online updates improve the
> final policy over its tensor-identical frozen pretraining checkpoint?

The primary estimand is candidate final minus candidate frozen pretrain. Final
minus MDL-2 is not evidence of online learning because most of the established
benefit already exists before online interaction.

## 2. Single candidate

For a behavior transition at state `s`, let `a_b` be the executed action and
`a_0` the projected MDL-2 action. A paired target is collected only when their
specimen slices differ after the same 120-patient integer-lot quantization used
by the critic and environment.

Two deep-copied environments start from the exact same pre-action state and RNG
state. One executes `a_b`; the other executes `a_0`. Both then follow MDL-2 for
the remaining steps of the existing four-step n-step horizon. The target is

```text
A_pair(s, a_b) = sum[t=0..3] gamma^t
                  (r_behavior,t - r_anchor,t).
```

The candidate adds one pointwise critic loss,

```text
L_pair = MSE(Q(s, a_b) - Q(s, a_0), reward_scale * A_pair).
```

Its fixed weight is `3.0`, equal to the already locked online teacher-ranking
weight. This value is not tuned. The Bellman loss and frozen teacher-ranking
loss remain present and unchanged.

## 3. Paired control

Both arms use support-matched structured specimen exploration with the locked
Stage C3 legal set `{MDL-2, -0.05, +0.05, -0.10, +0.10}` and selection
probability `0.20`. This gives both arms the same behavior support that Stage C3
already verified. The control has `online_paired_advantage_critic.enabled=false`;
the candidate changes only that field to `true` at the episode-0 fork.

The following remain tensor- and contract-matched at the fork:

- actor, critic, targets, optimizers, and frozen reference actor;
- offline replay, environment state, ordinary OU state, and every RNG state;
- actor objective, deterministic DDPG update, gate, quantization, residual
  scale, reward, self-imitation, and update schedule;
- persistent-hotspot scenario assignment and 100-episode budget; and
- GCN/flat model capacity and parameter-matching rule.

Counterfactual rollouts operate only on deep copies. They do not advance the
live environment or any live policy RNG.

## 4. Fresh development design

- Algorithms: GCN residual DDPG and parameter-matched flat residual DDPG.
- Training seeds: 60, 61, and 62.
- Scenario assignment: seeds 60/61/62 map once to persistent hotspot clusters
  1/2/3.
- Arms: paired control and paired-advantage candidate.
- Online episodes: 100 per run, strictly serial.
- Checkpoints and full states: every five episodes.
- Curves: pretrain, episode 10, 25, 50, 75, and 100.
- Execution-only evaluation seed: 96600000 with one replication.
- Development evaluation seed: 96700000 with 50 paired replications.
- Bootstrap seed: 96750000.
- Formal holdout 91100000 and every prior development stream are forbidden.

The complete campaign has 12 training jobs, 1,200 online episodes, and 72
fixed-checkpoint evaluations. No checkpoint is selected from the curve.

## 5. Integrity gates

Before full execution:

1. paired-return, quantization, replay, fork, GCN, and flat tests pass;
2. all pre-existing focused DDPG and training-state tests pass;
3. a CPU smoke produces at least one finite paired target and nonzero critic
   gradient while the control produces zero paired targets;
4. zero-trajectory MPS construction passes with fallback disabled;
5. control/candidate episode-0 payloads are identical outside declared contract
   metadata; and
6. source, configs, teacher, and launch assets are hash-locked.

During execution require finite metrics, nonzero routing and online updates,
all five structured options in every completed run, paired-target coverage in
the candidate, no paired targets in the control, exact first-step cloned reward
reproduction, GCN/flat parameter gap at most 1%, and no duplicate process.

## 6. Prospective decision

An online-learning claim advances only if all primary GCN conditions hold:

1. pooled final-minus-frozen mean total cost is favorable and its paired 95%
   bootstrap interval is wholly below zero;
2. at least two of three training seeds are favorable;
3. all locked clinical noninferiority checks pass;
4. candidate final performance does not regress relative to the matched
   control, operationalized prospectively as a nonpositive paired mean total-
   cost difference plus all clinical noninferiority checks; and
5. the result is not driven by missing/invalid paired targets or a numerical
   failure.

Candidate-minus-control, checkpoint-curve AUC, paired-target sign accuracy, and
matched-flat difference-in-differences are supporting mechanism analyses. They
cannot rescue a failed final-minus-frozen primary gate.

If the gate passes, a separate fresh confirmation protocol may be reviewed. If
it fails, online DDPG attribution is closed and the paper retains the existing
transparent statement that online contribution is not established.
