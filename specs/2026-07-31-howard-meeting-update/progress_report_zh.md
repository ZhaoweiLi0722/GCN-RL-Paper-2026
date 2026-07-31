# Howard Meeting Progress Update

- 会议时间：2026-07-31 10:30
- 汇报人：Zhaowei
- 建议时长：8--10 分钟

## 一句话结论

我们现在已经有证据表明，基于 MDL-2 的 graph-aware anchored residual
controller 在具有 patient condition、真实 geography、transfer delay 和
regional demand shift 的环境中可以显著优于 MDL-2 和 matched flat policy。
但是，目前还需要把两件事分开：

1. graph representation 的优势已经在 regional setting 中得到较强支持；
2. online actor--critic RL 相对于 frozen pretrained policy 的额外贡献仍未达到
   pooled statistical significance，正在用五场景 zero-shot evaluation 进一步检验。

因此，当前最稳妥的 proposed method 仍然是
**AFR-GCN-DDPG**，TD3 是 matched backbone/stability ablation，而不是替代主方法。

## 1. 在 Howard 上次 PR 基础上完成的工作

Howard 的 PR 提供了重要基础：

- patient-condition simulation environment；
- expiry-aware inventory、forecast、disruption 和 stress scenarios；
- GCN-DDPG、GCN-TD3、GCN-SAC、GCN-PPO 以及 flat baselines；
- uMYO、fMYO 等公平 heuristic baselines；
- manuscript 的大规模重写。

我们在此基础上进一步完成了：

- 把 20 clinics 的真实 geographic structure 纳入环境；
- 根据距离引入 1--3 个周期的 transfer lead time，而不再统一设为 1；
- 显式建模 reagent 和 idle-bioreactor 的 delayed transfer；
- 保持 patient specimen identity-bound，禁止 specimen pooling/transfer；
- 将 patient deterioration、manufacturing-period ineligibility、therapy discard
  和 bioreactor cleaning 纳入统一环境；
- 将 demand shift 从单一 global multiplier 扩展为 regional drift、abrupt
  regime shift 和 compound regional stress；
- 建立 MDL-2 anchor、matched flat、GCN、DDPG/TD3 family 和 clinical
  noninferiority 的 paired common-random-number evaluation；
- 在 RTX 4090 上完成可复核的 CUDA training/evaluation pipeline。

## 2. Proposed method 的当前定义

### AFR-GCN-DDPG

AFR 表示 **advantage-filtered residual**，是完整 controller 的名称，不是一个
新的 actor--critic algorithm。

它包括：

1. **MDL-2 anchor**：提供透明、稳定的基础决策；
2. **GCN state encoder**：编码 clinic node state、geographic edges、distance、
   lead time、transfer cost、patient risk 和 regional pressure；
3. **DDPG residual actor**：只学习对 anchor action 的 bounded correction；
4. **advantage-filtered distillation (AFD)**：只有在 common-random-number
   look-ahead 中优于 anchor 的 correction 才保留为 teacher target；
5. **feasibility projection and deployment safeguards**：保证库存、capacity、
   identity、connectivity 和 clinical constraints 可行。

这里 AFD 是 AFR 内部的训练步骤，不是另一种 deployed algorithm。

### Network extension

早期版本只修正 replenishment，graph information 很难转化为真正的网络动作。
Network AFR-GCN-DDPG 增加了：

- node-level signed replenishment correction；
- edge-level reagent transfer correction；
- edge-level idle-bioreactor transfer correction；
- specimen transfer 永久 mask 为 0。

这使 GCN 可以直接利用 geographic and regional information。

## 3. 本周已验证结果

### 3.1 Replenishment-only AFR-GCN-DDPG：300 episodes x 5 seeds

场景是 integrated patient condition + geography + demand-prior drift。
每个 seed 使用 100 个独立 paired replications，共 500 对。

AFR-GCN-DDPG 相对 MDL-2：

- mean cost difference：-$207,485；
- relative cost gap：-0.016360%；
- paired 95% CI：[-$332,641, -$91,367]；
- 323/500 paired wins；
- completion service：+0.000340；
- eligibility：+0.000211；
- average waiting time：-0.000612；
- patients lost：-2.32；
- at-risk unserved patients：-2.55。

这个结果说明 proposed residual DDPG 能够稳定优于 misspecified MDL-2
anchor，但绝对成本改善仍然很小。

Matched flat residual DDPG 也优于 MDL-2：

- relative cost gap：-0.013216%；
- 95% CI：[-$286,899, -$46,230]。

GCN 相对 flat 的点估计进一步降低成本 0.003145%，但 95% CI
[-$89,702, $14,025] 跨过 0。因此，在这个 replenishment-only setting 中，
可以证明 residual learning 的价值，但不能单独证明 graph representation 的价值。

### 3.2 DDPG 与 TD3 的 matched comparison

同一 300 episodes x 5 seeds protocol 下：

- AFR-GCN-TD3 相对 MDL-2：-0.001248%；
- 95% CI：[-$142,607, $99,437]，跨过 0；
- AFR-GCN-DDPG 相对 TD3：-0.015112%；
- DDPG-minus-TD3 95% CI：[-$234,659, -$148,401]；
- 五个 seed 的 mean result 全部支持 DDPG。

结论是 DDPG 应继续作为 canonical proposed backbone。TD3 能稳定部署 residual，
但在这个 sparse-correction problem 中，其 conservative twin-critic target
可能抑制了一部分正向 correction signal。

Pure GCN-DDPG、GCN-TD3、GCN-SAC 和 GCN-PPO 从头学习完整 policy 时表现明显更差。
这说明真正有效的是 strong operational anchor + graph-aware residual learning，
而不是简单增加一个更复杂的 RL backbone。

### 3.3 Network AFR-GCN-DDPG：regional abrupt-shift result

场景同时包含：

- patient condition；
- 20-clinic geography；
- distance-based transfer delay；
- regional demand heterogeneity；
- abrupt regional regime shift。

三 seed、每 seed 100 个 paired holdout replications 的结果：

AFR-GCN-DDPG 相对 MDL-2：

- relative cost gap：-1.1337%；
- mean cost difference：-$25.39M per 52-week episode；
- 95% CI：[-$29.76M, -$20.95M]；
- 237/300 paired wins；
- completion service：+0.001357；
- patients lost：-0.70，达到 aggregate noninferiority。

AFR-GCN-DDPG 相对 parameter-matched flat residual：

- relative cost gap：-1.1166%；
- mean cost difference：-$25.01M；
- 95% CI：[-$29.35M, -$20.61M]；
- completion service：+0.002836；
- patients lost：-9.86，95% CI [-15.24, -4.31]。

这是目前最强的 graph-specific evidence：当 action space 真正包含 edge-level
resource transfer，而且 demand shock 具有 geographic structure 时，GCN 显著优于
parameter-matched flat model。

需要诚实说明：这个 checkpoint 的主要价值来自 advantage-filtered teacher
distillation；早期 offline DDPG/TD3 fine-tuning 并不稳定。因此，这一结果证明了
graph-aware policy 的价值，但还不能单独证明 online RL update 的价值。

### 3.4 RTX 4090 conservative online TD3 screen

我们完成了 GCN 和 matched flat 各 3 个 seeds、每个 seed 100 training episodes
的 conservative TD3 screen。每个 run 都有：

- 5,200 update calls；
- 5,073 actual updates；
- 2,286 actor updates；
- finite diagnostics；
- nonzero actor drift；
- no Traceback、NaN、OOM 或 CPU fallback。

Final GCN policy 相对 MDL-2 的 per-seed cost gaps：

- seed 0：-0.051936%；
- seed 1：-0.139249%；
- seed 2：-0.119193%。

Pooled GCN 相对 MDL-2：

- mean cost difference：-$2.544M；
- 95% CI：[-$3.872M, -$1.131M]；
- clinical noninferiority：3/3 seeds 通过。

Matched flat 相对 MDL-2 的三个 seed 均变差：

- +0.049693%、+0.040470%、+0.097624%；
- clinical noninferiority：0/3 seeds 通过，主要由 patients lost 导致。

Final GCN 相对 flat：

- mean cost difference：-$4.083M；
- relative gap：-0.165951%；
- 95% CI：[-$5.633M, -$2.436M]；
- completion service：+0.001203；
- patients lost：-7.05。

这表明最终 graph TD3 controller 显著优于 MDL-2 和 matched flat。

不过，final checkpoint 相对 frozen-pretrain checkpoint 的 pooled difference 是：

- GCN final-minus-frozen：-$0.350M；
- 95% CI：[-$1.032M, $0.155M]；
- GCN-minus-flat difference-in-differences：-$0.485M；
- 95% CI：[-$1.489M, $0.173M]。

两者方向有利，但 confidence interval 都跨过 0。因此目前最准确的表述是：

> Graph-aware final controller 的优势成立；online TD3 update 的独立增量贡献尚未
> 在跨 seed pooled inference 中得到确认。

## 4. 当前正在进行的实验

RTX 4090 正在运行 evaluation-only five-scenario zero-shot campaign。它不会重训，
而是复用现有的六个 final checkpoints 和六个 frozen-pretrain checkpoints。

五个场景是：

1. nominal history；
2. severe global demand drift；
3. regional drift；
4. abrupt regional regime shift；
5. compound regional stress。

Protocol：

- GCN 和 matched flat；
- seeds 0、1、2；
- 每个 scenario/seed 100 个 paired holdout replications；
- final 与 frozen 使用完全相同的 CRN keys；
- 52-week horizon；
- fixed deployment scale 0.1；
- clinical noninferiority 同步检验。

之前的中断发生在 MDL-2 anchor 的一个 info-only patient-risk counter，不在
GCN/TD3 policy inference、environment transition、reward 或 action 中。修复仅恢复
reporting field，并对每行记录 recovery diagnostic；实验已按预注册 protocol 恢复。

这轮结果将回答三个决定性问题：

1. graph policy 相对 MDL-2 和 flat 的优势能否跨场景保持；
2. final policy 相对 frozen pretrain 是否有稳定增益；
3. online TD3 attribution 应归类为 Strong、Scenario-specific，还是 Not established。

## 5. 当前可发表的结论与尚不能声称的结论

### 已有证据支持

- anchored residual learning 在 demand-prior misspecification 下可以小幅但稳定地
  优于 MDL-2；
- DDPG 在 matched 300 x 5 comparison 中优于 TD3，因此适合作为 proposed
  backbone；
- 当 residual action 包含 geographic edge transfer 时，GCN 显著优于
  parameter-matched flat policy；
- 在 regional drift 下，final graph TD3 controller 显著优于 MDL-2 和 flat，
  并通过 clinical noninferiority。

### 目前不能过度声称

- 不能说 RL 在所有场景下普遍优于 heuristics；
- 不能把 teacher-distillation gain 全部归因于 actor--critic RL；
- 不能声称当前 checkpoint 已经具备 cross-scenario robustness；
- 不能把当前 3-seed screen 当作最终 5-seed x 500-replication evidence。

## 6. Zero-shot 结果出来后的决策

### 如果 online TD3 attribution = Strong

- 保留 AFR-GCN-DDPG 为主方法；
- 将 AFR-GCN-TD3 作为 strong matched backbone ablation；
- 增加 seeds 3、4；
- 对最终方法运行 300 episodes x 5 seeds；
- 进入 500 paired Monte Carlo replications 和完整 baseline table。

### 如果 attribution = Scenario-specific

- 识别 TD3 真正产生增益的 scenario；
- 做 scenario-conditioned 或 multi-scenario training；
- 保留 nominal scenario 作为 no-harm/noninferiority case；
- 用 regional/compound scenarios 展示 adaptive graph control 的价值。

### 如果 attribution = Not established

- 不继续盲目增加相同训练预算；
- 回到 GCN-DDPG 主线，使用 multi-scenario replay、persistent AFD regularization
  和 conservative advantage constraint；
- 把 paper 的现阶段结论写成：
  graph representation and residual control improve deployable decisions,
  while the incremental value of online actor--critic adaptation remains limited。

## 7. 建议向 Howard 提出的四个问题

1. 是否同意继续把 AFR-GCN-DDPG 作为 proposed method，把 TD3 放在 principal
   backbone ablation？
2. 是否同意把核心 paper claim 分成两层：graph representation advantage 已成立，
   online RL attribution 由 final-vs-frozen evaluation 单独检验？
3. 最终 paper 的 primary robustness scenarios 应重点放 regional drift、abrupt
   shift 和 compound stress，还是还需要把 severe global drift 放入主表？
4. 如果 online TD3 的增量仍不显著，我们应优先做 GCN-DDPG multi-scenario
   fine-tuning，还是将 contribution 定位为 graph-aware residual policy learning
   加 patient/geography testbed？

## 8. 建议口头讲稿

Howard，这周我们主要把三件事分开验证了：residual learning 有没有价值，
graph representation 有没有价值，以及 online RL updates 有没有额外价值。

第一，replenishment-only AFR-GCN-DDPG 在 300 episodes、5 seeds、500 paired
replications 下，相对 MDL-2 降低 0.01636% 成本，confidence interval 完全低于
zero，同时 patient-facing metrics 略有改善。这个 improvement 很小，但统计上稳定。
Matched TD3 的结果只有 -0.00125%，区间跨过 zero，而且 DDPG 在五个 seed 中都优于
TD3。所以我们认为 DDPG 应继续作为主 backbone。

第二，我们把 action space 扩展到 geography-aware reagent 和 idle-bioreactor
transfer 后，Network AFR-GCN-DDPG 在 regional abrupt-shift setting 中相对 MDL-2
降低 1.13%，相对 parameter-matched flat policy 降低 1.12%，两个 confidence
interval 都完全低于 zero，而且 completion service 变好。这个结果说明 graph 的优势
只有在 policy 真正控制 edge-level network action 时才会体现出来。

第三，我们在 4090 上跑了 conservative online TD3 screen。Final GCN policy 在三个
seed 中都优于 MDL-2，pooled CI 也完全低于 zero，并且明显优于 flat。但是 final
checkpoint 相对 frozen pretrained checkpoint 的 pooled CI 仍然跨 zero。所以现在
可以说 final graph controller 更好，但还不能完全说这部分 improvement 是 online
TD3 updates 单独造成的。

我们现在正在做五个场景的 final-versus-frozen zero-shot evaluation，不需要重新训练。
它会直接告诉我们 online TD3 的 contribution 是 strong、scenario-specific，还是尚未
established。根据结果，我们再决定是进入 5-seed full confirmation，还是回到
GCN-DDPG 做 multi-scenario conservative fine-tuning。

我希望今天和你确认两点：第一，我们是否继续把 AFR-GCN-DDPG 作为 proposed method，
TD3 作为 matched stability ablation；第二，我们是否同意把 paper 的 claim 明确拆成
graph representation、anchored residual control 和 online RL attribution 三个层次。

## 9. 汇报时最重要的措辞

建议说：

> The graph-aware anchored residual controller significantly outperforms
> MDL-2 and a parameter-matched flat controller under geographically
> structured regional demand shifts.

当前不要说：

> Online GCN-TD3 reinforcement learning has been proven to outperform all
> heuristics across all scenarios.

更准确的 online RL 表述是：

> Online TD3 fine-tuning is directionally favorable, but its incremental
> contribution beyond the frozen graph-pretrained controller is still being
> evaluated across unseen scenarios.
