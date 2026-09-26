# Research progress update — 2026-09-26

- Reporting period: 2026-08-29 to 2026-09-26
- Audience: Howard / Zhaowei discussion
- Branch: `e2b-run` (last commit 2026-09-16; manuscript last edited 2026-08-22)

Sources: `specs/2026-09-15-*` and `specs/2026-09-16-*` results, `docs/submission_readiness_review_2026-09-15.md`, `docs/reporting_audit_2026-09-15.md`, `reports/anchor_decision_2026-09-15/`.

## 1. 目前整體狀態

**Formal campaign 已關閉（Stage E）**，主結果不變。正式設計為 5 seeds × 4 scenarios × 100 paired replications（候選減比較對象，越低越好）：

| 比較 | Δ objective | 95% CI (M) |
| --- | ---: | ---: |
| AFR-GCN-DDPG vs MDL-2 | −0.66% | [−19.4, −15.7] |
| AFR-Flat-DDPG vs MDL-2 | −0.34% | [−10.6, −7.5] |
| GCN vs matched flat | −0.32% | [−11.1, −6.4] |
| Final GCN vs frozen pretrained | +0.003% | 無配對 CI |

**論文定位已確定為「honest-negative for online RL」**：增益來自 offline 蒸餾的 graph residual，online 更新沒有可量測貢獻。8/29 的 postmortem（`docs/online_rl_attribution_postmortem_and_followup_brief.md`）找出五個疊加的失敗原因：量化梯度消失、counterfactual label 不穩定、horizon 不匹配、auxiliary loss 被淹沒、強 anchor。Label 不穩定是主因。

**投稿前 review（9/15）的結論是「先修 reporting 再投」**，不是再訓練。

## 2. 9 月新增結果

全部在 development seed stream 99.7M 上，world-paired，不動 formal holdout。

1. **Anchor 覆蓋不足**（`specs/2026-09-15-prior-oracle-gap-screen/`）。MDL-2 lookahead 為 2，生產 lead time 為 3。MDL-3 在 fresh worlds 上比 MDL-2 便宜 3.65%（RL 為 0.66%）。但 MDL-3 使 manufacturing-ineligibility 上升 0.2–0.5pp，違反 0.1pp 臨床 guardrail。Anticipation oracle 沒有價值；價值來自 hedging/coverage。
2. **Decision B（2026-09-16，Howard）**：保留 MDL-2 為正式 anchor，臨床 gate 不變，MDL-3 成為每個 cell 都必須報告的 comparator。Manuscript 必須說明為何排除 MDL-3 的 3.65%。
3. **GCN residual zero-shot 放到 MDL-3 上**（`specs/2026-09-16-learned-vs-retuned-anchor-fresh-worlds/`，探索性）：對 MDL-3 −0.19%（3/3 seeds），並修復 MDL-3 的 guardrail 違規；合併對 MDL-2 為 −4.33% 且通過 pooled gate。Flat residual 失敗（+0.71%）。使用的是 local TD3 dev checkpoints，不是 formal DDPG。
4. **Hindsight LP 下界，Measurement B**（`specs/2026-09-16-hindsight-lower-bound/`）：200/200 worlds 上有效，但 gap ≤39%，無法區分 policy。可用的 certified 數字：任何 policy 至少損失 28.3% 的到達病人；MDL-2 為 40.1%。
5. **Information-matched rollout planner，Measurement A**（`specs/2026-09-16-measurement-a-rollout-planner/`），本月最重要結果：

| Planner | Δcost vs MDL-2 | 95% CI | worlds | guardrail C/I/L |
| --- | ---: | ---: | ---: | :--: |
| Restricted class（與 residual 相同的 action class） | −1.83% | [−2.19, −1.48] | 40 | 通過 |
| AFR-GCN-TD3 residual（同 40 worlds） | −0.69% | [−0.87, −0.51] | 40 | 通過 |
| Full class（加 MDL-3 anchor、reagent transfer） | −6.64% | [−7.98, −5.22] | 12 | 通過 |
| MDL-3（同 12 worlds） | −4.79% | [−6.24, −3.26] | 12 | 失敗 I/L |
| Value of information（privileged − matched） | −0.37% | [−0.59, −0.14] | 12 | — |

解讀：
- 同樣觀測、同樣 action 權限下，不學習的 planner 拿到的改善是 learned residual 的 2.7 倍。
- 放寬到現有 full controls 後為 6.6%，且通過 MDL-3 無法通過的 gate（planner 在 52% 的決策用 MDL-3 動作，但在會把 frail 病人推進製造的狀態下拒絕它）。
- MDL-2 的 optimality gap 現在有實作得到的區間 [6.6%, 39%]。
- 資訊價值只有 0.4%，剩下的 slack 是 control authority 與 LP relaxation，不是資訊。
- Planner 每 episode 10–20 分鐘，不可部署；它的 decision logs（`artifacts/full_decision_logs.json`）可作為 teacher。

## 2a. 各環境的增益空間 vs. 各方法實際拿到的比例

所有數字都是相對 MDL-2 的 total objective 變化（負值為改善）。

### 四個 routing-primary scenario 的模擬參數

共同設定（`experiments/configs/20_clinic_patient_condition_geo.json` 加上 benchmark plan 的 routing overrides）：

| 項目 | 設定 |
| --- | --- |
| 網路 | 20 家 clinic，四個地理 cluster 各 5 家（西岸 / 山區 / 中部 / 東部），每家有自己的 bioreactor、reagent、specimen 庫存；含 central capacity hub |
| 每 cluster 的基準需求率（病人/epoch） | 7.5 / 3.2 / 6.0 / 3.5，Poisson 到達 |
| 每 cluster 的初始 idle bioreactors 與上限 | 5→9 / 16→24 / 7→11 / 14→22 |
| 每 cluster 的初始 reagents 與上限 | 60→130 / 170→240 / 80→150 / 150→220 |
| Horizon | 52 epochs（週），production lead time 3，transfer lead time 3（依距離分級 500/1500 mile） |
| Specimen routing | 開啟，36 條合格 specimen edge，routing lead time 1 epoch，每病人最多轉一次，運送損失 0，成品回原 clinic、回程 lead time 0 |
| 供應商 | 每 cluster 的 disruption rate 0.45 / 0.12 / 0.35 / 0.12 |
| 病人生命週期 | healthy decay 0.0013、frail decay 0.0156、risk type 機率 0.45/0.35/0.20（decay 倍率 0.8/1.0/1.35）、Weibull shape 1.5 scale 4.0、post-shock 倍率 3.0、eligibility threshold 0.8 |
| 保存期限 | material 6 epochs，finished product 2 epochs |
| Cost weights | patient lost 500k、expiry 100k、urgency 25k（2026-07-26 後的校準） |
| Anchor 觀測 | 12-epoch demand history 進 state；MDL-2 forecast horizon 2 |
| Action | facility_net，4n 區塊（specimen / capacity / reagent net transfer + replenishment）；formal residual 只開 specimen 通道（scale 0.1），其餘 0 |

各 scenario 的差異：

| Scenario | 需求 regime 變化 | 隨機 demand shock | 區域供應商中斷 | Forecast error | 意圖 |
| --- | --- | --- | --- | --- | --- |
| Nominal history | 無（倍率全 1.0） | p=0.12/epoch，×2.4，持續 4 epochs，cluster size 4 | p=0.08，持續 2 epochs，cluster size 4 | 0 | 基準 |
| Abrupt regime shift | 第 26 epoch 瞬間切換：西岸 1.25→0.75、山區 0.8→1.4、中部 1.15→0.8、東部 0.85→1.3 | 關閉 | 關閉 | 0 | 需求 prior 突然失效，測反應速度 |
| Regional drift | 從第 0 epoch 線性漂移 52 epochs：西岸 1.0→1.4、山區 1.0→0.75、中部 1.0→1.3、東部 1.0→0.8 | 關閉 | 關閉 | 0 | 需求 prior 緩慢失效，測追蹤 |
| Compound regional stress | 第 13 epoch 起 13 epochs 內漂移到：西岸 1.5、山區 0.7、中部 1.35、東部 0.75 | p=0.16，×2.6 | p=0.12，持續 3 epochs | 0.1 | 最嚴苛，三種壓力同時 |

三個非 nominal scenario 的 MDL-2 demand forecast 都改用 `prior_estimate`（固定初始需求率），所以 anchor 的 prior 在 regime 變化後就是錯的。

### 名詞定義（表 A 的欄位）

- **空間 restricted**：Measurement A 的 information-matched rollout planner，只能用 learned residual 完全相同的五個候選動作（MDL-2 anchor，加上 centred specimen pattern 的 ±0.05、±0.10）。每個決策點對每個候選跑 6 個 CRN world 到 episode 結束、MDL-2 接續，選平均成本最低者。它只看 policy 看得到的資訊，所以代表「在同樣的觀測與同樣的 action 權限下，最多能改善多少」。
- **空間 full**：同一個 planner，但候選擴到 14 個：上面五個、同樣五個改建在 MDL-3 anchor 上、以及 MDL-2 上的 ±0.05 reagent transfer 與 ±0.05 combined transfer。代表「用現有 simulator 已允許的 full controls，最多能改善多少」。
- **MDL-3 單獨**：不學習，只把 anchor 的 lookahead 從 2 改成 3。括號是違反的臨床 guardrail。
- **Flat 蒸餾 / GCN 蒸餾**：AFD（advantage-filtered distillation）的 frozen pretrained actor，也就是 online RL 開始前的 checkpoint。訓練方式是用 teacher 的 counterfactual label 做監督式蒸餾（300 epochs 加兩輪 DAgger），沒有 actor-critic 更新。Flat 是參數量匹配的 MLP encoder，GCN 是 graph encoder。
- **GCN online RL 增量**：同一個 GCN actor 經過 100 episodes online DDPG 之後（final checkpoint）減去 frozen pretrained 的成本差。這就是「online RL 本身」貢獻了多少；正值代表變差。
- **蒸餾 ÷ restricted 空間**：GCN 蒸餾的改善除以 restricted 空間，代表在相同 action class 內，learned policy 拿到了可達改善的幾成。

### 表 A：routing-primary 四個 scenario（目前校準後的環境）

| Scenario | 增益空間 restricted (planner) | 增益空間 full (planner) | MDL-3 單獨 (guardrail) | Flat 蒸餾 (frozen) | GCN 蒸餾 (frozen) | GCN online RL 增量 (final − frozen) | GCN 蒸餾 ÷ restricted 空間 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Nominal history | −2.35% | −9.22% | −5.60% (I ✗) | −0.46% | −0.69% | +0.002% | 29% |
| Abrupt regime shift | −1.98% | −7.17% | −5.38% (I ✗) | −0.05% | −0.43% | −0.006% | 22% |
| Regional drift | −1.16% | −4.82% | −3.05% (I ✗, L ✗) | −0.32% | −0.68% | +0.004% | 59% |
| Compound regional stress | −1.79% | −5.79% | −3.09% (通過) | −0.43% | −0.76% | +0.008% | 43% |
| **Pooled** | **−1.83% [−2.19, −1.48]** | **−6.64% [−7.98, −5.22]** | **−4.14%** (I ✗) | **−0.34%** | **−0.66% / −0.69%** | **+0.003%** | **36–38%** |

資料來源與注意事項：
- 增益空間 restricted：40 worlds（每 scenario 10）；full：12 worlds（每 scenario 3，pilot budget）。兩者都在 dev seed stream 99.7M。
- MDL-3 per-scenario 來自 `reports/anchor_decision_2026-09-15/`（100 paired worlds/scenario，同一 seed stream）；pooled 為 routing4。括號為違反的 guardrail（C 完成率、I 製造期不適格、L 損失病人）。
- Flat 與 GCN 蒸餾、online 增量 per-scenario 來自 formal holdout（5 seeds × 100 replications，`formal/pretrain_summary.json` 與 `final_summary.json`），是不同 seed stream。Pooled 的 −0.69% 是 TD3 dev checkpoints 在同一 99.7M worlds 上的數字，可與 planner 直接比較；−0.66% 是 formal DDPG。
- 蒸餾拿到的比例 pooled 為 0.66/1.83 ≈ 36% 或 0.69/1.83 ≈ 38%；若以 full class 空間計算只有約 10%。
- Online RL 增量在四個 scenario 全部小於 0.01%，遠小於 CI 寬度（約 ±0.02–0.03%），不可歸因。
- Certified LP 下界（Measurement B）給的上限是每個 scenario 34–44%、pooled 39%，但資訊價值只有 0.37%，所以上限的 slack 幾乎都是 LP 的 control authority，不是可被 policy 拿到的空間。

### 最優解區間：表 A 沒有直接回答的問題

表 A 只給了可達到的改善（下界），沒有給最優解的上限。目前對「MDL-2 距離最優解多遠」的完整答案是一個區間：

| 界 | 來源 | 數值（pooled） | 各 scenario | 性質 |
| --- | --- | ---: | --- | --- |
| 下界：最優解至少比 MDL-2 好這麼多 | Measurement A full-class planner（12 worlds） | −6.64% | −9.22 / −7.17 / −4.82 / −5.79 | 實際存在的 policy 做到的，可達 |
| 上界：最優解最多比 MDL-2 好這麼多 | Measurement B certified perfect-information LP（200 worlds） | 39.0% | 39.5 / 43.5 / 41.4 / 34.3 | 有效但鬆，不可達 |

所以 optimality gap 在 [6.6%, 39%] 之間，區間寬到目前沒有實用價值。

上界為什麼鬆，已經查清楚：
- **不是資訊。** 把真實的病人 latent 與需求率給 planner，只多拿 0.37% [0.14, 0.59]。39 個點裡「知道未來」的成分不到 1 個點。
- **是 LP 的控制權。** LP 可以挑選要啟動哪些等待中的病人、可以保留 slot 不用；simulator 強制 production = min(waiting, idle, reagents)，這是非凸的，LP 寫不進去，所以 LP 擁有任何 policy 都沒有的 authority。
- **不是約束。** 加 site/transfer cap 只讓 bound 動 0.03%。

要收緊上界需要尊重 forced-start 規則的 relaxation，或 penalized information relaxation（duality-based bound）。兩者都未做，也不是投稿依賴。

Measurement B 給的另一個數字比 39% 有用：**任何 policy 至少損失 28.3% 的到達病人**（各 scenario 22–37%），MDL-2 為 40.1%，最佳 learned 配置 39.6%。控制最多只能影響 12 個百分點，而所有測過的 policy 之間只差 0.5 個百分點。這是 motivation section 該用的句子：這個系統的損失由容量決定，控制只在邊際上有用。

### 表 B：其他試過的環境 / 決策通道

| 環境（spec 日期） | 增益空間的量測方式 | 增益空間 | Offline 蒸餾 GCN | Flat 對照 | Online RL 增量 | 結論 |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Regional regime network transfers（07-25，舊校準、無 routing） | End-to-go teacher cache（in-sample，單 rollout） | −2.07% | −1.13% [−1.22, −0.86]（3 seeds） | ≈ −0.02%（三 seeds −0.06/+0.07/−0.06） | 無 online arm | 唯一過 1% gate 的 graph 效果；zero-shot 轉移到其他 scenario 失敗（+0.38%） |
| Demand-prior drift（07-20，舊 penalty weights 50k） | 未量測 | — | −0.016%（CI 低於零，5 seeds） | Flat 也贏；graph−flat 跨零 | 無 | 顯示 AFD 打贏錯 prior 的 anchor，不是 graph 的效果 |
| Geo regional drift, conservative AFR-TD3（07-30，校準後） | 未量測 | — | −0.082% [−0.12, −0.03] | +0.073%（輸） | 無（`num_episodes=0`） | 第一個 CI 低於零的 learned-beats-heuristic；純蒸餾 |
| Routing-primary formal（08-06 → 08-17） | 見表 A | −1.83% / −6.64% | −0.66% | −0.34% | +0.003% | 論文主結果 |
| Continuous overtime channel（08-29） | State-dependence value screen（不訓練） | Reserved scenario +0.45%（門檻 0.5%）；out-of-sample lookup 比常數差 −0.083% | 未訓練 | — | — | Gate 失敗；headroom 是 corner solution，policy 學不到 |
| Stochastic procurement lead time（08-29） | Heuristic 在隨機 vs 固定 lead time 的成本差 | +0.176%（門檻 1%）；lead-aware planner 只回收 0.9% | 未訓練 | — | — | Gate 失敗；review 指出這是 variability cost，不是 optimality gap |
| Routing label stability, G1 / label budget（08-06, 08-29） | Counterfactual label 跨 CRN 的一致率 | 54.5% 一致（門檻 70%） | — | — | — | Online RL 的 binding failure：label 不穩定 |

表 B 的 headroom 量測方式彼此不同，不能橫向比較；只有表 A 的 Measurement A 是資訊匹配且 forced-start 一致的 planner。

## 3. 卡住的地方，需要 Zhaowei 提供

- **Formal row-level 資料只在 Zhaowei 的 Mac 上**，gitignored 路徑 `results/patient_indexed_specimen_routing_mac_mps_primary/`。Crossed-design bootstrap 工具（`evaluation/crossed_design_bootstrap_audit.py`）已建好並驗證，沒有 raw rows 就無法重算 formal CI。檔案清單在 `docs/reporting_audit_2026-09-15.md` §2；請匯出為唯讀 archive 並附 SHA-256 manifest。
- **Formal DDPG checkpoints**（10 個 `*_seed1{0..4}_episode100.pt`）也在他那邊。GCN-on-MDL-3 要升級為 preregistered confirmation 需要它們。
- **0.1pp manufacturing-ineligibility margin 沒有書面臨床理由**。Decision B 要求 manuscript 說明，repo 內沒有，需要臨床端提供。

## 4. 接下來要做的事（建議優先序）

1. **拿到 archive，跑 crossed bootstrap，補 final-vs-frozen 的 paired interval。** 目前可用資料下沒有結論會改變，但 two-way interval 預期寬 10–40%。
2. **Manuscript 收斂**（8/22 後未動）：
   - Demand-drift 表加註 cost weight 為 50k/40k/5k 時代（2026-07-26 後改為 500k/100k/25k）。
   - 刪除未執行的 RQ4 temporal encoder 與 graph edge ablations。
   - 把 frozen pretrained 拉進主比較。
   - DDPG 方程式改為實際執行的 anchor-relative target。
   - 歷史研究移到 appendix；README 改為 primary-study entry point。
   - Review 建議的標題方向：強調 residual control 的 benefits and limits of online learning。
3. **決定 Measurement A 怎麼進論文。** 建議放在 discussion 作為「achievable, not deployable」的 headroom，明確寫出 learned residual 只拿到 restricted class 的約三分之一，不主張學習可達 6.6%。
4. **下一個學習候選（follow-up paper，非投稿依賴）**：從 full planner 的 decision logs 蒸餾一個對 coverage 與 routing 都有權限的 residual。這是 Environment ≻ Algorithm ≻ Architecture 路線的延續，訓練資料已存在。
5. **Package and freeze**：configs、teacher、checkpoints、rows、manifests、reporting commands 的 reproducibility archive。

## 5. 討論點

- Zhaowei 是否同意 Decision B（MDL-2 留作 anchor、MDL-3 為 mandatory comparator）。
- 誰負責寫 0.1pp margin 的臨床理由。
- Archive 匯出時程。這是投稿的第一個 blocker。
- Measurement A full class 只有 12 worlds，是否在他的機器上補到每 scenario 10 worlds 後再寫進論文。
- Follow-up 的 planner 蒸餾要現在開 spec，還是等 freeze 之後。
