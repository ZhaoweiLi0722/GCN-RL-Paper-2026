# Current EAAI Manuscript

Target: **Engineering Applications of Artificial Intelligence (EAAI)**, not
AAAI. The user confirmed the journal on 2026-10-11. This is a local
methods-stage discussion draft, not a submission-ready paper.

## Current Entry Point

Open [main.tex](main.tex) in the Codex LaTeX editor for the current source and
compiled PDF preview. It now describes **Native Value-TD on the original
twenty-site task**. It contains its bibliography directly so the editor can
compile the document without external project inputs. `references.bib` remains
unchanged for the preserved historical manuscripts.

- Section 2: physical task, all three transfers, eighty request coordinates,
  original cost and finite 52-epoch endpoint.
- Section 3: learned cost value, public-information planning, graph/flat
  encoders, observed TD targets, A's fourteen templates, and B's sequence CEM.
- Section 4: why the research direction changed from actor-critic DDPG to
  value learning with planning, including the trade-offs.
- Section 5: prospective independent evaluation and the distinct
  MDL-2/H8/Flat questions. Results remain pending.
- Appendices: exact A template construction and preserved historical findings.

The two Value-TD planners are distinct frozen implementations, not a
post-hoc winner selected from training outcomes. The current draft does not
claim that TD beat DDPG or that either A or B passed its independent screen.

## Historical Material

- [DDPG method archive](archive/ddpg-method-record.md): mechanism, positive
  operational evidence, unresolved incremental learning, adverse sensitivity,
  limitations and a Chinese meeting explanation.
- [Full historical record](historical-research-record.tex): unchanged original
  method and results, including their then-current research decisions.
- [Pre-revision files](archive/2026-10-11-pre-native-value-td/): byte-preserved
  previous main source (four-site support-hour Value-TD), title page,
  highlights and availability statement. The preserved files were already
  inconsistent with one another; they are historical, not current metadata.
- `archive/2026-10-11-pre-native-value-td/legacy-main-unmatched.pdf`: the old
  PDF, whose date predates the previous source. It is not a verified rendering
  of either the archived four-site source or the current manuscript. It was
  moved out of `main.pdf` to prevent accidental use as the current paper.
- `main_diff.tex`, `main_diff_clean.tex` and prior figures are historical
  artifacts, not regenerated views of this revision.

Do not run the old archived TeX source in place and assume its relative figure
paths resolve. Its original build context remains available through Git; its
byte-preservation does not certify a newly compiled historical PDF.

## Howard Meeting Summary

现在的主方法是 **TD cost-value learning + native-model planning**。网络学习
“当前状态往后还要付出多少成本”，规划器用原有仿真器比较近期决策，再加上这个
长期成本估计，只执行第一步、随后重新规划。A 使用 14 个随状态变化的候选请求；
B 用 CEM 搜索八步、每步 80 维的残差请求序列。两者都保留原二十站点物理规则。

选择这条路线，是把可行性处理交给原仿真器，让学习集中在短规划窗口看不到的
后续代价上，并用相同规划器的零尾值 H8 来检验学习增量。它避免训练直接动作
actor，但付出了更多在线规划计算。DDPG 的历史正面结果仍保留；未得到独立证明
的是后续 actor-critic 更新带来的额外收益，并不是 DDPG 完全无效。

**结果口径：当前独立评估尚未完成，不能宣布新方法已经优于 DDPG、H8 或 Flat。**

English summary: We learn a downstream cost value from the controller's own
trajectories and use it to augment native-model receding-horizon planning.
The simulator handles physical execution; the value model supplies the
cost beyond the short planning window. A uses fourteen state-dependent
request templates, whereas B searches residual request sequences with CEM.
DDPG is retained as historical evidence. The new method's operational,
learning and architecture benefits still require independent evaluation.

The source/evidence mapping and author-review limits are in
[method-editorial-readout.md](method-editorial-readout.md) and
[method-evidence.json](method-evidence.json). No manuscript or experiment
artifact has been uploaded or sent to Howard by this edit.
