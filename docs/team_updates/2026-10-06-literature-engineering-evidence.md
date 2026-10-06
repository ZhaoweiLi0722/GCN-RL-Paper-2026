# Literature Evidence, Synthetic Sensitivity Design, and TD/TD3 Clarification

Date: 2026-10-06. Status: literature-supported research note and prospective
sensitivity design, not a new experiment or completed calibration.

## Decision and scope

Zhaowei reported that site-level engineering data are unavailable and requested
published engineering/clinical evidence and defensible sensitivity ranges. The
study can proceed as a **methods study in synthetic scenarios, with literature
benchmarks and explicit unvalidated assumptions**. Literature does not turn the
existing simulator into a validated digital twin. No site records, patient-level
dataset, expert elicitation, new trajectories, model forwards, optimizer updates,
or parameter changes were obtained or performed in this work.

Completed experiments retain their original parameters and negative findings.
These newly collected references are not retrospectively claimed as their
calibration inputs. The operational-calibration gap (E1) remains open, with a
better documented literature basis. Clinical efficacy and deployment safety are
outside the claim. Reward remains unchanged.

## Source register

Sources below are original studies, not figures repeated from a review. Full text
and the stated locators were inspected by Codex on 2026-10-06. Bibliographic and
claim checks by the accountable human authors remain pending; no expert or human
verification is implied. Only public bibliographic queries were used externally.

### E01: Process costs and qualified staffing

Ran T, Eichmueller SB, Schmidt P, Schlander M. *Cost of decentralized CAR T-cell
production in an academic nonprofit setting*. International Journal of Cancer.
2020;147:3438-3445. [DOI and publisher full text](https://onlinelibrary.wiley.com/doi/10.1002/ijc.33156).

- Locators: Methods 2.1-2.2; Tables 1-3; limitations 4.1.
- German academic CAR-T process: 12-14-day production; two experienced technicians.
  Annual fixed cost EUR 438,098 and variable cost EUR 34,798/production, in 2018
  euros. These are process/accounting estimates, not current commercial prices.
- Use: separate fixed staffing, consumables, and capacity utilization. Do not
  convert annual salary directly to marginal overtime cost or assume that adding
  staff accelerates cell expansion. Geographic, price-year, and cost-boundary
  adjustments are required before any numerical import.

### E02: Published sensitivity ranges, not observed population frequencies

Lopes AG, Noel R, Sinclair A. *Cost analysis of vein-to-vein CAR T-cell therapy:
automated manufacturing and supply chain*. Cell & Gene Therapy Insights.
2020;6(3):487-510. DOI: [10.18609/cgti.2020.058](https://doi.org/10.18609/cgti.2020.058).
The DOI resolver failed during this read; the
[author-uploaded full text](https://www.researchgate.net/publication/342076749_Cost_analysis_of_vein-to-vein_CAR_T-cell_therapy_automated_manufacturing_and_supply_chain)
was accessible.

- Locator: Table 4, p. 499. Low/base/high batch-failure assumptions are 5/10/15%
  for manual, 3/5/10% for partially automated, and 1/3/5% for fully automated
  processing.
- Use: a transparent precedent for process-specific sensitivity bands. These
  are that paper's modeled inputs, not confidence bounds or expert consensus.
  Do not pool the three processes into one distribution or equate batch failure
  with death, waiting-list dropout, or the current simulator's patient-loss event.

### E03: Labor structure, with an explicit cross-product limit

Fitzgerald JC, Duffy N, Cattaruzzi G, et al. *GMP-Compliant Production of
Autologous Adipose-Derived Stromal Cells in the NANT 001 Closed Automated
Bioreactor*. Frontiers in Bioengineering and Biotechnology. 2022;10:834267.
[Publisher full text](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.834267/full).

- Locators: labor-analysis Methods/Results; Figures 5-6.
- The labor analysis reports 43% fewer total labor hours per batch with automation.
  The economic model includes two manufacturing technicians, one QC technician,
  and one Qualified Person.
- This is an ASC process, not CAR-T. Use the task/qualification separation, not a
  transferable productivity multiplier. The text assumes 8 +/- 1 production days
  while figure captions say 7 +/- 1; do not silently resolve that inconsistency
  or adopt a duration from it. No timestamped staffing-response dataset was read.

### E04: Real-patient aggregate evidence, not a patient-level training dataset

Nastoupil LJ, Jain MD, Feng L, et al. *Standard-of-Care Axicabtagene Ciloleucel
for Relapsed or Refractory Large B-Cell Lymphoma: Results From the US Lymphoma
CAR T Consortium*. Journal of Clinical Oncology. 2020;38(27):3119-3128.
DOI: 10.1200/JCO.19.02104.
[Full text](https://pmc.ncbi.nlm.nih.gov/articles/PMC7499611/).

- Locator: Results, Patient Characteristics and Disposition.
- At 17 US centers, 298 patients underwent leukapheresis and 275 received infusion.
  Leukapheresis-to-conditioning time: median 21 days, IQR 20-24, range 11-71.
- The timing endpoint is **conditioning, not infusion or manufacturing completion**.
  These aggregates can contextualize pathway attrition/timing; they do not identify
  a daily deterioration hazard or the causal effect of a scheduling policy.
  No individual-level data or data-use authorization was obtained.

## How to use ranges without inventing calibration

Use four provenance labels: observed study summary, published modeled range,
author-selected synthetic stress, and unresolved site input. An expert-informed
label requires a named, documented published elicitation or an actual authorized
consultation. None has occurred here. Never relabel an assumed +/-20% band as
expert consensus. Currency conversion and inflation would be separate declared
transformations; no current-price conversion is made in this note.

The following is a proposed analysis design, **not an executable configuration**:

| Dimension | Prospective values or treatment | Provenance and restriction |
|---|---|---|
| CAR-T production duration | 12, 13, 14 days as endpoint/midpoint scenarios | E01 supplies the endpoints; 13 is our arithmetic midpoint, not an empirical median. A mapping to simulator time units and process stages is needed. Do not substitute for whole vein-to-vein time. |
| Manufacturing failure | E02 process-specific low/base/high triplet | Separate structural extension if batch failure is not represented. Hold process type fixed within a comparison. The controller must not change biological failure probability without evidence. |
| Staff-booking notice | 0, 1, 2, 4 decision epochs | Author-selected synthetic stress, not industry measurements. Preserve paid commitments, qualification requirements, and equipment constraints. |
| Record-availability delay | 0, 1, 2 decision epochs | Author-selected synthetic stress; all controllers receive the same information interface. No future-corrected observations. |
| Marginal staffing benefit | Include a zero-benefit/equipment-blocked condition alongside declared nonzero synthetic responses | Structural stress test, not a fitted response law. Report idle paid hours and infeasible requests separately. |
| Patient pathway | Keep E04 as an external aggregate benchmark, separate from simulation outcomes | Do not fit a daily hazard from a single time summary and infusion fraction. Different indications, treatment eras, and selection processes require separate strata. |
| Economic interpretation | Report existing component costs, labor, waiting, completions, and losses separately | Preserve the primary reward. A post hoc accounting sensitivity must be labeled as such; revised training objectives need a prospective amendment. |

No independent-uniform probability distribution is implied by these grids.
Correlations, censoring, and incompatible process combinations must be explicit.
The recommendation is to embed a small predeclared sensitivity panel into the next
complete method comparison, not initiate a separate sequence of diagnostic runs.

Freeze the scenarios before new evaluation; use paired external tapes without
claiming event-level pairing when endogenous events diverge. Define whether the
estimand is frozen-policy robustness or retraining under a new process. For the
former, train once on the declared base setting and freeze before evaluating the
panel. Any retraining is a separate budgeted treatment. Report all conditions,
cost/service trade-offs, adverse worlds/blocks, uncertainty and total computation.
Do not select ranges, checkpoints, or rewards based on which makes RL win.

The budget, seed allocation, dimensional mapping and action/observation interface
for this panel are not frozen yet. This literature work does not reopen consumed
attempts or launch new scientific execution.

## TD is not TD3; current algorithm assessment

Temporal-difference (TD) learning is a learning principle. Twin Delayed DDPG (TD3)
is an actor-critic algorithm; its original paper uses paired critics and delayed
policy updates: Fujimoto, van Hoof and Meger, ICML 2018,
[Addressing Function Approximation Error in Actor-Critic Methods](https://proceedings.mlr.press/v80/fujimoto18a.html).
The latest native-return experiment is GCN TD value learning inside MPC, with
zero actor updates; it is not TD3.

The saved local evidence does not support a general TD3-over-DDPG claim:

- In the older pure-policy screen, TD3 was stronger than pure DDPG, but both
  remained worse than the strong heuristic. See
  [RL-family readout](../../specs/2026-07-23-rl-family-comparison/results.md).
- In that campaign's matched residual 300-episode/five-seed comparison, DDPG
  cost was 0.015112% lower than TD3; cost difference -191,662.15, interval
  [-234,659.45, -148,400.79], all five seed means favorable to DDPG. See
  [saved paired summary](../../specs/2026-07-23-rl-family-comparison/ddpg_afd_vs_td3_afd_targeted300_five_seed.json).
- The later exploratory routing/MDL-3 TD3 result includes a changed anchor;
  its MDL-3 rows have no matched DDPG counterpart. A favorable package result
  cannot identify superiority of TD3. See
  [fresh-world readout](../../specs/2026-09-16-learned-vs-retuned-anchor-fresh-worlds/results.md).
- The newest native TD value-MPC has favorable secondary contrasts but failed
  its primary strong-baseline screen. This is neither a TD3 result nor a matched
  comparison with the historical routing DDPG task. See
  [native-return readout](../../specs/2026-10-05-native-return-value/terminal-readout.md).

### Direction decision after the user's follow-up

Zhaowei asked whether continuing TD has a defensible purpose without a clear
advantage. Current evidence does **not** justify promoting this terminal-value
TD controller as a superior main method or automatically launching another
target-refinement round. The recommendation is to stop incremental continuation
of this consumed TD line, retain MPC as the engineering reference, and retain
TD as an experimental enhancement with mixed results. This is a research
recommendation, not proof that every TD method is ineffective.

Persistent-change savings against H8 are 1.813%, with the cost-difference interval
including zero. Against forecast TD, 0.462% mean savings coexist with 2.40 more
modeled patient losses per world; neither establishes population-level benefit
or harm. The favorable fast-fluctuation/H8 and same-native-data TD/MC secondary
results warrant reporting but do not rescue the primary criterion. No H16
performance equivalence has been demonstrated, so the cheaper H8-length decision
cannot yet be marketed as equivalent long-horizon planning at lower cost.

Candidate-relative TD, suggested in discussion, remains an untested hypothesis,
not evidence supporting another automatic campaign. A future learning study
needs a decision-relevant estimand: better cost with protected service at a
matched resource budget, or noninferior cost/service with materially less planning
computation including training amortization. The practical margins and budgets
must be justified prospectively; no arbitrary new pass threshold is added here.
Literature can strengthen the problem assumptions but cannot establish an RL
increment. Do not switch back to DDPG, or to TD3, on algorithm name alone.
Keep historical DDPG, TD3 and current value-MPC evidence distinct in the paper.

## Manuscript and handoff

The Discussion now labels the work as synthetic methods research and distinguishes
these prospective external benchmarks from completed calibration or sensitivity
experiments. Four references were added. The source register remains subject to
human review. No scientific source, config, frozen contract, or result was edited.

Writing used the installed scientific-writing and manuscript-writing-review
instructions. Software reference: Kassis T, Agarwal V, He Y, Patel D, Brueckner AM
(2026), *Scientific Agent Skills: A Library of Procedural Knowledge for Research
Agents*, arXiv:2609.00065, current v2 inspected on 2026-10-06,
[DOI](https://doi.org/10.48550/arXiv.2609.00065). This is writing provenance, not
scientific evidence for the controller or the manufacturing assumptions.

Validation: git whitespace checks, workflow JSON parsing, citation-key resolution
(62 bibliography entries, 45 cited keys), and LaTeX environment nesting passed.
No code changed; no training tests or compileall were repeated. The multi-file
manuscript PDF was not recompiled or visually verified. No new cloud sync or
remote publication is implied by this note.
