# Related-Work Literature Evidence

Date: 2026-10-31. Status: local author-review draft, not submission-ready.
Assistant checker: Codex. Accountable human verifier: pending for every entry.

## Scope and Delivery

Only these three files belong to this assignment:

- `paper/Graph_Aware_Deep_Reinforcement_Learning_for_Adaptive_Capacity_Planning_in_Distributed_Personalized_Regenerative_Medicine_Manufacturing_Networks/sections/related-work.tex`
- The same project's `references-current.bib`.
- This evidence record.

The section contains approximately 1,160 words excluding headings and citation commands, 13 evidence-linked paragraphs, and 26 distinct citation keys. The bibliography contains 27 unique entries. `Kassis2026` is retained for the existing procedural-aid disclosure in `main.tex`, not inserted as manufacturing or algorithm-performance evidence. All eight required existing keys are preserved: `Wang2023`, `Tseng2025`, `Sutton1988`, `Kipf2017`, `Hansen2022`, `DeBoer2005`, `Lillicrap2016`, and `Kassis2026`.

No experiments, simulator calls, models, fits, scientific data analysis, or `compileall` were run. No `main.tex`, historical `references.bib`, results, or figures were edited by this assignment. No manuscript was uploaded. External requests used public bibliographic metadata only. No commit, push, PR, external message, or disclosure approval was performed. Compilation and integration belong to the parent.

## Verification Convention

This bounded assignment consolidates the scientific-writing skill's source and claim records into this requested Markdown file rather than creating additional manifests. E IDs below map to exact BibTeX entries, which contain the titles, complete author lists, publication years, and identifiers. C IDs are also present as nonprinting comments in the section. The bibliography is a curated replacement candidate, not a copy of the historical library.

- **Opened**: the assistant opened an original publisher, proceedings, or arXiv record and inspected the stated support. An abstract-only check does not imply a full-text review.
- **Original-text mirror**: the assistant opened the authors' original abstract or published paper through NLM or an author-institution repository because direct publisher access failed. This is not a secondary review, but the publisher-access gap remains recorded.
- **Indexed only**: direct primary access failed; publisher-indexed metadata/abstract text was retrieved through browsing. This is a discovery-level check, not completed exact-source verification under the scientific-writing skill.
- **Human pending**: no accountable human is recorded as having opened and approved the exact source-to-claim mapping. This applies to all 27 sources and all 13 paragraphs, including entries marked Opened.

Twenty-five records have opened source content at the limited scopes stated below; E001 and E002 remain indexed-only. No entry is labeled human-verified. No source is used to certify clinical validity, deployment safety, a numerical gain in this study, or manuscript readiness.

## Source Register

### E001: Wang2023

- Metadata: Wang, Tseng, Li, White, Wang, Levine, and Fesnak; *A simulation-based comparison of centralized and point-of-care supply chain strategies for autologous cell therapy*. Cytotherapy 25(12), 1370-1379 (2023). DOI: `10.1016/j.jcyt.2023.08.007`.
- Primary URL: [publisher article](https://www.sciencedirect.com/science/article/abs/pii/S1465324923010381). Supplementary original-paper locator: [NSF-deposited publisher PDF](https://par.nsf.gov/servlets/purl/10522560).
- Assistant status: **Indexed only**. Publisher abstract/section text and the publisher PDF's first-page metadata were retrieved in indexed results. Direct publisher access returned 403; the NSF PDF open returned 502. These failures are not full-text verification.
- Intended support: abstract Methods identifies cost, fulfillment time, service level, and utilization; Results makes the comparison conditional on demand, capacity, and operating rules. Used in C001 and the interpretation in C011. Human pending.
- Correction: the published author list uses **Chip White**. No expansion to a different given name was inferred. The paper's simulation findings do not validate this project's simulator.

### E002: Tseng2025

- Metadata: Chin-Yuan Tseng, Junxuan Li, Li-Hsiang Lin, Kan Wang, Chelsea C. White III, Ben Wang; *Deep reinforcement learning approach for dynamic capacity planning in decentralised regenerative medicine supply chains*. International Journal of Production Research 63(2), 555-570 (2025). DOI: `10.1080/00207543.2023.2262043`.
- Primary URLs: [publisher article](https://www.tandfonline.com/doi/full/10.1080/00207543.2023.2262043), [issue contents](https://www.tandfonline.com/toc/tprs20/63/2).
- Assistant status: **Indexed only**. Publisher metadata, full abstract text, and issue listing were retrieved in indexed results; direct article/PDF opens failed. No full-text method or result verification claimed.
- Intended support: a simulation/RL antecedent for dynamic PRM capacity planning under demand and supply uncertainty, C004/C012. No unqualified generalization or guarantee is imported. Human pending.
- Date/title handling: 2025 is the volume year; online publication was 26 September 2023. The exact title uses **decentralised**, correcting the historical library's Americanized spelling.

### E003: Sutton1988

- Metadata: Richard S. Sutton; *Learning to predict by the methods of temporal differences*. Machine Learning 3, 9-44 (1988). DOI: `10.1007/BF00115009`.
- Source: [Springer record](https://link.springer.com/article/10.1007/BF00115009).
- Assistant status: **Opened**, metadata and abstract. Locator: Abstract, distinction between outcome errors and successive-prediction errors.
- Support: the TD distinction in C006. This is not the definition of every simulation-trained cost predictor. Human pending.

### E004: Kipf2017

- Metadata: Thomas N. Kipf and Max Welling; *Semi-Supervised Classification with Graph Convolutional Networks*. ICLR (2017), arXiv `1609.02907`.
- Source: [author arXiv record](https://arxiv.org/abs/1609.02907).
- Assistant status: **Opened**, metadata, abstract, and ICLR-publication comment. OpenReview direct access was challenged; no interaction with the challenge occurred.
- Support: graph structure and node features combined through graph convolution, C007. The reported experiments are classification experiments, not manufacturing-control evidence. Human pending.
- DOI handling: no conference DOI is asserted. The arXiv identifier and URL are retained rather than representing an arXiv DOI as a conference DOI.

### E005: Hansen2022

- Metadata: Nicklas A Hansen, Hao Su, Xiaolong Wang; *Temporal Difference Learning for Model Predictive Control*. PMLR 162, 8387-8406 (2022).
- Source: [PMLR original proceedings record and BibTeX](https://proceedings.mlr.press/v162/hansen22a.html).
- Assistant status: **Opened**, metadata and abstract. Locator: abstract's short-horizon latent-model optimization and learned terminal value, jointly trained by TD.
- Support: C006's comparison to TD-MPC. No identity with the current full-tail candidate-cost correction is asserted. No DOI was supplied by the opened proceedings record. Human pending.

### E006: DeBoer2005

- Metadata: Pieter-Tjerk de Boer, Dirk P. Kroese, Shie Mannor, Reuven Y. Rubinstein; *A Tutorial on the Cross-Entropy Method*. Annals of Operations Research 134, 19-67 (2005). DOI: `10.1007/s10479-005-5724-z`.
- Source: [Springer record](https://link.springer.com/article/10.1007/s10479-005-5724-z).
- Assistant status: **Opened**, metadata and abstract. Locator: abstract's optimization and rare-event-simulation scope.
- Support: C006 identifies CEM as an optimization method, not proof of a learned increment. No solver-based gain or optimality guarantee is transferred. Human pending.

### E007: Lillicrap2016

- Metadata: Timothy P. Lillicrap, Jonathan J. Hunt, Alexander Pritzel, Nicolas Heess, Tom Erez, Yuval Tassa, David Silver, Daan Wierstra; *Continuous control with deep reinforcement learning*. ICLR (2016), arXiv `1509.02971`.
- Source: [author arXiv record](https://arxiv.org/abs/1509.02971).
- Assistant status: **Opened**, metadata and abstract. Locator: abstract's model-free deterministic-policy-gradient actor-critic formulation.
- Support: DDPG distinction in C006. First arXiv submission was 2015; the retained key/year denotes the 2016 conference version. No conference DOI asserted. Human pending.

### E008: Kassis2026

- Metadata: Timothy Kassis, Vinayak Agarwal, Yuhuan He, Darshil Patel, Aubrey M. Brueckner; *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents* (2026). DOI: `10.48550/arXiv.2609.00065`.
- Source: [current arXiv record](https://arxiv.org/abs/2609.00065).
- Assistant status: **Opened**. The record lists v2, revised 2 September 2026; no journal reference was displayed. The reference uses the version-independent URL/DOI as requested by the installed skill.
- Scope: retained solely for the existing procedural-aid citation. The abstract explicitly reports no task-level evaluation; this source is not support for this manuscript's correctness, RL effectiveness, or validation. Human pending.

### E009: Li2023

- Metadata: Junxuan Li and Chelsea C. White; *Capacity planning in a decentralized autologous cell therapy manufacturing network for low-cost resilience*. Flexible Services and Manufacturing Journal 35, 295-319 (2023). DOI: `10.1007/s10696-022-09475-6`.
- Source: [Springer record](https://link.springer.com/article/10.1007/s10696-022-09475-6).
- Assistant status: **Opened**, metadata and abstract. Online date: 5 November 2022; volume year: 2023.
- Support: abstract explicitly describes relocatable bioreactors, supplier disruption, an MDP, and resilience-constrained capacity allocation, C001. Its partial use of facility data does not calibrate the present simulator. Human pending.

### E010: Tseng2024PatientHealth

- Metadata: Chin-Yuan Tseng, Kan Wang, Li-Hsiang Lin, Chuck Zhang, Chelsea C. White III, Ben Wang; *Advancing autologous CAR T-cell therapy through real-time patient health data integration: a simulation-based approach*. Cytotherapy 26(10), 1152-1162 (2024). DOI: `10.1016/j.jcyt.2024.05.001`.
- Primary URL: [publisher article](https://www.sciencedirect.com/science/article/abs/pii/S1465324924007072). Opened original-text mirror: [NLM author abstract](https://pubmed.ncbi.nlm.nih.gov/38795115/).
- Assistant status: **Original-text mirror**, metadata and original abstract opened; publisher direct access returned 403. Locator: Abstract, SimPAC description and modeled health-dependent manufacturing/scheduling.
- Support: C001 and contextual interpretation in C011. Case-study benefits are simulated, not evidence that a scheduling intervention improves patient survival in practice. Human pending.

### E011: Ran2020

- Metadata: Tao Ran, Stefan B. Eichmueller, Patrick Schmidt, Michael Schlander; *Cost of decentralized CAR T-cell production in an academic nonprofit setting*. International Journal of Cancer 147(12), 3438-3445 (2020). DOI: `10.1002/ijc.33156`.
- Source: [Wiley full text](https://onlinelibrary.wiley.com/doi/10.1002/ijc.33156).
- Assistant status: **Opened**, metadata and relevant full text. Locators: Methods 2.1-2.2; Results Tables 1-2.
- Support: C002's qualified-personnel, equipment, consumable, and fixed/variable cost distinctions. Costs are process/accounting estimates in 2018 currency, not current commercial acquisition prices or a marginal staffing-response law. No numeric input was imported. Human pending.

### E012: Lopes2020

- Metadata: Adriana Lopes, Rob Noel, Andrew Sinclair; *Cost analysis of vein-to-vein CAR T-cell therapy: automated manufacturing and supply chain*. Cell & Gene Therapy Insights 6(3), 487-510 (2020). DOI: `10.18609/cgti.2020.058`.
- Source: [publisher record and abstract](https://www.insights.bio/immuno-oncology-insights/journal/article/1628/cost-analysis-of-vein-to-vein-car-t-cell-therapy-automated-manufacturing-and-supply-chain).
- Assistant status: **Opened**, metadata and abstract; full text requires email/login. No email, login, or access request was submitted.
- Support: abstract describes a Biosolve cost model, automation/scale-out, and sensitivity analysis, C002. No numerical batch-failure triplet is reproduced: the historical note's Table 4 was not independently reopened here. Modeled scenarios must not be presented as observed clinical frequencies. Human pending.
- Name handling: the opened publisher record says Adriana Lopes; no unshown middle name was expanded.

### E013: Fitzgerald2022

- Metadata: Joan C. Fitzgerald and 20 coauthors, all transcribed in the BibTeX entry; *GMP-Compliant Production of Autologous Adipose-Derived Stromal Cells in the NANT 001 Closed Automated Bioreactor*. Frontiers in Bioengineering and Biotechnology 10, 834267 (2022). DOI: `10.3389/fbioe.2022.834267`.
- Source: [Frontiers full text](https://www.frontiersin.org/journals/bioengineering-and-biotechnology/articles/10.3389/fbioe.2022.834267/full).
- Assistant status: **Opened**, full author list, abstract, relevant Methods/Results, labor-analysis text and Figure 6 caption; publisher Citation block cross-checked.
- Support: C002's manual/automated ASC expansion and role-specific labor accounting. Cross-product limitation is explicit: healthy-donor ASC expansion is not CAR-T production or a clinical treatment-effect study. No labor multiplier or production-duration estimate was imported. Human pending.

### E014: Nastoupil2020

- Metadata: Loretta J. Nastoupil and 27 coauthors, all transcribed in the BibTeX entry; *Standard-of-Care Axicabtagene Ciloleucel for Relapsed or Refractory Large B-Cell Lymphoma: Results From the US Lymphoma CAR T Consortium*. Journal of Clinical Oncology 38(27), 3119-3128 (2020). DOI: `10.1200/JCO.19.02104`.
- Primary URL: [ASCO article](https://ascopubs.org/doi/full/10.1200/JCO.19.02104). Opened original-text mirror: [NLM author abstract](https://pubmed.ncbi.nlm.nih.gov/32401634/).
- Assistant status: **Original-text mirror**, full author list, metadata, and original abstract opened. ASCO indexed citation also inspected; direct ASCO open failed and PMC presented a browser challenge.
- Support: abstract Methods/Results follows leukapheresed patients and distinguishes those infused, C003. No daily hazard or causal scheduling effect is identified. The historical note's leukapheresis-to-conditioning endpoint is not relabeled infusion time; no timing statistic is repeated in this section. Human pending.

### E015: Gijsbrechts2022InventoryDRL

- Metadata: Joren Gijsbrechts, Robert N. Boute, Jan A. Van Mieghem, Dennis J. Zhang; *Can Deep Reinforcement Learning Improve Inventory Management? Performance on Lost Sales, Dual-Sourcing, and Multi-Echelon Problems*. Manufacturing & Service Operations Management 24(3), 1349-1368 (2022). DOI: `10.1287/msom.2021.1064`.
- Source: [INFORMS original record](https://pubsonline.informs.org/doi/10.1287/msom.2021.1064).
- Assistant status: **Opened**, metadata and abstract. Locators: abstract Methodology/Results and Managerial implications.
- Support: C004's carefully tuned A3C comparison, stationary inventory scope, and initial tuning cost. The paper says performance can match strong methods; no universal superiority claim is made. Human pending.

### E016: Dehaybe2024NonstationaryInventory

- Metadata: Henri Dehaybe, Daniele Catanzaro, Philippe Chevalier; *Deep Reinforcement Learning for inventory optimization with non-stationary uncertain demand*. European Journal of Operational Research 314(2), 433-445 (2024). DOI: `10.1016/j.ejor.2023.10.007`.
- Primary URL: [publisher article](https://www.sciencedirect.com/science/article/pii/S0377221723007646). Opened author-institution record: [UCLouvain original abstract and paper link](https://research.dial.uclouvain.be/entities/publication/184d6f34-1b43-4e70-b963-f6edc1e37e1d).
- Assistant status: **Original-text mirror**, original abstract and metadata opened. Publisher-indexed abstract/introduction also inspected; direct publisher open returned 403. No full-text read claimed.
- Support: forecast-conditioned rolling-horizon decisions and randomized training environments, C004/C012. This is not evidence of robustness under every form of simulator misspecification. Human pending.

### E017: Stranieri2024HybridInventory

- Metadata: Francesco Stranieri, Edoardo Fadda, Fabio Stella; *Combining deep reinforcement learning and multi-stage stochastic programming to address the supply chain inventory management problem*. International Journal of Production Economics 268, 109099 (2024). DOI: `10.1016/j.ijpe.2023.109099`.
- Primary URL: [publisher article](https://www.sciencedirect.com/science/article/pii/S0925527323003316). Opened version of record: [author-institution deposited publisher PDF](https://iris.polito.it/retrieve/handle/11583/2984070/687092).
- Assistant status: **Original-text mirror**, publisher PDF title/author/DOI page and abstract opened. Locator: article first page, Abstract; the repository cover is an extra first PDF page. Issue year 2024; online date 15 November 2023.
- Support: two-echelon decomposition into RL production decisions and stochastic-programming shipping, C005. This does not imply this project's shared planner has implemented that solver. Human pending.

### E018: Temizoz2025DeepControlledLearning

- Metadata: Tarkan Temizoz, Christina Imdahl, Remco Dijkman, Douniel Lamghari-Idrissi, Willem van Jaarsveld; *Deep Controlled Learning for Inventory Control*. European Journal of Operational Research 324(1), 104-117 (2025). DOI: `10.1016/j.ejor.2025.01.026`.
- Sources: [author arXiv record](https://arxiv.org/abs/2011.15122), [publisher record](https://www.sciencedirect.com/science/article/pii/S0377221725000463).
- Assistant status: **Opened**, arXiv v7 abstract, complete metadata, journal-reference field and related publisher DOI. Direct publisher access returned 403; indexed publisher abstract was also inspected.
- Support: approximate policy iteration, sequential halving and common random numbers, C005. No numerical optimality-gap result is imported. Older arXiv snippets differ from the latest abstract, so the current record controls. Human pending.

### E019: Johannink2019ResidualRL

- Metadata: Tobias Johannink, Shikhar Bahl, Ashvin Nair, Jianlan Luo, Avinash Kumar, Matthias Loskyll, Juan Aparicio Ojea, Eugen Solowjow, Sergey Levine; *Residual Reinforcement Learning for Robot Control*. ICRA (2019). DOI: `10.1109/ICRA.2019.8794127`.
- Sources: [author arXiv record](https://arxiv.org/abs/1812.03201), [IEEE record](https://ieeexplore.ieee.org/abstract/document/8794127/figures).
- Assistant status: **Opened**, arXiv author list and abstract. Conference/year/DOI cross-checked against indexed IEEE metadata; DOI direct open failed.
- Support: superposition of conventional and learned control signals, demonstrated on robot assembly, C005. Action residuals are distinguished from cost corrections; no stability or patient-protection guarantee is imported. Human pending.

### E020: Kotecha2025GNNMARL

- Metadata: Niki Kotecha and Antonio del Rio Chanona; *Leveraging graph neural networks and multi-agent reinforcement learning for inventory control in supply chains*. Computers & Chemical Engineering 199, 109111 (2025). DOI: `10.1016/j.compchemeng.2025.109111`.
- Sources: [author arXiv record](https://arxiv.org/abs/2410.18631), [publisher record](https://www.sciencedirect.com/science/article/pii/S0098135425001152).
- Assistant status: **Opened**, arXiv metadata and abstract; journal/year/DOI cross-checked against indexed publisher record. Direct publisher open returned 403.
- Support: graph state representations, multi-agent RL, and adaptive heuristic-policy parameters, C007. Its inventory problem is not equated with patient-identity-bound manufacturing, and it does not establish a universal GCN advantage. Human pending.

### E021: Tolstaya2020

- Metadata: Ekaterina Tolstaya, Fernando Gama, James Paulos, George Pappas, Vijay Kumar, Alejandro Ribeiro; *Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks*. PMLR 100, 671-682 (2020).
- Source: [PMLR original record and BibTeX](https://proceedings.mlr.press/v100/tolstaya20a.html).
- Assistant status: **Opened**, metadata and abstract. The earlier local decentralized-literature note was reused only as a discovery/interpretation aid, not substituted for this source check.
- Support: expert imitation, local communications, and multihop information, C008. Claims about matched non-graph controls are this section's experimental-design reasoning, not a result attributed to Tolstaya et al. No DOI supplied by the opened proceedings record. Human pending.

### E022: Brody2022GATv2

- Metadata: Shaked Brody, Uri Alon, Eran Yahav; *How Attentive are Graph Attention Networks?* ICLR (2022), arXiv `2105.14491`.
- Source: [author arXiv record](https://arxiv.org/abs/2105.14491).
- Assistant status: **Opened**, metadata, abstract, and ICLR-publication comment. Locator: abstract's query-independent attention ranking and dynamic GATv2 formulation.
- Support: future representation question in C009. Expressiveness is not a cost or patient-service result. No conference DOI asserted. Human pending.

### E023: Elmachtoub2022SmartPredict

- Metadata: Adam N. Elmachtoub and Paul Grigas; *Smart "Predict, then Optimize"*. Management Science 68(1), 9-26 (2022 issue). DOI: `10.1287/mnsc.2020.3922`.
- Source: [INFORMS original record](https://pubsonline.informs.org/doi/10.1287/mnsc.2020.3922).
- Assistant status: **Opened**, author list, abstract, article information and issue details. Locators: Abstract, SPO decision-error loss and SPO+ surrogate.
- Support: downstream decision error differs from prediction error, C010/C011. No claim that the current method optimizes SPO/SPO+ or inherits consistency guarantees.
- Date discrepancy retained for human review: page says online 12 March 2021 and its auto-generated citation uses 2021; issue is January 2022. BibTeX consistently uses the final issue year 2022. Human pending.

### E024: Donti2017TaskBased

- Metadata: Priya L. Donti, Brandon Amos, J. Zico Kolter; *Task-based End-to-end Model Learning in Stochastic Optimization*. Advances in Neural Information Processing Systems 30, 5484-5494 (2017), arXiv `1703.04529`.
- Source: [author arXiv record](https://arxiv.org/abs/1703.04529).
- Assistant status: **Opened**, metadata, original abstract, conference comment and journal-reference field.
- Support: task-objective model learning and an inventory example, C010/C011. This is methodological context, not evidence that a generic cost-correction fit is end-to-end differentiation through an optimizer. No conference DOI asserted. Human pending.

### E025: Packer2018Generalization

- Metadata: Charles Packer, Katelyn Gao, Jernej Kos, Philipp Kraehenbuehl, Vladlen Koltun, Dawn Song; *Assessing Generalization in Deep Reinforcement Learning*. arXiv `1810.12282` (2018). DOI: `10.48550/arXiv.1810.12282`.
- Source: [author arXiv record](https://arxiv.org/abs/1810.12282).
- Assistant status: **Opened**, metadata and abstract. Initial submission: 29 October 2018; current revision: 15 March 2019. Kept as a 2018 preprint; no unverified conference acceptance asserted.
- Support: distinction between in-distribution and out-of-distribution generalization, C012. No manufacturing-specific robust-control result is inferred. Human pending.

### E026: Henderson2018DRLMatters

- Metadata: Peter Henderson, Riashat Islam, Philip Bachman, Joelle Pineau, Doina Precup, David Meger; *Deep Reinforcement Learning That Matters*. AAAI 32(1) (2018). DOI: `10.1609/aaai.v32i1.11694`.
- Sources: [author arXiv record](https://arxiv.org/abs/1709.06560), [AAAI proceedings record](https://ojs.aaai.org/index.php/AAAI/article/view/11694).
- Assistant status: **Opened**, arXiv metadata and original abstract; publication date/venue/DOI cross-checked against indexed AAAI record. Direct AAAI open was denied.
- Support: variability and reporting/reproducibility concerns, C013. The training-block/world distinction is our design interpretation, not a specific bootstrap recipe taken from this paper. Human pending.

### E027: Agarwal2021RLiable

- Metadata: Rishabh Agarwal, Max Schwarzer, Pablo Samuel Castro, Aaron Courville, Marc G. Bellemare; *Deep Reinforcement Learning at the Edge of the Statistical Precipice*. NeurIPS (2021), arXiv `2108.13264`.
- Source: [author arXiv record](https://arxiv.org/abs/2108.13264).
- Assistant status: **Opened**, metadata, original abstract and NeurIPS-publication comment.
- Support: few-run statistical uncertainty, interval estimates, performance profiles and robust aggregate summaries, C013. No new interval, significance test, or ranking was computed here. No proceedings DOI asserted. Human pending.

## Claim Registry

Each row binds the single physical paragraph line immediately after its C-ID comment in `related-work.tex`. SHA-256 input is that UTF-8 paragraph line after trimming and collapsing all whitespace to single spaces. TeX source, citation commands, punctuation and case are preserved. Hashes detect subsequent wording changes; they neither prove truth nor certify human review. All rows remain human-review pending.

| Claim | Evidence | Type and boundary | Paragraph SHA-256 |
|---|---|---|---|
| C001 | E001,E009,E010; current main's Introduction for present problem scope | Prior modeling precedents plus explicit current-study scope; E001 direct check incomplete | `9cc8571710da17426067ad90778b87a0ef32e539c40826465b4573302bdb6c1d` |
| C002 | E011,E012,E013 | Process/accounting evidence, cost modeling and cross-product limits; no numeric calibration | `3c0ad63ff8b394ccdde76638571e58912b5a951c24375586c4600274bca4fba1` |
| C003 | E014 | Observational clinical context; causal and synthetic-outcome limits are interpretation | `cba0b195b2202f5e38b2adf9d87ab24a57b6c74a380d21bc52c303486be6e96e` |
| C004 | E002,E015,E016 | Literature methods and bounded findings; E002 direct check incomplete | `d93d6ba1ab098d3a32c01384e827a2b363a31556185dac59c136991228d90ca9` |
| C005 | E017,E018,E019 | Distinct hybrid, policy-iteration and action-residual mechanisms; no guarantee transfer | `b8d604f3a9b7905dbad86e7505ccc13b7e136375fe223f4e266eee4389b0a25a` |
| C006 | E003,E005,E006,E007; current main's Introduction for correction definition | Method distinction and matched-planner attribution; not new implementation claims | `16ec1bfd1efe16829af6b43566e3ef31c717be81eab3f81916e3ea24a9cbaecd` |
| C007 | E004,E020 | Graph/inventory precedents and current design rationale; no first-ever claim | `29d098c0ed3f5721ae1c8bdfd166fe4c2a77e3b301d73355ba175efe506c7fa2` |
| C008 | E021 | Imitation/communication precedent; information-matching requirements are our inference | `699a242e34e08ac7fad40cdf263a144f24201032046a6d1d0fe812e56d39dc83` |
| C009 | E022 | Expressiveness result; application and service gain explicitly untested | `83a398e314a796df871c5365c49f870374188b67fcb1b9969b27b100d6dfd74f` |
| C010 | E023,E024 | Decision-focused methodological context; no implementation/theorem inheritance | `5603cd2ca3b9b2c8e238b67e40a1009ab2cdd8b674386ab1fa479a5cb976ce8a` |
| C011 | E001,E010,E014,E023,E024 | Operational interpretation and reporting criteria, not a new clinical or numerical finding | `0d1592ac02487d37ae02e7362cefce27ab4f49d2f7cc7dc0a4c2ce4b2c46bf10` |
| C012 | E002,E016,E025 | Prospective generalization questions; no new uncertainty test performed | `e67bd0b0881e90106b49dc4296757eb878a9b125c64aa7f90ae17d299f92b68e` |
| C013 | E026,E027 | Literature statistical cautions plus design interpretation; no new statistical analysis | `682bcb6c4192131af4e4e28c9955fc89a51c36211d47e26e66d737496eb8039b` |

## Reuse and Exclusions

Local discovery sources were `historical-research-record.tex`, historical `references.bib`, `docs/team_updates/2026-10-06-literature-engineering-evidence.md`, and `reports/2026-10-25-gcn-communication-ablation-review/decentralized-literature.md`. The current main's Introduction was read to distinguish its candidate-cost correction from earlier action-residual and TD-tail methods. Historical claims of broad novelty, clinical efficacy, automatic algorithm gains, and retrospective calibration were not copied. Irrelevant journal-launch editorials, market forecasts, and algorithm catalogues were excluded.

The scientific-writing and manuscript-writing-review skills were reused in place; nothing was installed. A prose pass removed promotional comparisons, kept technical terms stable, and separated literature findings from our inferences. This is not a systematic review and does not certify exhaustive literature coverage.

## Honest Gaps and Parent Handoff

- E001/E002 need direct source confirmation by an accountable author; their bibliography entries are preserved because the existing main requires the keys. Indexed primary text is not marked fully verified.
- E010/E014 were checked against the authors' original abstracts through NLM, E016 through its author-institution record, and E017 through a deposited publisher PDF. Human review should confirm the publisher versions. E012 supports only abstract-level cost-model claims here; no inaccessible Table 4 failure assumptions are asserted.
- Confirm date/style choices for E002, E007, E009, E017, E023 and E025 when applying the journal bibliography style. In particular, online-first years and final issue years are not silently interchangeable.
- All exact claim/source mappings still require accountable-human review. No authorship, funding, ethics, AI-policy compliance, or submission-readiness declaration is supplied by this assignment.
- Howard's suggestion, as supplied by the user, motivates the prospective uncertainty paragraph C012. It is not an authorized new experiment or evidence that broader uncertainty has already been tested. No SMS account was accessed.
- GATv2 and solver literature motivate possible future comparisons only. Neither guarantees a gain, a three-percent improvement, clinical benefit, or universal GCN superiority.
- Parent owns `main.tex`, all figures, numerical claims, compilation, final integration, and any separately authorized GitHub action. Existing historical bibliography stays untouched. The parent has reported inserting this section; this assignment did not perform that edit.

## Static Checks

Text-only inspection found 27 unique BibTeX keys, 26 distinct related-work citations, no unresolved section citations, and all eight required existing keys. The TeX/BibTeX files use ASCII with standard LaTeX accent escapes. No research code was changed. A complete LaTeX/BibTeX build and journal-format check were not run by this assignment.
