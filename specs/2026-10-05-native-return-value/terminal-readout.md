# Native-Return Value Comparison: Terminal Readout

## Decision

The approved single attempt completed normally on 2026-10-06 at approximately
07:19 UTC. The prespecified primary development screen FAILED. This is a
completed comparison with mixed scientific findings, not an engineering abort.
Native TD did not establish the required persistent-change cost advantage over
plain H8, the existing frozen GCN, or matched forecast TD. Against forecast TD,
mean patient losses increased in all three conditions. Retain these negative
findings alongside the favorable secondary TD/MC and fast-fluctuation contrasts.
No retry, reward revision, extra training, or new experiment follows automatically.

Approval was Zhaowei's exact `批准此完整单次包`, not approval on Howard's behalf.
Scientific implementation: `7afafe821fb24fbdfb683a10dda5e199ada16b5c`;
frozen implementation: `9a1dac67323622d13b693ff2d7d6a4e4209ac9b8`;
execution: `02e69207578265c34239abc6b0886c951c3de1bd`;
packet: `202a93608a136859b30f9a54b026b766d00f591249d1d054dc10e7885bdb90fa`.

## Execution And Preservation

Root: `results/capacity_native_tail_20261005`. Supervisor and child exited zero;
`launcher/terminal.json` confirms scientific completion, no automatic retry,
and 31,159.209742 seconds (8.6553 hours), including recording and archival.
At 07:19:24.904706 UTC, the host process scan found neither PID 36670 nor 36695
and no related experiment process. All failure, overrun, and preservation-error
markers are absent. Final stderr, stdout, and detached launcher log are 0 bytes.

- Reference worlds: 120/120; native dependent branches: 240/240.
- Forecast TD, native TD, native MC: each 120 fits and 3,840 updates; total
  11,520 value updates, zero actor updates. Each model received 768 new updates.
- Five ancestor bindings and 15 final models sealed before any evaluation.
  Reuse the verified 360 after-fit states and all-sealed barrier in
  `reports/capacity_native_tail_monitor_20261006T0342Z.json`.
- Frozen evaluation: 360/360, 60 per role and 72 per block; zero test updates.
- Native steps: 30,720 main + 9,720 branch = 40,440. Native operations:
  31,680 main + 9,720 branch = 41,400, plus 240 state clones.
- Neural forwards: 40,045. Prediction steps: 16,303,440, comprising 9,953,280
  main, 4,723,200 forecast-parent, 1,574,400 native-parent, 5,760 prefix, and
  46,800 paired-tail steps. Parent decisions: 12,300 forecast and 4,100 native.
- Filter transitions: 6,960,000. All counter totals equal their frozen caps;
  no pending chunks, refunds, phase transfers, or observed budget violations.
- Phase seconds: admission 5.190; reference/labels 16,542.284; fitting 77.560;
  frozen evaluation 14,436.126; analysis/archive 97.586; failure reserve unused.
  All 855 completed job-time caps and phase-time caps reconciled.
- Final run tree: 4,915 files, 1,477,438,804 bytes, below 9,000 files/12 GiB.

The existing single archive is reused, not regenerated:
`archives/payload.tar.gz`, 631,668,661 bytes, 4,905 members,
SHA256 `b31fff4305ede5f7f310c564ba41d0e4868c6d87bf0bb3775bba101dec90eeaa`.
The program's receipt confirms every member was read and verified. Inventory
and manifest bindings were independently reconciled without repeating that scan.

## Independent Saved-Data Reconciliation

`terminal-saved-data.json` records the independent terminal read at
07:26:48 UTC. It read all 480 main raw trajectories (30,720 rows), all 240 native
branch raw files (9,720 rows), and all 7,800 native suffix rows. It checked 960
main raw/summary hashes and 240 branch raw hashes; cost-component sums, reward
sign, patient identity/status and loss increments, settlement, requested versus
committed versus actually applied hours, two-step implementation lag, and
decision timing all reconcile with the complete saved comparison.

All 15 controller pairs were independently reconstructed across 3 conditions
and 5 blocks: 900 paired-world contrasts, 45 condition rows, and 225 block rows.
Full costs, patient outcomes, components, actions, actual hours, and adverse
world/block counts are retained in that JSON and `payload/comparison.json`.
The existing five-block, 2,000-resample two-level descriptive intervals are
reused unchanged; there are no new bootstrap draws or historical three-block
substitutions. No model was deserialized and no scientific forward, environment,
optimizer, or clone call was made by the terminal reader.

Comparison SHA256:
`2f2c241a99215c2c21747bada6963b64acf86e0474ebcf6b039abddaac8ad863`.
Full budget-file SHA256:
`77083f0b0846bfe2009d0def28a11f5ab7933f1000428f4e2bc2c6d99373b3ad`.
This file hash is distinct from the ledger's final chain digest.

## Results

Candidate is native TD throughout. Savings are comparator minus candidate
cost, in millions of synthetic objective units. Percentages are mean paired-world
percentages, not ratios of pooled means. Extra losses are native TD minus
comparator; positive means worse. Every row has 20 paired worlds across 5 blocks.
Intervals are descriptive, not multiplicity-adjusted confirmatory evidence.
Conditions: 0 no change, 1 persistent change, 2 fast fluctuation.

| Comparator | Condition | Savings M [descriptive 95%] | Savings % | Extra losses | Positive cost blocks | Cost/loss harm worlds |
|---|---:|---:|---:|---:|---:|---:|
| Plain H8 | 0 | 0.049 [-0.568, 0.895] | 0.354 | -11.05 | 2/5 | 11/3 |
| Plain H8 | 1 | 0.894 [-0.017, 1.845] | 1.813 | -8.70 | 4/5 | 6/4 |
| Plain H8 | 2 | 1.073 [0.031, 2.096] | 2.470 | -17.95 | 5/5 | 4/0 |
| Existing frozen | 0 | 0.366 [-0.323, 1.154] | 0.824 | -2.20 | 3/5 | 11/6 |
| Existing frozen | 1 | 0.274 [-0.174, 0.764] | 0.571 | -0.95 | 4/5 | 7/6 |
| Existing frozen | 2 | 0.608 [0.097, 1.113] | 1.515 | -2.50 | 5/5 | 5/5 |
| Forecast TD | 0 | -0.118 [-0.755, 0.600] | -0.408 | +2.75 | 2/5 | 11/10 |
| Forecast TD | 1 | 0.244 [-0.431, 0.939] | 0.462 | +2.40 | 3/5 | 8/15 |
| Forecast TD | 2 | 0.287 [-0.244, 0.841] | 0.684 | +1.90 | 4/5 | 7/13 |
| Native MC | 0 | 0.944 [-0.036, 2.384] | 2.301 | -13.75 | 3/5 | 8/3 |
| Native MC | 1 | 0.955 [0.171, 1.738] | 1.880 | -11.95 | 4/5 | 6/4 |
| Native MC | 2 | 0.895 [-0.322, 2.373] | 2.253 | -18.50 | 3/5 | 9/3 |
| Plain H16 | 0 | -0.176 [-0.832, 0.616] | -0.425 | -0.40 | 2/5 | 11/8 |
| Plain H16 | 1 | 0.403 [-0.322, 1.106] | 0.766 | -1.55 | 4/5 | 7/8 |
| Plain H16 | 2 | 0.622 [-0.056, 1.375] | 1.384 | -7.15 | 4/5 | 9/4 |

All three primary persistent-change cost intervals include zero; positive
block counts are 4/5, 4/5, and 3/5, rather than the required 5/5. Against forecast
TD, all condition-mean extra losses are positive. That fails the sample-mean
criterion, but their descriptive intervals include zero and do not establish
population-level harm: [-0.05, 6.50125], [-0.55, 4.65], [-2.00125, 5.9].

The secondary same-native-data TD/MC contrast under persistent change is
favorable at this fixed budget: cost saving 0.955334 M [0.171109, 1.738155],
mean paired saving 1.880452%, and 11.95 fewer losses (extra-loss interval
[-25.75125, -0.04875]). Four cost blocks improve, but six worlds cost more and
four have extra losses. This supports a limited TD-target advantage over this
direct-return-regression control, not general RL superiority or isolated GCN
causality. TD and MC share suffix data and update counts, not forward compute.

Fast fluctuation has favorable cost intervals against H8 and the existing
model. Against H8, extra-loss interval is [-24.3, -12.04875]; against the existing
model it is [-5.95, 0.95125], including zero. All five cost block means favor TD,
but adverse individual worlds remain. These secondary results do not repair
the failed primary screen. Every H16 cost interval includes zero; this is not
evidence of equivalence.

## Actions, Labor And Compute

Under persistent change, native TD and H8 differ at 701/960 control boundaries.
Native TD applies 69.2806 more flexible hours per world. Mean component savings
include 434,866.64 bioreactor-shortage, 435,000 patient-loss, and 306,000 expiry
units, offset by 288,891.91 more reagent-purchase, 47,577.49 more reagent-shortage,
and 7,800.48 more flexible-labor cost. The cost gain is not free labor.

Against forecast TD, native TD applies 13.7985 fewer hours under persistent
change while losing 2.40 more patients; improved bioreactor-shortage cost offsets
worse patient/expiry costs. Against native MC, TD applies 76.1430 more hours and
changes 600/960 boundaries while delivering 11.95 more patients. Against existing
frozen/H16 it applies 9.6000/24.4057 more hours. Full condition/component and
requested/committed/applied contrasts remain in the machine-readable readout.

Evaluation median/p95 decision seconds (2,880 decisions per role): H8
0.6944/0.8599; existing 0.6942/0.8585; forecast TD 0.6930/0.8598; native TD
0.6944/0.8600; native MC 0.6952/0.8640; H16 1.3912/1.6366. H16 consumes twice
the native planning queries. Label-collection and training costs are additional,
including every nested parent decision, not just the 240 branch roots.

## Scope And Handoff

Reward, physical scenarios, architecture, candidate support, samples, and seeds
were unchanged. Native and forecast training states differ, so this is not a
label-only causal contrast or exact correction of model bias. Branches are
dependent simulated continuations, not independent worlds or event-level CRN.
The fixed historical parent is not the changing student's exact on-policy
evaluation. Five reused ancestors are not five new independent training runs;
block zero retains mixed predictor history. E1 calibration remains missing.
No deployment-time adaptation, DDPG/full TD-MPC, isolated GCN contribution,
clinical safety, or publication guarantee is established.

The manuscript source now reports this experiment and its limits. No TeX compiler
is available locally, so a newly compiled PDF is not claimed. Reuse the 43
necessary zero-update tests and compileall from the unchanged frozen source;
there was no reason to repeat science or historical audits.

The existing archive, complete comparison, independent saved-data analysis,
authority, and updated manuscript were copied additively to Research Artifacts/
`capacity_native_tail_20261005` at 07:31:15 UTC. All 25 core files, 801,081,755
bytes, passed destination SHA256/length checks; no differing file was overwritten.
See `reports/dropbox_native_tail_20261006_{plan,receipts,completion}` with their
JSON/JSONL extensions. Terminal handoff/state documents follow as a separate
small additive closure batch, with their own receipt in `dropbox-handoff.md`.
Cloud synchronization and collaborator access remain UNVERIFIED: the current
read-only Dropbox UI request timed out. No permission change or message was sent.
The four closure documents (633,763 bytes) passed destination verification at
07:33:59 UTC. The same visible `gcn-rl` monitor was then PAUSED, confirmed by the
app tool and local configuration readback at 07:34:37 UTC; see
`monitor-closure.json`. Final monitor-closed state documents are exported
additively under `handoff/monitor_closed`, preserving the earlier snapshots. The
single attempt is consumed; subsequent scientific work requires a separately
specified prospective package, not extension of these test worlds.
