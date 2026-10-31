# Multi-Topology Training: Completed, Fixed Screens Not Passed

Read-only verification on 2026-10-22 UTC. This nonlocked readout summarizes the
existing comparison and archive; it adds no scientific computation.

## Execution And Preservation

- Root: `results/native_gcn_multitopology_training_20261021`.
- Packet: `177f18b5090b93040444ebde0a55de39124ebb143eca43b4f285b2d7938dc57d`.
- Terminal: completed at 2026-10-22T00:45:44.044242+00:00, or October21
  20:45 EDT. Launcher elapsed including IO: 14937.656740 seconds, about4h9m.
- Worker completion receipts: six collection jobs/192 main trajectories,
  twelve fits/12288 updates, twenty-one evaluation jobs/504 trajectories.
  Twelve new short finals and six retained full endpoints; evaluation updates0.
- All raw work, matched initial tensors/batch indices, labels, topology
  minibatches and zero-update evaluation passed the existing reconciliation.
- Read-only host full-command ps found no matching run process or its bound
  caffeinate. Every supervisor/worker PID from current claims was absent.
- The unique archive, comparison and inventory SHA256/bytes match the saved
  receipt. Reused its full12276-member verification; no archive was rebuilt.

| Artifact | Bytes | SHA256 |
| --- | ---: | --- |
| archives/payload.tar.gz | 568303624 | 83d788ef302b853037bc6ea362c5822c6decdf991487dbe54e473aaef053ea06 |
| payload/comparison.json | 58552067 | e11896c27f2c7c3425aa9d29bb2c9949c0e009eb806386cdaea89fbfa4c20969 |
| payload/inventory.json | 2497024 | f81e8cd98a060ae50e074f8e8feced73de009ca9147e0c46f97e732f152a106f |

Actual accounted work includes98428 native steps,6799196 prediction steps,
2284 branch tails,104688 advantage forward calls and12288 optimizer steps.
The upper bounds are not presented as realized work. All63 panel and21 held-out
aggregate comparisons and their block/condition/world harms remain preserved.

## Held-Out Results

These are the existing aggregate results over two fixed unseen connection
combinations:48 physical settings from24 paired seed tuples, three training
blocks. Positive cost gain means the candidate costs less than its reference;
negative patient difference means fewer simulated patient losses per trajectory.

| Candidate / reference | Cost gain | Mean patient-loss difference |
| --- | ---: | ---: |
| Multi-topology Graph / matched multi-topology Self | 0.437969% | -12.520833 |
| Source-only short Graph / matched source-only short Self | 0.298892% | -8.041667 |
| Multi-topology Graph / source-only short Graph | -0.104875% | -8.291667 |
| Multi-topology Self / source-only short Self | -0.244710% | -3.812500 |
| Multi-topology Graph / retained full Graph | -0.190082% | -0.625000 |
| Multi-topology Graph / H8 | 1.377407% | 16.270833 |

The matched multi-topology Graph/Self cost-gain descriptive95% interval is
[-0.018297%,1.166060%]; the patient-difference interval is[-28.3125,0.9375].
Block cost gains are1.166060%,0.097380% and-0.018297%. The Graph-specific
training interaction is0.140149 percentage points, interval[-0.863899,1.087411],
below the prespecified0.25-point threshold.

Thus the wider Graph/Self gap does not show absolute Graph cost improvement:
Graph worsened by0.104875% and Self worsened by0.244710% against their matched
source-only short models. Graph's pooled patient loss improved, but multiple
condition-level patient guards failed. Compared with H8, costs fell while mean
patient losses increased16.270833;35/48 settings had greater patient loss.
Matched Graph/Self also had17/48 patient-harmed and21/48 cost-harmed settings.

The retained full model is a practical reference, not a matched training
attribution:96 training worlds/3072 updates versus32worlds/1024updates here.
On the original topology, multi-topology Graph cost increased0.247153% and
patient losses increased0.875 against retained Graph. Do not pool original and
held-out panels or replace historical results with this new development sample.

## Decision

Both fixed mechanism and practical screens failed. The existing program's
decision is `archive_close_fixed_recipe_negative_or_inconclusive`.
Close this fixed multi-topology recipe; no automatic longer run, retries,
alternative-edge selection, threshold relaxation, goal resumption or manuscript
claim. Preserve the old Graph/Self and all new negative results.

This is completed science with a negative/inconclusive development result, not
an implementation failure. Three blocks and two synthetic topology changes do
not establish the original-task3% goal, corrected significance or clinical
safety. Any new direction needs a separately defined scope and authorization.
No remote Git, collaborator message, cloud delivery or automation was changed.
