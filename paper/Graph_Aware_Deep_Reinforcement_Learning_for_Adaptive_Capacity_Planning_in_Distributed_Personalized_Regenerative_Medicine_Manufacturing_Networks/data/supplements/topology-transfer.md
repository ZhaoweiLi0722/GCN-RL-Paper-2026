# Same-Site Topology Transfer: Completed, Screen Failed

2026-10-21. Local execution readout, not manuscript text or a publication claim.
The user requested completion of this round followed by a stop before any goal
revision or further experiment. That boundary remains in effect.

## Completed Scope

The single attempt ran from18:18:29Z to19:02:11Z (2622 seconds, about44minutes).
All20 worker jobs and280 unique52step zero-update trajectories completed:
120 on original connections and160 on the fixed changed connections. These
cover40 paired seed tuples,80 physical settings and five reused training blocks.
All10 original Graph/Self endpoints remain unchanged; no training, labels,
optimizer updates or new model seals occurred. H8 and Self were not weakened.

The intervention removed only the approved bilateral(0,1) qualification.
Target Graph and stale Graph used the same target physical rules, candidates
and H8; they differed only in the Graph adjacency used at inference.

## Results

Values below are transcribed from the program's existing `payload/comparison.json`.
Positive cost saving favors the candidate. Negative patient difference means
fewer simulated patient losses per52step world. Intervals are descriptive95%
paired training-block bootstrap intervals, not adjusted significance tests.

| Panel and contrast | Cost saving | Cost interval | Patient difference | Patient interval |
| --- | ---: | --- | ---: | --- |
| Original Graph / Self | 0.3113% | [-0.0222%, 0.6643%] | -5.125 | [-10.300, 0.600] |
| Target Graph / Self | 0.2202% | [-0.0922%, 0.5183%] | -3.125 | [-6.575, 1.125] |
| Target Graph / stale Graph | 0.0311% | [-0.0413%, 0.0806%] | -0.950 | [-4.000, 2.350] |
| Target Graph / H8 | 3.3360% | [2.3689%, 4.2803%] | -33.400 | [-65.650, -3.275] |
| Target Self / H8 | 3.1227% | [2.0894%, 4.2312%] | -30.275 | [-60.475, -1.000] |

The fixed interaction was-0.0913%, interval[-0.2336%,0.0653%], rather than a
supported widening of the Graph/Self gap. Target Graph saved cost in only3/5
blocks against Self, below the required4/5. Both the0.5% Graph/Self and0.25%
target/stale Graph thresholds failed, as did their lower-interval requirements.

All three patient guards failed. Target Graph/Self condition means were
-28.4,+14.3,+8.0,-6.4;17/40 worlds had higher patient losses. Target Graph/stale
Graph had a+0.4 condition harm and10/40 harmed worlds. Target Graph/H8 had a
+1.4 condition harm and16/40 harmed worlds, despite its favorable pooled mean.
The source artifact retains every world, block, condition, component and all nine
within-panel comparisons. The selected table above does not replace that record.

## Interpretation And Stop

This fixed connection withdrawal did not establish a useful additional benefit
from updating Graph adjacency. The small positive point estimates are not
evidence of equivalence or proof that GCN can never help. The approximately3.34%
Graph/H8 saving cannot be called independent GCN gain: unchanged Self/H8 saved
approximately3.12% on the same target panel.

This is a synthetic DEVELOPMENT supplement with five reused training blocks,
two worlds per condition/block, one link change and fixed learned weights.
Shared candidates and H8 already adapt to the physical graph, and capacity
remains represented through a hub rather than complete typed-edge fidelity.
The original and changed panels are not pooled. This fresh original-panel
estimate does not overwrite any older Graph/Self comparison.

The program's fixed decision is `close_fixed_one_link_recipe`. No other edge,
threshold, model or seed subset will be tried under this approval. No goal was
resumed, modified, deleted or marked complete. The native goal's last-read
status remains `blocked`. Work stops after this readout pending discussion.

## Verification

Full-command host process inspection confirmed supervisor8884, coordinator8900,
their20 claimed workers and sleepguard8898 no longer matched this run. Launcher
and coordinator exited0; terminal status is `completed`, with verified scientific
completion. No failure receipt or nonempty stderr was observed. Runtime aggregate
peak RSS was2248359936bytes, below4GiB. Actual native/prediction steps were
14560/1134212;10400 learner forwards used57427 examples. These are measured
counts, not ceilings. Full operation accounting is in the JSON readout.

The unique archive is `results/native_gcn_topology_transfer_20261021/archives/payload.tar.gz`:
SHA256 `7e741eea23591fc925127c42e3e32e6c477843f85b13a1dfb79916e59b8c04d5`,
126785721bytes,3078members. Fresh archive bytes/hash matched the receipt and
external manifest; the program's full-member verification was reused without
rebuilding or duplicating the archive. Fresh comparison/inventory hashes also
matched their archive member entries. No Dropbox/cloud export or collaborator
access verification occurred. No manuscript, remote, message or automation action.

Evidence: sibling `terminal-readout.json`; run-root `launcher/terminal.json`,
`payload/completion.json`, `payload/comparison.json`, `archive-receipt.json`,
`archives/payload.tar.gz.manifest.json`, worker claims/completion receipts and
the host budget log. Necessary53pure/mock checks and fullcompileall passed before
freeze; no additional scientific test or analysis was run by the monitor.
