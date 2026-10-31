# Preserved Training-Budget Evaluation: Final Readout

Completed at 2026-10-22 02:06:40 America/New_York (06:06:40 UTC).
Elapsed including IO: 6,996.46 seconds, about 1 hour 57 minutes.
This is a completed evaluation with failed/inconclusive research screens,
not an implementation failure.

## Completed Scope

All 35 worker receipts passed raw reconciliation: 840 evaluations,
120 complete seven-role paired worlds and 21 comparisons.
All 30 old checkpoints were reused. New training, labels, fits, optimizer
updates and model seals are zero. All owned processes have exited.

## Saved Comparisons

Positive cost saving favors the first role. Negative patient difference means
fewer losses per 52-step world. The suffix is the historical training-world
budget. These are transcribed saved results, not recomputed comparisons.

| Comparison | Cost saving (%) | Descriptive 95% interval (%) | Patient-loss difference | Descriptive 95% interval |
| --- | ---: | --- | ---: | --- |
| Graph32 / Self32 | 0.451 | [-0.092, 1.110] | -3.50 | [-20.99, 12.42] |
| Graph64 / Self64 | 0.073 | [-0.335, 0.646] | -0.56 | [-17.76, 13.64] |
| Graph96 / Self96 | 0.462 | [-0.209, 1.226] | -9.68 | [-35.00, 12.74] |
| Graph64 / Self96, primary | 0.266 | [-0.416, 1.004] | -4.82 | [-26.69, 18.88] |
| Self64 / Self96, control | 0.193 | [-0.540, 0.998] | -4.26 | [-28.57, 18.39] |
| Graph64 / H8 | 2.324 | [1.714, 3.021] | -11.11 | [-30.68, 7.38] |
| Graph96 / H8 | 2.516 | [1.623, 3.326] | -15.98 | [-36.63, 10.64] |

## Conclusion

The fixed primary, usefulness and GCN-attribution screens all failed.
Graph64/Self96 and Graph64/Self64 have favorable means, but cost intervals
cross zero, patient-loss upper bounds are positive, and some condition means
worsen. Graph64/H8 passes the cost checks but fails the patient checks.

This package does not establish that Graph can use one-third fewer main
training worlds and updates while matching Self, nor a 3% GCN advantage.
Same-budget Graph96/Self96 remains a small, uncertain positive mean difference.
Graph/H8 is a combined learned-method comparison, not isolated GCN evidence.
Unchanged old models were evaluated on fresh worlds; a changed percentage
is not an improvement to the model.

DEVELOPMENT, synthetic-only, five reused training blocks. Intervals are
descriptive, not adjusted significance or clinical safety evidence.
All 21 comparisons and block/condition/world harms remain in comparison.json.
The fixed package is closed; no automatic extension, retry or longer run.

## Preservation

Unique archive: 397,605,966 bytes, SHA256
`8e2d1bed0b9a8e0533abcff3ba096d8b75cb6638fce2a3f113ef18f9edaf7de0`.
Fresh archive/comparison/inventory hashes and bytes match their receipts.
Reuse the existing all-5,979-member readback after exact archive-byte matching.
Inventory has 5,978 entries because it excludes itself. No archive was rebuilt.

Actual new work: 43,680 native and 3,539,890 prediction steps,
87,360 planning clones, 37,440 learner forwards / 209,383 examples.
Peak recorded RSS: 3,626,680,320 bytes, below 4 GiB. All recorded host samples
used AC power. Launcher, archive and worker stderr files are empty; no failure
receipts. Full counts, process identities and hashes are in terminal-readout.json.

No manuscript, remote, cloud, message, goal or automation change was made.
