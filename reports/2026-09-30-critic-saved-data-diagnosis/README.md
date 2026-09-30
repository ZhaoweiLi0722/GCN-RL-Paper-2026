# R6 saved-data diagnosis evidence

The canonical derived result is `diagnosis.v2.json`. `diagnosis.repeat.json`
is its exact repeat, not another experiment. `verification.json` independently
checks raw arithmetic. `diagnosis.json` is an intentionally retained invalid
partial file from the serialization failure described in
`serialization_failure.md`; do not load it as a result.

Readout and limits:
`specs/2026-09-30-critic-saved-data-diagnosis/readout.md`.
No new simulation, inference, fitting, reward modification or online-RL claim.
