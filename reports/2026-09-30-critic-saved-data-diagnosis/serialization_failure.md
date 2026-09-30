# Saved-data report serialization failure

The initial read-only analysis at commit `51eb657` completed its computations
and before/after source checks, then exited 1 while JSON-serializing a NumPy
integer produced by summing NumPy comparison booleans. The partial
`diagnosis.json` is intentionally retained, is invalid JSON, and is not a
result. Its terminal error was:

```text
TypeError: Object of type int64 is not JSON serializable
```

This is a reporting defect, not a new R6 experiment or scientific failure.
No environment step, model inference/update, selection, or new label was made.
The repair converts the incoming prediction difference to a Python float and
serializes before opening the output, with a regression test. Recalculation
uses the same saved files and fixed diagnostic definitions, writes versioned
outputs, and does not overwrite the partial file or any R6 evidence.
