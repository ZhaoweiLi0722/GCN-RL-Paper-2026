# R5 engineering summary failure, preserved

Source `f089fd0b56b3e7eed36b26b79438234a103addca` exited 1 after all 16 seed60
observations had been evaluated on CPU-policy, CPU-full-state and MPS-policy.
`seed60_raw.json` was written before summary generation; parameter immutability
had been checked, and the continuous/discrete comparisons ran before the error.
No seed61/62 inference began. No environment step or learning update occurred.

The single-output gate stores a scalar margin, while the summary assumed a
list: `TypeError: 'float' object is not iterable`. This is a report-shape bug,
not a training failure or observed parity failure. Preserve the prior root,
traceback and raw file hashes in the continuation config.

The routine engineering repair normalizes scalar/vector margins identically
and adds regression tests. Complete the **unvisited** two seeds after committing
the repair, reusing seed60 bytes without any model re-execution. No tolerance,
observation index, baseline, budget, action or scientific parameter changes.
Total planned inference remains 48 distinct archived observations, not 64.
This bounded completion is not permission to relaunch a failed scientific
campaign or collect R3 outcomes. Original failure records are not overwritten.
