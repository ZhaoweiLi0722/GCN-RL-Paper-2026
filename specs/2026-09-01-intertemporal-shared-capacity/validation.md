# Validation Matrix

## Flag isolation

- Existing config dictionaries parse unchanged.
- With all new flags off, action size, observation size, snapshots, rewards,
  and fixed-seed trajectories match the parent commit exactly.
- Scheduled waves without intertemporal overtime alter demand only.
- Intertemporal overtime without scheduled waves remains valid for mechanics
  tests, but cannot enter J2.

## Schedule tests

- cluster order, start epoch, block boundary, repeat, and no-repeat behavior;
- future forecast includes the correct scheduled blocks over its horizon;
- malformed clusters, indices, multipliers, and durations are rejected;
- graph and flat observations expose the same forecast values.

## Commitment tests

- raw mapping and shared-simplex projection;
- exact lead-time timing;
- persistence recursion and zero-input decay;
- active capacity rather than current request controls production;
- activation cost and active cost decomposition;
- fatigue uses active utilization;
- no negative capacity, budget overflow, or fleet-accounting violation;
- patient snapshot round-trip including pipeline and active state;
- `facility_state_width` mirrors the environment under every new flag
  combination.

## Minimum commands

```
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
python3 -m unittest tests.test_intertemporal_shared_capacity
python3 -m unittest tests.test_intertemporal_shared_capacity_screen
python3 -m unittest tests.test_overtime_control_contract
python3 -m unittest tests.test_overtime_control_equivalence
python3 -m unittest tests.test_patient_capacity_planning
```

The repository PyTorch environment is used for the final run. Any known
baseline-only device failure is compared against the parent commit and
reported; thresholds are never weakened in this study.
