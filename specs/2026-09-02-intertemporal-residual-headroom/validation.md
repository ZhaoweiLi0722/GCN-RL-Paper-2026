# Validation Without Scientific Execution

Run from the repository root:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
python3 -m unittest tests.test_intertemporal_residual_headroom
python3 -m evaluation.screen_intertemporal_residual_allocation_headroom --describe
```

The `--describe` command validates immutable hashes and reports expected row
counts without constructing an environment or consuming a reserved seed.

The ordinary execution command must currently fail with `PermissionError`
naming Howard as the pending approval, before state generation and before the
result root is created:

```bash
python3 -m evaluation.screen_intertemporal_residual_allocation_headroom
```

Scientific execution requires a reviewed commit changing only Howard's
approval record from `false` to `true`. Passing this screen still does not
authorize DDPG training. The configured result root is single-use: if it
already exists, execution refuses to overwrite it.
