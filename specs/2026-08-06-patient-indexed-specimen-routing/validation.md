# Validation

## Local Validation Commands

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache python3 -m compileall .
python3 -m unittest tests.test_patient_indexed_specimen_routing
python3 -m unittest tests.test_patient_indexed_specimen_routing_mechanics_gate
python3 -m unittest tests.test_patient_indexed_specimen_routing_contract
python3 -m unittest tests.test_gcn_facility_action
python3 -m unittest tests.test_off_policy_training_state
```

PowerShell syntax is checked without executing the runner:

```powershell
$paths = @(
  "scripts\run_patient_indexed_specimen_routing.ps1",
  "scripts\start_patient_indexed_specimen_routing_phase.ps1",
  "scripts\invoke_patient_indexed_specimen_routing_phase_detached.ps1"
)
foreach ($path in $paths) {
  $tokens = $null
  $errors = $null
  [System.Management.Automation.Language.Parser]::ParseFile(
    $path,
    [ref]$tokens,
    [ref]$errors
  ) | Out-Null
  if ($errors.Count -ne 0) { throw ($errors | Out-String) }
}
```

## Required Automated Evidence

- immutable identity and lifecycle conservation;
- no pooling, splitting, substitution, multi-hop, cycle, or in-production route;
- explicit-edge enforcement, integer conservation, and deterministic ties;
- lead-zero/lead-one arrival, single aging, expiry, ineligibility, and loss;
- same identity at destination manufacturing and origin return;
- finite route cost and complete event metadata;
- routing-independent exogenous RNG state under paired seeds;
- full state and exact next-step restoration;
- frozen disabled-mode hash from parent `ce9b627`;
- forced nonzero legal route and MDL-2 headroom;
- matched GCN/flat observation, action, specimen head, and parameter budget;
- valid configs, isolated result paths, and no legacy routing checkpoint input;
- routing-primary attribution pairing and final-versus-pretrain arithmetic;
- runner AST parse and static no-force/no-overwrite contract.

No formal training or evaluation is part of repository validation.

## Recorded Implementation Validation

Validation was run from the isolated routing worktree on 2026-08-06 with
Python 3.9.6, PyTorch 2.8.0, and no routing result directory present:

- `python -m unittest discover -s tests`: 480 tests passed in 33.565 seconds;
- focused routing/contract suite: 39 tests passed in 0.445 seconds;
- `python -m compileall -q .`: passed;
- `jq empty experiments/configs/patient_indexed_specimen_routing_*.json`:
  passed;
- `git diff --check`: passed;
- PowerShell AST parse: passed with 2,832 tokens and zero parse errors;
- benchmark `routing_smoke` dry run: 3 algorithms, 4 default routing scenarios,
  1 seed, 8 learned training jobs, and 12 evaluation jobs.

The mechanics runner itself was not executed locally because it intentionally
writes the first artifact in the new result namespace. No teacher generation,
smoke training, pilot training, formal evaluation, or packaging was performed.
The manuscript protocol wording was subsequently updated, but no unobserved
routing result was inserted.

## Recovery 1 Repair Validation

After the first formal teacher attempt exposed the missing state-probe config
contract, Recovery 1 validation was run before any remote retry:

- full repository suite: 480 tests passed in 37.138 seconds;
- focused routing, mechanics, contract, action, training-state, and headroom
  suite: 48 tests passed in 0.492 seconds;
- routing contract suite alone: 7 tests passed in 1.683 seconds;
- a scratch `--smoke --skip-teacher` execution of the repaired routing teacher
  config completed three state-probe steps without a missing-key failure;
- `python -m compileall -q .`: passed;
- every patient-indexed routing JSON config parsed with `jq empty`;
- PowerShell AST parse: 2,860 tokens and zero parse errors;
- benchmark `routing_smoke` dry run retained 8 learned training jobs and 12
  evaluation jobs, all below the Recovery 1 output root;
- `git diff --check`: passed.

The scratch probe wrote only to `/private/tmp` and is not formal evidence. No
teacher generation, learned-policy training, or formal evaluation was executed
locally as part of this repair.

## Recovery 2 Launcher Validation

Recovery 2 preparation was validated against parent commit
`ecab3650780aafb746027fb26f2f02512b7b0495` before any PC launch:

- all 15 patient-indexed routing JSON configs were parsed structurally after
  replacing the Recovery 1 and Recovery 2 root strings with one placeholder;
  scientific config mismatches: 0;
- routing contract suite: 8 tests passed in 0.624 seconds;
- focused routing, mechanics, contract, action, training-state, and headroom
  suite: 49 tests passed in 0.507 seconds;
- full repository suite: 481 tests passed in 38.063 seconds;
- `python -m compileall -q .`: passed;
- every patient-indexed routing JSON config parsed with `jq empty`;
- main runner AST: 2,880 tokens and zero errors;
- detached launcher AST: 536 tokens and zero errors;
- detached exit wrapper AST: 348 tokens and zero errors;
- benchmark `routing_smoke` dry run retained 8 learned training jobs and 12
  evaluation jobs below the Recovery 2 root;
- a `/private/tmp` stub-runner check confirmed immediate launcher PID/path
  output and confirmed that the wrapper persists both exit 0 and exit 120 in
  status JSON without changing the exit code;
- `git diff --check`: passed.

macOS PowerShell does not support Windows `Start-Process -WindowStyle Hidden`,
and the local command sandbox reaps detached children. Therefore long-lived
background survival is a PC Validate launch acceptance check. The Mac test did
not run the scientific phase runner or create any formal result artifact.

## Recovery 3 Split-Execution Preparation

Recovery 3 preparation was validated before any formal Mac shard was launched:

- all 15 patient-indexed routing JSON configs were parsed structurally after
  replacing the Recovery 2 and Recovery 3 root strings with one placeholder;
  scientific config mismatches: 0;
- Mac pipeline, bundle, headroom, and routing contract suites: 29 tests passed;
- full repository suite: 495 tests passed in 34.489 seconds;
- `python -m compileall .`: passed;
- bundle extraction retains a verified `bundle_manifest.json` transfer receipt;
- state-probe and teacher commands are mutually exclusive shard modes;
- phase claims are single-use and every shard receives separate stdout,
  stderr, status, Python cache, and Matplotlib cache paths;
- `CUDA_VISIBLE_DEVICES` is empty for every Mac CPU shard;
- the PC phase sequence contains `ImportTeacher` and no formal `Teachers`
  phase.

PowerShell AST validation and detached import acceptance remain PC preflight
checks because PowerShell is unavailable in the Mac environment. No formal
state probe, teacher, CUDA training, or evaluation was run during preparation.
