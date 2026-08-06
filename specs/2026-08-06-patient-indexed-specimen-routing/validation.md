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
$tokens = $null
$errors = $null
[System.Management.Automation.Language.Parser]::ParseFile(
  "scripts\run_patient_indexed_specimen_routing.ps1",
  [ref]$tokens,
  [ref]$errors
) | Out-Null
if ($errors.Count -ne 0) { throw ($errors | Out-String) }
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
- attribution pairing and graph-interaction arithmetic;
- runner AST parse and static no-force/no-overwrite contract.

No formal training or evaluation is part of repository validation.

## Recorded Implementation Validation

Validation was run from the isolated routing worktree on 2026-08-06 with
Python 3.9.6, PyTorch 2.8.0, and no routing result directory present:

- `python -m unittest discover -s tests`: 481 tests passed in 33.544 seconds;
- focused routing/contract suite: 22 tests passed;
- `python -m compileall -q .`: passed;
- `jq empty experiments/configs/patient_indexed_specimen_routing_*.json`:
  passed;
- `git diff --check`: passed;
- PowerShell AST parse: passed with 3,169 tokens and zero parse errors;
- benchmark `routing_smoke` dry run: 3 algorithms, 10 scenarios, 1 seed,
  20 learned training jobs, and 30 evaluation jobs.

The mechanics runner itself was not executed locally because it intentionally
writes the first artifact in the new result namespace. No teacher generation,
smoke training, pilot training, formal evaluation, packaging, or manuscript
editing was performed.
