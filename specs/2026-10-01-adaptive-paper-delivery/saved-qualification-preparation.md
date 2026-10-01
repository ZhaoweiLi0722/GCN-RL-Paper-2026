# Saved qualification: ready engineering, unapproved execution

## Delivery and decision

Implementation: `1fd62b2afe29e618cf8747933d59cceabce26cce`.
Frozen packet: `saved-qualification-frozen.json`, content SHA256
`0f46307d6ecca4f3b317fe8a18b88c6291af5b6b866c86c0764be209475e2afb`.
The original proposal and its false authorization flags remain unchanged.
The packet binds343 source files,42 existing input files,and CPU float32 runtime.
Its preparation command completed without deserializing a research checkpoint,
scoring a saved scientific state,creating an environment or calling an optimizer.

This removes the implementation obstacle to checking whether the three saved
initializers meet the original qualification criterion on both saved paths.
It does not establish that they pass. The prior raw-only39-episode readout is
reused unchanged;it shows initialization fidelity,not an RL improvement.

## Execution boundary

The only proposed numerical work is four weights-only CPU loads:three final
initializers and the completed-collection checkpoint. Six saved paths have104
states each,for624 frozen forwards,within900 seconds including loading,
binding,scoring and terminal readback. No new patient step,optimizer update,
rollout,test episode,reward change or scenario change. One attempt;first error
is terminal. Passing does not authorize continuation or reopen original S1.

The wrapper reconstructs only the model container and overwrites every tensor
from the exact saved state. It never constructs/restores an optimizer,patient
environment or campaign. Both R4 and initializer-path raw headers must match
the saved initializer model hash. Public example identities,split,actor state,
state tokens,original request bank,raw file index and block/seed lineage must
match. Original qualification decoding and thresholds are reused unchanged.

Inputs are byte-bound before parsing;each load and forward is durably charged
first. The parent owns a single timeout-controlled child. An exclusive result
directory prevents duplicate/retry launches. Per-path receipts,hash-chain debits,
before/after input hashes and terminal readback remain separate from old evidence.
A negative qualification verdict is a valid completed readout,not hidden by cost
gains or treated as permission to change thresholds.

## Validation

92 tests passed in5.942seconds,using artificial arrays,fabricated saved states,
fake loaders and a dummy sleeping watchdog process. Adam/SGD steps were forbidden;
no scientific models or real patient environments were loaded. Full repository
`compileall -q .` completed exit0. All owned test processes ended.

The independent advancement agent wrote23 tests and found one genuine omitted
R4-path model-hash check. It was fixed before freeze. Two execution-test errors
were macOS temporary-directory aliases;fixtures now resolve their root. Neither
was a scientific attempt. The efficiency agent recommended reusing existing
qualification functions and prior raw analysis,without another review gate.
Both agents completed their finite tasks and were closed.

## Handoff

Only the already-issued approval question remains:
approve one saved-model-only qualification run using3 models and1 collection
checkpoint,624 saved state scorings,4 loads,900 seconds,zero new simulations,
optimizer updates or test episodes,no changed thresholds and no continuation?

No approval has been received. `saved-qualification-authorization.json` and
`results/dynamic_candidate_saved_qualification_20261001` do not exist.
`gcn-rl` is PAUSED and retained visibly. Do not recreate it or rerun preparation
while awaiting this one decision. After explicit approval,record the actual
user response with packet/implementation/proposal hashes,exact limits,and
`automatic_retry=false`, `training_after_qualification=false`;append change
control and commit before using the separate entrypoint:

```bash
PYTHONPYCACHEPREFIX=/private/tmp/gcn_rl_pycache "/Users/lizhaowei/GCN-RL Paper 2026/.venv/bin/python" -m experiments.scripts.run_dynamic_candidate_saved_qualification
```

That command currently refuses execution because the separate committed approval
is absent. If qualification later passes,the remaining paper-level blocker is
still the same-start PPO-versus-frozen and PPO-versus-BC comparison. Its recovery
needs a separately bounded proposal using these saved initializers,not refitting
or an automatic claim on the consumed S1 budget. No present claim of RL gain,
clinical noninferiority,online adaptation or publication readiness is supported.
