# Capacity Interface Delivery

2026-10-03 UTC. Engineering only; no new scientific trial or performance result.
Entry HEAD: `ad17fb085d6553879c3e87f77a1be58cb05ea646`.

## Delivered

`src/rl/capacity_adaptation_interface.py` reuses `ServiceEffortConfig` without
altering historical environments. Python float64 and batched differentiable
tensor paths implement the same site-cap/radial shared-budget transform.
Feasible actions remain fixed points, zero purchase is legal, and unused
budget is not forced into spending. Saturation still creates flat directions;
this is not a guarantee of useful critic or clinical gradients.

The public completion-receipt contract explicitly separates requested,
committed, delayed flexible and ordinary staffed hours. It validates integer
job counts, fees on commitments, receipt availability, pipeline timing and
failure-before-mutation. Missing history is masked. JSON restoration binds
the limits, site order, window and horizon and reconstructs the public pipeline.
No latent coefficient, work-progress measurement or future tape is exported.
This is a typed input validator, not authentication of real operational data.

Ordinary staffing costs still need binding in the producer's full accounting;
the receipt's labor/switching fields specifically cover incremental flexible
commitments. No patient preparation layer, six-controller campaign, fitted
initializer, estimator or research launcher is claimed complete.

## Validation

- `python -m unittest tests.test_capacity_adaptation_interface -v`: **13 passed**.
  Artificial vectors/receipts and tensor gradients only; no model or optimizer.
- Whole-repository `python -m compileall -q .`: passed with the external cache
  prefix documented in AGENTS. No patient smoke was permitted or run.
- `pytest` is not installed in this virtualenv. Used the repository's unittest
  path instead; did not install or change dependencies.
- Process inspection after the sandbox denied `ps` used the approved read-only
  system process listing. Filtered scientific-run commands had no matches.
- No old experiment rerun, results overwrite, new archive, model checkpoint
  access, patient environment construction/step or real optimizer call.

The advancement agent Galileo delivered [prior-evidence.md](prior-evidence.md)
and was closed. The read-only efficiency agent Lagrange delivered three
recommendations and was closed: do not reopen the old recipe, keep one narrow
action/observation interface, and stop artificial checks after integration in
favor of the complete comparison. Neither role is a running training process.

## Next Concrete Work

Follow steps 2-3 of [plan.md](plan.md): an additive patient support producer and
same-information controller integration with fake records, then one complete
numerical pilot proposal. Do not repeat an audit of old results or add toy
fitting campaigns. This local preparation is authorized; real scientific
execution needs the later specific numerical scope and committed locks.

The existing `gcn-rl` heartbeat was updated, not duplicated, and tool-confirmed
ACTIVE for this finite engineering chain at its existing 30-minute cadence.
The old conservative experiment remains closed. The schedule will PAUSE when
the complete pilot decision is ready, not launch unspecified training.
