"""New-work ledger for the explicitly approved remaining-only continuation."""

import copy
import os
from pathlib import Path

from src.rl.candidate_pilot_resources import digest
from src.rl.capacity_adaptation_campaign import numeric_contract, PHASES
from src.rl.capacity_pilot_resources import CapacityPilotBudget
from src.utils.research_clock import shared_monotonic


def remaining_contract(original, remainder):
    numeric_contract(original)
    r = remainder['limits']
    names = set(numeric_contract(original)['limits']) - {'trajectories', 'historical_model_loads'}
    limits = {name: r[name] for name in names}
    limits.update(trajectories=252, historical_model_loads=0, partial_restorations=1,
                  imported_seal_artifacts=1, saved_receipt_reconstruction_updates=751,
                  saved_reconstruction_hypothesis_transitions=75100,
                  combined_filter_transitions=1689600)
    if any(type(v) is not int or v < 0 for v in limits.values()):
        raise ValueError('invalid remaining counter')
    phases = {}
    for row in remainder['phase_limits']:
        phase = {k: v for k, v in row.items() if k not in ('id', 'new_world_starts', 'resumed_trajectories')}
        phase['trajectories'] = row['new_world_starts']
        phases[row['id']] = phase
    phases[PHASES[0]].update(partial_restorations=1, imported_seal_artifacts=1,
        saved_receipt_reconstruction_updates=751, saved_reconstruction_hypothesis_transitions=75100,
        planner_teacher_decisions=577, planner_evaluation_decisions=0,
        planner_total_model_epochs=577*384)
    phases[PHASES[1]].update(partial_restorations=0, imported_seal_artifacts=0,
        saved_receipt_reconstruction_updates=0, saved_reconstruction_hypothesis_transitions=0,
        planner_total_model_epochs=0)
    phases[PHASES[2]].update(partial_restorations=0, imported_seal_artifacts=0,
        saved_receipt_reconstruction_updates=0, saved_reconstruction_hypothesis_transitions=0,
        planner_teacher_decisions=0, planner_evaluation_decisions=1728,
        planner_total_model_epochs=1728*384)
    for key in ('native_steps', 'actor_optimizer_steps', 'critic_optimizer_steps',
                'control_steps', 'tail_steps', 'neural_forward_module_calls', 'trajectories'):
        if sum(p[key] for p in phases.values()) != limits[key]:
            raise ValueError('remaining phase totals differ: '+key)
    return dict(limits=limits, phase_limits=phases)


class CapacityRecovery2Budget(CapacityPilotBudget):
    format = 'capacity-recovery2-budget-v1'

    def __init__(self, root, proposal, remainder, *, clock=shared_monotonic, started=None):
        self.root, self.proposal, self.clock = Path(root), copy.deepcopy(proposal), clock
        self.proposal['proposed_budget']['time_seconds'] = remainder['time_seconds']
        for key, value in remainder['resources'].items():
            if key in self.proposal['proposed_budget']:
                self.proposal['proposed_budget'][key] = value
        self.contract = remaining_contract(proposal, remainder)
        self.limits = self.contract['limits']
        self.counts = dict.fromkeys(self.limits, 0)
        self.phase_counts = {k: dict.fromkeys(self.limits, 0) for k in self.contract['phase_limits']}
        self.started = self.last_clock = clock() if started is None else started
        self.times = dict.fromkeys(remainder['time_seconds'], 0.)
        self.owner, self.phase, self.failed = 'admission_lock_binding', None, False
        self.chunks = {}
        self.executed_chunks = dict(planner_total_model_epochs=0, estimator_hypothesis_transitions=0,
                                    saved_reconstruction_hypothesis_transitions=0)
        self.previous, self.sequence = '0'*64, 0
        self.root.mkdir(parents=True, exist_ok=True)
        self.handle = (self.root/'budget.jsonl').open('x', encoding='utf8')
        self._append(dict(event='claim', pid=os.getpid(), started=self.started,
            proposal_sha256=digest(proposal), remainder_sha256=digest(remainder),
            limits=self.limits, old_debits_refunded=False))

    def debit(self, charges):
        charges = dict(charges)
        transitions = charges.get('estimator_hypothesis_transitions', 0) + charges.get('saved_reconstruction_hypothesis_transitions', 0)
        if transitions:
            if 'combined_filter_transitions' in charges:
                raise ValueError('combined transitions are derived, not directly debit-able')
            charges['combined_filter_transitions'] = transitions
        super().debit(charges)
