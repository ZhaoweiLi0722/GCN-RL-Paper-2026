"""One new fixed-resource learner; reuse native/public IO without rescheduling old trials."""

from src.rl.capacity_native import keyed_seed
from src.rl.capacity_pilot_runner import trajectory_id
from src.rl.capacity_public_features import resource_adjacency
from src.rl.patient_constrained_runner import PatientConstrainedRunner
from src.rl.fixed_budget_capacity_resources import PHASES


def worlds(proposal, phase, block):
    if phase not in ("training", "evaluation") or block not in range(3):
        raise ValueError("unknown stream phase/block")
    streams = proposal["fixed_budget_study"]["streams"]
    base = streams[phase+"_base"]
    for index in range(12):
        c, j = index % 3, index // 3
        yield dict(phase=phase, block=block, condition=c, replicate=j,
                   seed=base+1000*block+100*c+j)


class FixedBudgetCapacityRunner(PatientConstrainedRunner):
    def __init__(self, root, proposal, budget, *, admission, learner_factory=None, **backends):
        if admission.get("verified") is not True or admission.get("scope") != "fixed-budget-capacity-v1":
            raise PermissionError("new exact fixed-budget admission required")
        if learner_factory is None:
            from src.rl.fixed_budget_capacity_learner import FixedBudgetCapacityLearner
            learner_factory = FixedBudgetCapacityLearner
        super().__init__(root, proposal, budget,
            admission=dict(verified=True, scope="patient-constrained-capacity-v1"),
            learner_factory=learner_factory, **backends)

    def run(self):
        if self.started or self.finished:
            raise RuntimeError("single serial attempt; no repeat or resume")
        self.started = True
        for block, seed in enumerate(self.p["design"]["training_seeds"]):
            self.budget.enter(PHASES[0], PHASES[0])
            self.budget.job(f"reference-b{block}", "reference_block")
            self.learner = None
            reference = []
            for world in worlds(self.p, "training", block):
                key = (block, world["condition"], world["replicate"])
                tape = self._tape(world)
                self.training_tapes[key] = tape
                rows, summary = self._episode(world, "fixed_allocation_reference", tape)
                reference.extend(rows)
                self.reference_losses[key] = summary["lost"]
            self.budget.debit({"fresh_initializers": 1})
            self.learner = self.learner_factory.from_proposal(self.p, node_input_dim=len(reference[0].state[0]),
                model_seed=seed, replay_seed=keyed_seed(self.p, world, "replay"), adjacency=resource_adjacency(self.p),
                before_forward=self._forward, before_optimizer=self._optimizer)
            for row in reference:
                self.learner.add_reference(row)
            self.active = dict(world, role="warmup")
            for update in range(256):
                self._fit("warmup", update % 3)
                if (update+1) % 64 == 0:
                    self.status("warmup_boundary", update=update+1)
            initial = self._seal(block, initial=True)
            self.budget.enter(PHASES[1], PHASES[1])
            for arm in self.p["fixed_budget_study"]["design"]["training_arms"]:
                self.budget.job(f"training-b{block}-{arm}", "learned_arm_block")
                self.budget.debit({"training_fork_restorations": 1})
                self.learner = self.learner_factory.from_snapshot(initial,
                    before_forward=self._forward, before_optimizer=self._optimizer)
                self.learner.set_arm(arm)
                for world in worlds(self.p, "training", block):
                    key = (block, world["condition"], world["replicate"])
                    rows, summary = self._episode(world, arm, self.training_tapes[key], seal_reference=initial.sha256)
                    for row in rows:
                        self.learner.add_training(row)
                    if arm == "constrained":
                        self.budget.debit({"scalar_multiplier_updates": 1})
                        receipt = self.learner.update_multiplier(world["condition"], summary["lost"], self.reference_losses[key])
                        self._update_receipt("multiplier", receipt, 0.)
                    for _ in range(48):
                        self._fit("training", world["condition"])
                    self._save_state(trajectory_id(world, arm)+"-after-fit")
                    self.status("training_world_fit_completed", arm=arm)
                self._seal((block, arm))
        if len(self.seals) != 3:
            raise RuntimeError("all three final models required before test stream")
        self.status("all_models_sealed")
        self.budget.enter(PHASES[2], PHASES[2])
        for block in range(3):
            for world in worlds(self.p, "evaluation", block):
                tape = self._tape(world)
                for role in self.p["fixed_budget_study"]["design"]["evaluation_arms"]:
                    self.budget.job(f"eval-b{block}-c{world['condition']}-{role}", "evaluation_controller_block_condition")
                    self.learner, seal = None, None
                    if role.endswith("_greedy_frozen"):
                        arm = role.removesuffix("_greedy_frozen")
                        seal = self.seals[(block, arm)]
                        self.budget.debit({"learned_evaluation_arm_restorations": 1})
                        self.learner = self.learner_factory.from_snapshot(seal,
                            before_forward=self._forward, before_optimizer=self._optimizer)
                    self._episode(world, role, tape, seal_reference=None if seal is None else seal.sha256)
        if self.budget.counts != self.budget.limits:
            raise RuntimeError("final numerical counters differ: "+repr({k:(self.budget.counts[k],v)
                for k,v in self.budget.limits.items() if self.budget.counts[k]!=v}))
        self.finished = True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed), final_models={f"{k[0]}:{k[1]}":v.sha256 for k,v in self.seals.items()},
                    initial_models={str(k):v.sha256 for k,v in self.initial_seals.items()}, budget=self.budget.snapshot())
