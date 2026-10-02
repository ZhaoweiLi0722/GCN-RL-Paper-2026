"""Versioned prefix/tail dispatcher; never updates a model or starts a campaign."""

import copy

from src.env.cohort_followup import ClosedCohortClockMixin
from src.rl.cohort_objective import cohort_reward_receipt


class CohortCollection:
    """Delay training admission until the common economic window is complete.

    The caller records prefix events unchanged and finishes its original prefix
    recorder before closing enrollment. Tail events have a separate schema and
    cannot be passed through the old fixed-52-step reader/learner as new actions.
    This dispatcher is not yet wired into a scientifically admitted campaign.
    """

    def __init__(self, prefix, *, objective, split, trajectory_id,
                 followup_action, finish_prefix, enabled=False):
        if (enabled is not True or not isinstance(prefix.env, ClosedCohortClockMixin)
                or prefix.index != 0 or prefix.closed
                or objective not in ("window", "cohort", "none")
                or split not in ("training", "preflight", "test")
                or not callable(followup_action) or not callable(finish_prefix)
                or not isinstance(trajectory_id, str) or not trajectory_id):
            raise ValueError("fresh explicit prefix, fixed objective and recording hooks required")
        if split != "training" and objective != "none":
            raise ValueError("nontraining collection cannot admit training targets")
        self.prefix, self.env = prefix, prefix.env
        self.objective, self.split, self.trajectory_id = objective, split, trajectory_id
        self.followup_action, self.finish_prefix = followup_action, finish_prefix
        self.prefix_costs, self.tail_events = [], []
        self.prefix_snapshot = self.prefix_receipt = self.failure = None

    @property
    def closed(self):
        return self.env._cohort_steps == self.env.cohort_spec.accounting_steps

    def step(self, *, before_step):
        if self.failure is not None or self.closed:
            raise ValueError("failed or completed collection cannot advance")
        try:
            if not self.prefix.closed:
                event = self.prefix.step(before_step=before_step)
                self.prefix_costs.append(float(event["info"]["cost"]))
                if self.prefix.closed:
                    # Persist original done/observation/cost lineage before the
                    # closure overlay changes only future exogenous inputs.
                    self.prefix_snapshot = copy.deepcopy(self.prefix.state_dict())
                    self.prefix_receipt = copy.deepcopy(self.finish_prefix(self.prefix))
                    self.env.close_enrollment()
                return dict(stage="prefix", event=event)
            request = self.followup_action(self.env)
            _, _, _, event = self.env.step_followup(request, before_step=before_step)
            self.tail_events.append(copy.deepcopy(event))
            return dict(stage="tail", event=event)
        except BaseException as error:
            self.failure = dict(error=repr(error), prefix_steps=len(self.prefix_costs),
                                tail_steps=len(self.tail_events), refunded=False)
            raise

    def target_receipt(self):
        if self.failure is not None or not self.closed or self.objective == "none":
            raise ValueError("only fully closed training acquisition may supply targets")
        return cohort_reward_receipt(self.prefix_costs, [r["cost"] for r in self.tail_events],
            objective=self.objective, prefix_steps=self.env.cohort_spec.enrollment_steps,
            accounting_steps=self.env.cohort_spec.accounting_steps,
            active_at_end=self.env._cohort_counts(), trajectory_id=self.trajectory_id, split=self.split)

    def state_dict(self):
        return dict(format="cohort-collection-v1", objective=self.objective, split=self.split,
                    trajectory_id=self.trajectory_id, prefix_costs=list(self.prefix_costs),
                    prefix_snapshot=copy.deepcopy(self.prefix_snapshot),
                    prefix_receipt=copy.deepcopy(self.prefix_receipt),
                    prefix_live=copy.deepcopy(self.prefix.state_dict()) if self.prefix_snapshot is None else None,
                    tail_events=copy.deepcopy(self.tail_events),
                    followup=self.env.followup_state_dict(), failure=copy.deepcopy(self.failure))
