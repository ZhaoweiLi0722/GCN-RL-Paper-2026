"""Versioned prefix/tail dispatcher; never updates a model or starts a campaign."""

import copy
import math

from src.env.cohort_followup import ClosedCohortClockMixin
from src.rl.cohort_objective import cohort_reward_receipt
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays, encode_arrays


class CohortCollection:
    """Delay training admission until the common economic window is complete.

    The caller records prefix events unchanged and finishes its original prefix
    recorder before closing enrollment. Tail events have a separate schema and
    cannot be passed through the old fixed-52-step reader/learner as new actions.
    This dispatcher is not yet wired into a scientifically admitted campaign.
    """

    def __init__(self, prefix, *, objective, split, trajectory_id,
                 followup_action, finish_prefix, enabled=False,
                 record_prefix=None, record_tail=None):
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
        if any(f is not None and not callable(f) for f in (record_prefix, record_tail)):
            raise ValueError("explicit raw recording hooks required")
        self.record_prefix = record_prefix or (lambda session, event: None)
        self.record_tail = record_tail or (lambda session, event: None)
        self.prefix_costs, self.tail_events = [], []
        self.prefix_snapshot = self.prefix_receipt = self.failure = None
        self.initial_prefix_sha256 = state_digest(encode_arrays(prefix.state_dict()))

    @property
    def index(self):
        return len(self.prefix_costs) + len(self.tail_events)

    @property
    def learner(self):
        return self.prefix.learner

    @property
    def examples(self):
        return self.prefix.examples

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
                # The final prefix row must reach disk BEFORE its original
                # recorder finalizes at t=52 and BEFORE the closure overlay.
                self.record_prefix(self.prefix, event)
                if self.prefix.closed:
                    # Persist original done/observation/cost lineage before the
                    # closure overlay changes only future exogenous inputs.
                    self.prefix_snapshot = copy.deepcopy(self.prefix.state_dict())
                    self.prefix_receipt = copy.deepcopy(self.finish_prefix(self.prefix))
                    self.env.close_enrollment()
                return dict(stage="prefix", event=event)
            request = self.followup_action(self.env)
            observation = self.env.observation().copy()
            before_hash = evidence_digest(self.env.state_dict())
            resource_fields = ("reagents", "bioreactors", "reagent_transfer_pipeline",
                               "capacity_transfer_pipeline", "reagent_purchase_pipeline")
            resources = {k: copy.deepcopy(getattr(self.env, k)) for k in resource_fields
                         if hasattr(self.env, k)}
            _, _, _, event = self.env.step_followup(request, before_step=before_step)
            event.update(public_observation=observation, before_state_sha256=before_hash,
                         after_state_sha256=evidence_digest(self.env.state_dict()),
                         resources_before=resources,
                         resources_after={k: copy.deepcopy(getattr(self.env, k)) for k in resources})
            self.tail_events.append(copy.deepcopy(event))
            self.record_tail(self, event)
            return dict(stage="tail", event=event)
        except BaseException as error:
            self.failure = dict(error=repr(error), prefix_steps=len(self.prefix_costs),
                                tail_steps=len(self.tail_events), refunded=False)
            raise

    def segment(self, gae_lambda):
        if self.failure is not None or not self.closed or self.objective == "none":
            raise ValueError("only a complete training prefix plus tail may train")
        return self.prefix.segment(gae_lambda)

    def closure_receipt(self):
        self.target_receipt()
        return dict(trajectory_id=self.trajectory_id, split=self.split,
                    environment_seed=self.prefix.manifest["environment_seed"],
                    source_id=self.prefix.source_id,
                    tail_costs=tuple(r["cost"] for r in self.tail_events),
                    terminal_active=self.env._cohort_counts(),
                    prefix_state_sha256=evidence_digest(self.env._cohort_prefix_state),
                    final_state_sha256=evidence_digest(self.env.state_dict()),
                    tail_rows_sha256=evidence_digest(self.tail_events))

    def target_receipt(self):
        if self.failure is not None or not self.closed or self.objective == "none":
            raise ValueError("only fully closed training acquisition may supply targets")
        return cohort_reward_receipt(self.prefix_costs, [r["cost"] for r in self.tail_events],
            objective=self.objective, prefix_steps=self.env.cohort_spec.enrollment_steps,
            accounting_steps=self.env.cohort_spec.accounting_steps,
            active_at_end=self.env._cohort_counts(), trajectory_id=self.trajectory_id, split=self.split)

    def state_dict(self):
        return encode_arrays(dict(format="cohort-collection-v1", objective=self.objective, split=self.split,
                    initial_prefix_sha256=self.initial_prefix_sha256,
                    trajectory_id=self.trajectory_id, prefix_costs=list(self.prefix_costs),
                    prefix_snapshot=copy.deepcopy(self.prefix_snapshot),
                    prefix_receipt=copy.deepcopy(self.prefix_receipt),
                    prefix_live=copy.deepcopy(self.prefix.state_dict()) if self.prefix_snapshot is None else None,
                    tail_events=copy.deepcopy(self.tail_events),
                    followup=self.env.followup_state_dict(), failure=copy.deepcopy(self.failure)))

    def load_state_dict(self, state):
        """Atomic owned reconstruction, without replay, writes or budget changes.

        Outer continuation/recorder must additionally validate the live ledger
        and persisted trace. Failed collections are evidence, never resumable.
        """
        if self.failure is not None or self.env._cohort_failure is not None:
            raise ValueError("failed collection cannot be restored to retry")
        saved = decode_arrays(state)
        fixed = ("format", "objective", "split", "trajectory_id", "initial_prefix_sha256")
        current = decode_arrays(self.state_dict())
        if (not isinstance(saved, dict) or set(saved) != set(current)
                or any(saved[k] != current[k] for k in fixed) or saved["failure"] is not None):
            raise ValueError("collection identity/objective/failure differs")
        costs, rows, tail = saved["prefix_costs"], saved["tail_events"], saved["followup"]
        if (not isinstance(costs, list) or not isinstance(rows, list)
                or len(costs) > self.env.cohort_spec.enrollment_steps
                or any(type(c) not in (int, float) or not math.isfinite(c) or c < 0 for c in costs)
                or len(costs) + len(rows) < self.index):
            raise ValueError("invalid costs or attempted collection rewind")
        candidate = copy.copy(self)
        candidate.prefix = copy.deepcopy(self.prefix)
        candidate.env = candidate.prefix.env
        # A copy at a tail boundary must be returned to its owned prefix for
        # the unchanged session reader. No transition or admission is executed.
        for name in ("_cohort_closed", "_cohort_ids", "_cohort_enrolled",
                     "_cohort_resolution_step", "_cohort_prefix_state"):
            setattr(candidate.env, name, False if name == "_cohort_closed" else None)
        candidate.env._cohort_steps, candidate.env._cohort_costs = 0, []
        closed = saved["prefix_snapshot"] is not None
        if closed:
            if (saved["prefix_live"] is not None or saved["prefix_receipt"] is None
                    or len(costs) != candidate.env.cohort_spec.enrollment_steps):
                raise ValueError("missing complete prefix recording boundary")
            candidate.prefix.load_state_dict(encode_arrays(saved["prefix_snapshot"]))
            if not candidate.prefix.closed or candidate.prefix.index != len(costs):
                raise ValueError("prefix snapshot does not close at declared horizon")
            candidate.env = candidate.prefix.env
            candidate.env.reconstruct_followup_from_prefix(tail)
        else:
            if (saved["prefix_live"] is None or saved["prefix_receipt"] is not None or rows):
                raise ValueError("tail evidence before a completed prefix")
            candidate.prefix.load_state_dict(encode_arrays(saved["prefix_live"]))
            candidate.env = candidate.prefix.env
            if candidate.prefix.closed or candidate.prefix.index != len(costs):
                raise ValueError("prefix cursor and costs differ")
            candidate.env.load_followup_state_dict(tail)
        if ([float(e["info"]["cost"]) for e in candidate.prefix.events] != costs
                or len(rows) != tail["steps"] or [r["cost"] for r in rows] != tail["costs"]):
            raise ValueError("raw event costs and progress differ")
        for i, row in enumerate(rows, 1):
            if (row["index"] != i or row["cost"] != row["info"]["cost"]
                    or row["raw_reward"] != -row["cost"]
                    or row["accounting_done"] != (i == candidate.env.cohort_spec.accounting_steps)):
                raise ValueError("tail event arithmetic or endpoint differs")
        candidate.prefix_costs, candidate.tail_events = costs, rows
        candidate.prefix_snapshot = saved["prefix_snapshot"]
        candidate.prefix_receipt = saved["prefix_receipt"]
        if state_digest(candidate.state_dict()) != state_digest(encode_arrays(saved)):
            raise ValueError("collection/RNG/prefix/tail envelope did not roundtrip")
        self.__dict__.update(candidate.__dict__)
