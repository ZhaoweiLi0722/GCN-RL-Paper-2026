"""Learner-visited contexts and frozen-current-policy conditional continuations.

Additive to the closed paired-cohort implementation. This module provides no
execution authority: every construction, clone and step requires outer admission.
"""

import copy
import math

import numpy as np

from src.env.cohort_followup import same_payload
from src.rl.dynamic_candidate_session import dynamic_context_from_public, _capture_failure
from src.rl.dynamic_candidate_rollout import evaluate_dynamic_policy
from src.rl.paired_cohort_backend import PairedCohortBackend, ReferenceContextSession
from src.rl.paired_cohort_collection import PairedCohortBranch, make_branch
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays
from src.rl.routing_candidate_contract import choose_candidate


class LearnerContextSession(ReferenceContextSession):
    """Explicit training-only greedy collection, never passed to PPO updates."""

    @classmethod
    def from_fresh_reference_session(cls, session):
        if (type(session) is not ReferenceContextSession or session.index != 0
                or session.events or session.examples or session._failure is not None):
            raise ValueError("only an unstepped admitted reference session can be bound")
        obj = cls.__new__(cls)
        obj.__dict__.update(session.__dict__)
        obj.selection = "greedy"
        obj.manifest = copy.deepcopy(session.manifest)
        obj.manifest.update(format="conservative-learner-context-session-v1", selection="greedy",
                            target_admission="current_policy_conditional_costs_only")
        return obj


class ConservativeCohortBackend(PairedCohortBackend):
    def session(self, block, kernel, seed, *, trajectory, split, selection):
        if split != "training":
            return super().session(block, kernel, seed, trajectory=trajectory, split=split, selection=selection)
        if selection != "greedy":
            raise ValueError("current frozen greedy learner must collect training contexts")
        session = super().session(block, kernel, seed, trajectory=trajectory, split=split, selection="reference")
        return LearnerContextSession.from_fresh_reference_session(session)


def greedy_request(env, producer, reference, options, owner):
    """The learned continuation receives only the same public inference inputs."""
    observation, bank = dynamic_context_from_public(producer, reference, env.observation(),
                                                   options, evidence_digest(env.state_dict()))
    evaluation = evaluate_dynamic_policy(owner.policy, observation, bank, owner.contract)
    choice = choose_candidate(bank, int(np.argmax(evaluation.log_probs)))
    return np.asarray(choice.submitted_request, dtype=np.float64).copy()


def make_current_policy_branch(template_env, producer_factory, reference, context, *,
                               continuation, round_index, before_forward, **kwargs):
    """Reuse exactly the original counted clone and future-RNG-only replacement."""
    if (type(round_index) is not int or round_index not in (0, 1)
            or continuation.mode != "frozen" or continuation._failure is not None
            or not callable(before_forward)):
        raise ValueError("fixed healthy round-start owner and admitted inference required")
    original = make_branch(template_env, producer_factory, reference, context, **kwargs)
    branch = CurrentPolicyBranch.__new__(CurrentPolicyBranch)
    branch.__dict__.update(original.__dict__)
    branch.continuation = continuation
    branch.before_forward = before_forward
    branch.continuation_sha256 = state_digest(continuation.state_dict())
    branch.manifest.update(format="conservative-cohort-branch-v1", round=round_index,
                           continuation_sha256=branch.continuation_sha256,
                           continuation="frozen_round_start_greedy_then_common_tail")
    return branch


class CurrentPolicyBranch(PairedCohortBranch):
    """First candidate, then the recorded current policy; fixed MDL-2 tail."""

    def _check_continuation(self):
        if state_digest(self.continuation.state_dict()) != self.continuation_sha256:
            raise ValueError("frozen round-start continuation mutated")

    def step(self, *, before_step):
        if self.failure is not None or self.closed or not callable(before_step):
            raise ValueError("live branch and nonrefundable debit required")
        self._check_continuation()
        if (self.env.t != self.start + len(self.rows)
                or evidence_digest(self.env.state_dict()) != self.expected_token
                or self.reference.checkpoint_sha256 != self.manifest["reference_sha256"]):
            raise ValueError("branch clock/state/reference changed")
        before, prefix_final = copy.deepcopy(self.env), copy.deepcopy(self.prefix_final)
        debit_started, stage = False, "request"
        try:
            t = self.env.t
            raw, token = self.env.observation().copy(), evidence_digest(self.env.state_dict())
            if not self.rows:
                request, action_source = self.request.copy(), "candidate"
            elif t < self.prefix_end:
                self.before_forward()
                request = greedy_request(self.env, self.producer, self.reference,
                                         self.context["options"], self.continuation)
                self._check_continuation()
                action_source = "frozen_round_start_policy"
            else:
                request = np.asarray(self.env.common_followup_request(self.context["anchor_config"]),
                                     dtype=np.float64).copy()
                action_source = "common_tail"
            if request.shape != (self.env.action_size,) or not np.isfinite(request).all():
                raise ValueError("finite original full-width request required")
            resources = {k: copy.deepcopy(getattr(self.env, k)) for k in
                ("reagents", "bioreactors", "reagent_transfer_pipeline", "capacity_transfer_pipeline",
                 "reagent_purchase_pipeline") if hasattr(self.env, k)}
            def debit():
                nonlocal debit_started
                debit_started = True
                before_step()
            stage = "environment_step"
            if t < self.prefix_end:
                debit()
                _, reward, done, info = self.env.step(request.copy())
                if bool(done) != (self.env.t == self.prefix_end):
                    raise ValueError("changed prefix boundary")
                event = dict(info=copy.deepcopy(info), raw_reward=float(reward), cost=float(info["cost"]))
            else:
                _, reward, _, event = self.env.step_followup(request.copy(), before_step=debit)
                info = event["info"]
            if (self.env.t != t + 1 or not math.isfinite(float(info["cost"]))
                    or info["cost"] < 0 or reward != -info["cost"]):
                raise ValueError("raw reward/cost or clock differs")
            event.update(stage="prefix" if t < self.prefix_end else "tail", absolute_step=t,
                branch_step=len(self.rows), action_source=action_source, action=request,
                action_dtype="float64", public_observation=raw, before_state_sha256=token,
                after_state_sha256=evidence_digest(self.env.state_dict()), resources_before=resources,
                resources_after={k: copy.deepcopy(getattr(self.env, k)) for k in resources})
            stage = "raw_publication"
            self.record(self, copy.deepcopy(event))
            if self.env.t == self.prefix_end:
                self.prefix_final = copy.deepcopy(self.env.state_dict())
                self.env.close_enrollment()
            self.rows.append(copy.deepcopy(event))
            self.expected_token = evidence_digest(self.env.state_dict())
            return copy.deepcopy(event)
        except BaseException as error:
            self.failure = dict(error=repr(error), stage=stage, debit_started=debit_started,
                completed_steps=len(self.rows), refunded=False,
                partial_environment=_capture_failure(self.env.followup_state_dict))
            self.env, self.prefix_final = before, prefix_final
            self.producer = None if self.env._cohort_closed else self.producer_factory(self.env)
            raise

    def load_state_dict(self, saved):
        """Restore a forward boundary only with the identical frozen continuation."""
        if self.failure is not None:
            raise ValueError("terminal failure cannot restore")
        self._check_continuation()
        state = decode_arrays(copy.deepcopy(saved))
        if (set(state) != set(decode_arrays(self.state_dict()))
                or state["format"] != "paired-cohort-branch-state-v1"
                or state["manifest"] != self.manifest or state["failure"] is not None
                or state["followup"]["failure"] is not None):
            raise ValueError("bound current-policy branch required")
        rows = state["rows"]
        if (not isinstance(rows, list) or not len(self.rows) <= len(rows) <= self.end - self.start
                or evidence_digest(rows[:len(self.rows)]) != evidence_digest(self.rows)
                or (len(rows) == len(self.rows) and state_digest(saved) != state_digest(self.state_dict()))):
            raise ValueError("branch history rewind or changed boundary")
        for i, row in enumerate(rows):
            t = self.start + i
            expected = "candidate" if i == 0 else "frozen_round_start_policy" if t < self.prefix_end else "common_tail"
            if (row["absolute_step"] != t or row["branch_step"] != i
                    or row["action_source"] != expected
                    or row["stage"] != ("prefix" if t < self.prefix_end else "tail")
                    or not math.isfinite(row["cost"]) or row["cost"] < 0
                    or row["cost"] != row["info"]["cost"] or row["raw_reward"] != -row["cost"]
                    or (i == 0 and not np.array_equal(row["action"], self.request))):
                raise ValueError("branch raw rows conflict with continuation contract")
        candidate = copy.copy(self)
        candidate.env = copy.deepcopy(self._initial_env)
        if state["prefix_final"] is None:
            if state["followup"]["closed"] or self.start + len(rows) >= self.prefix_end:
                raise ValueError("missing prefix boundary")
            candidate.env.load_state_dict(state["followup"]["environment"])
        else:
            if self.start + len(rows) < self.prefix_end:
                raise ValueError("premature prefix boundary")
            candidate.env.load_state_dict(state["prefix_final"])
            candidate.env.reconstruct_followup_from_prefix(state["followup"])
        if (candidate.env.t != self.start + len(rows)
                or evidence_digest(candidate.env.state_dict()) != state["expected_token"]
                or not same_payload(candidate.env.followup_state_dict(), state["followup"])):
            raise ValueError("full environment/RNG restore differs")
        candidate.rows, candidate.prefix_final = rows, state["prefix_final"]
        candidate.expected_token = state["expected_token"]
        candidate.producer = None if candidate.env._cohort_closed else candidate.producer_factory(candidate.env)
        if state_digest(candidate.state_dict()) != state_digest(saved):
            raise ValueError("branch restore did not roundtrip")
        self.__dict__.update(candidate.__dict__)
