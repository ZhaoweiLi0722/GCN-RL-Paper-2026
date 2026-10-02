"""Conditional future branches over an admitted, unchanged cohort environment.

No environment/model construction or scientific admission lives here. Callers
own the exclusive campaign, source locks, durable budgets and recording hooks.
"""

import copy
from dataclasses import asdict
import math

import numpy as np

from src.env.cohort_followup import ClosedCohortClockMixin, same_payload
from src.rl.candidate_imitation import public_example
from src.rl.dynamic_candidate_session import dynamic_context_from_public, _capture_failure
from src.rl.patient_replay_collector import evidence_digest
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays
from src.rl.routing_candidate_contract import RequestCandidates, RoutingRequestSchema, choose_candidate
from src.rl.time_baseline_verification import _decimal_seed


def _open(env):
    if (not isinstance(env, ClosedCohortClockMixin) or env._cohort_closed
            or env._cohort_failure is not None or env._cohort_steps != 0
            or type(env.t) is not int or not 0 <= env.t < env.cohort_spec.enrollment_steps):
        raise ValueError("live open cohort prefix required")


def capture_context(env, producer, reference, options, *, context_id, block,
                    cohort, allowed_times, source_id):
    """Capture a full-state training label source and a separate public example."""
    _open(env)
    if (env.t not in allowed_times or type(block) is not int or type(cohort) is not int
            or cohort < 0 or not all(type(x) is str and x for x in (context_id, source_id))):
        raise ValueError("prespecified context identity/time required")
    producer.check_environment(env)
    before = copy.deepcopy(env.state_dict())
    token = evidence_digest(before)
    observation, bank = dynamic_context_from_public(producer, reference, env.observation(), options, token)
    example = public_example(observation, bank, producer.contract, split="training", identity=context_id)
    if not same_payload(env.state_dict(), before):
        raise ValueError("public context inference mutated the environment")
    result = dict(format="paired-cohort-context-v1", context_id=context_id, block=block,
                  cohort=cohort, after_prefix_steps=env.t, source_id=source_id,
                  reference_sha256=reference.checkpoint_sha256, options=copy.deepcopy(options),
                  cohort_spec=asdict(env.cohort_spec), environment=before, environment_sha256=token,
                  public_example=example, anchor_config=copy.deepcopy(producer.anchor_config))
    result["context_sha256"] = evidence_digest(result)
    return encode_arrays(result)


def validate_context(saved):
    context = decode_arrays(copy.deepcopy(saved))
    body = dict(context)
    sha = body.pop("context_sha256")
    if (context["format"] != "paired-cohort-context-v1" or evidence_digest(body) != sha
            or evidence_digest(context["environment"]) != context["environment_sha256"]
            or context["public_example"]["split"] != "training"
            or context["public_example"]["identity"] != context["context_id"]):
        raise ValueError("context/public example binding differs")
    raw = context["public_example"]["candidates"]
    bank = RequestCandidates(RoutingRequestSchema(**raw["schema"]), raw["state_token"], raw["requests"])
    if evidence_digest(asdict(bank)) != evidence_digest(raw) or bank.state_token != context["environment_sha256"]:
        raise ValueError("canonical support or source state differs")
    return context, bank


def make_branch(template_env, producer_factory, reference, context, *, future_seed,
                candidate_index, replication, branch_id, before_clone, record, enabled=False):
    """Clone a prefix template, restore a context, and change ONLY future RNG.

    The template can be a previously admitted open prefix. Cloning never steps
    it or constructs a fresh simulator. Each instance is externally counted.
    """
    if enabled is not True or not callable(before_clone) or not callable(record):
        raise ValueError("explicit enablement, clone admission and raw recorder required")
    _open(template_env)
    saved, bank = validate_context(context)
    if (asdict(template_env.cohort_spec) != saved["cohort_spec"]
            or reference.checkpoint_sha256 != saved["reference_sha256"]
            or type(replication) is not int or replication < 0
            or type(branch_id) is not str or not branch_id):
        raise ValueError("branch reference/clock/identity differs")
    choice = choose_candidate(bank, candidate_index)
    seed = _decimal_seed(str(future_seed) if type(future_seed) is int else future_seed)
    before_clone()
    env = copy.deepcopy(template_env)
    env.load_state_dict(saved["environment"])
    if (env.t != saved["after_prefix_steps"]
            or not same_payload(env.state_dict(), saved["environment"])):
        raise ValueError("context clone/RNG did not restore exactly")
    producer = producer_factory(env)
    producer.check_environment(env)
    if producer.anchor_config != saved["anchor_config"]:
        raise ValueError("common tail anchor configuration changed")
    # Existing states use PCG64. Do not silently replace another generator type.
    if type(env.rng.bit_generator) is not np.random.PCG64:
        raise ValueError("declared PCG64 future generator required")
    env.rng.bit_generator.state = np.random.PCG64(seed).state
    replaced = copy.deepcopy(saved["environment"])
    replaced["rng_state"] = copy.deepcopy(env.rng.bit_generator.state)
    if not same_payload(env.state_dict(), replaced):
        raise ValueError("conditional branch changed non-RNG source payload")
    return PairedCohortBranch(env, producer_factory, reference, saved, choice,
                             seed, replication, branch_id, record, enabled=True)


class PairedCohortBranch:
    """First candidate exactly once, then R4 prefix and the common fixed tail."""

    def __init__(self, env, producer_factory, reference, context, choice, seed,
                 replication, branch_id, record, *, enabled=False):
        if enabled is not True:
            raise ValueError("explicit branch enablement required")
        _open(env)
        self.env, self.reference, self.producer_factory = env, reference, producer_factory
        self.producer, self.record = producer_factory(env), record
        self.context = copy.deepcopy(context)
        self.request = np.asarray(choice.submitted_request, dtype=np.float64).copy()
        self.start, self.prefix_end = env.t, env.cohort_spec.enrollment_steps
        self.end = self.prefix_end + env.cohort_spec.accounting_steps
        self.manifest = dict(format="paired-cohort-branch-v1", branch_id=branch_id,
            context_sha256=context["context_sha256"], source_id=context["source_id"],
            context_id=context["context_id"], block=context["block"], cohort=context["cohort"],
            after_prefix_steps=self.start, endpoint=self.end, candidate_index=choice.class_index,
            first_request=tuple(self.request), replication=replication, future_seed=str(seed),
            reference_sha256=reference.checkpoint_sha256, initial_state_sha256=evidence_digest(env.state_dict()),
            future_rng_only=True, event_aligned_crn_after_divergence_claimed=False)
        self.rows, self.prefix_final, self.failure = [], None, None
        self._initial_env = copy.deepcopy(env)
        self.expected_token = self.manifest["initial_state_sha256"]

    @property
    def closed(self):
        return self.env.t == self.end and self.env._cohort_steps == self.env.cohort_spec.accounting_steps

    def step(self, *, before_step):
        if self.failure is not None or self.closed or not callable(before_step):
            raise ValueError("live branch and nonrefundable before_step required")
        if (self.env.t != self.start + len(self.rows)
                or evidence_digest(self.env.state_dict()) != self.expected_token
                or self.reference.checkpoint_sha256 != self.manifest["reference_sha256"]):
            raise ValueError("branch state/reference/clock drift")
        before = copy.deepcopy(self.env)
        prefix_final = copy.deepcopy(self.prefix_final)
        debit_started = False
        stage = "request"
        try:
            t = self.env.t
            raw = self.env.observation().copy()
            before_state = evidence_digest(self.env.state_dict())
            if t < self.prefix_end:
                self.producer.check_environment(self.env)
                request = self.request.copy() if not self.rows else np.asarray(self.reference.act(raw), dtype=np.float64).copy()
                action_source = "candidate" if not self.rows else "fixed_r4"
            else:
                request = np.asarray(self.env.common_followup_request(self.context["anchor_config"]), dtype=np.float64).copy()
                action_source = "common_tail"
            if request.shape != (self.env.action_size,) or not np.isfinite(request).all():
                raise ValueError("finite original full-width request required")
            resources = {k: copy.deepcopy(getattr(self.env, k)) for k in
                ("reagents", "bioreactors", "reagent_transfer_pipeline", "capacity_transfer_pipeline", "reagent_purchase_pipeline")
                if hasattr(self.env, k)}
            def debit():
                nonlocal debit_started
                debit_started = True
                before_step()
            stage = "environment_step"
            if t < self.prefix_end:
                debit()
                _, reward, done, info = self.env.step(request.copy())
                event = dict(info=copy.deepcopy(info), raw_reward=float(reward), cost=float(info["cost"]))
                if bool(done) != (self.env.t == self.prefix_end):
                    raise ValueError("prefix boundary changed")
            else:
                _, reward, done, event = self.env.step_followup(request.copy(), before_step=debit)
                info = event["info"]
            if (self.env.t != t + 1 or not math.isfinite(float(info["cost"]))
                    or info["cost"] < 0 or reward != -info["cost"]):
                raise ValueError("raw cost/reward or branch clock differs")
            event.update(stage="prefix" if t < self.prefix_end else "tail", absolute_step=t,
                branch_step=len(self.rows), action_source=action_source, action=request, action_dtype="float64",
                public_observation=raw, before_state_sha256=before_state,
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
            self.env = before
            self.producer = None if self.env._cohort_closed else self.producer_factory(self.env)
            self.prefix_final = prefix_final
            raise

    def receipt(self):
        if self.failure is not None or not self.closed or self.env._cohort_counts() != 0:
            raise ValueError("only a complete settled branch supplies a training label")
        return dict(manifest=copy.deepcopy(self.manifest), steps=len(self.rows),
                    remaining_raw_cost=math.fsum(float(r["info"]["cost"]) for r in self.rows),
                    prefix_raw_cost=math.fsum(float(r["info"]["cost"]) for r in self.rows if r["stage"] == "prefix"),
                    tail_raw_cost=math.fsum(float(r["info"]["cost"]) for r in self.rows if r["stage"] == "tail"),
                    raw_rows_sha256=evidence_digest(self.rows), final_state_sha256=evidence_digest(self.env.state_dict()),
                    terminal_active=0, included_sunk_cost=False)

    def completed_payload(self):
        return dict(manifest=copy.deepcopy(self.manifest), initial_state=self._initial_env.state_dict(),
                    prefix_final=copy.deepcopy(self.prefix_final), final_state=self.env.state_dict(),
                    rows=copy.deepcopy(self.rows), receipt=self.receipt())

    def state_dict(self):
        return encode_arrays(dict(format="paired-cohort-branch-state-v1", manifest=self.manifest,
            rows=copy.deepcopy(self.rows), prefix_final=copy.deepcopy(self.prefix_final),
            followup=self.env.followup_state_dict(), expected_token=self.expected_token,
            failure=copy.deepcopy(self.failure)))

    def load_state_dict(self, saved):
        """Atomic owned restore; the outer ledger and disk rows must also match."""
        if self.failure is not None:
            raise ValueError("failed branch is terminal, not restorable")
        state = decode_arrays(copy.deepcopy(saved))
        if (set(state) != set(decode_arrays(self.state_dict())) or state["format"] != "paired-cohort-branch-state-v1"
                or state["manifest"] != self.manifest or state["failure"] is not None
                or state["followup"]["failure"] is not None or len(state["rows"]) < len(self.rows)):
            raise ValueError("branch binding/failure/progress differs or rewinds")
        rows = state["rows"]
        if not isinstance(rows, list) or len(rows) > self.end - self.start:
            raise ValueError("invalid branch record count")
        if (evidence_digest(rows[:len(self.rows)]) != evidence_digest(self.rows)
                or (len(rows) == len(self.rows) and state_digest(saved) != state_digest(self.state_dict()))):
            raise ValueError("existing branch prefix or same-counter state differs")
        for i, row in enumerate(rows):
            t = self.start + i
            if (row["absolute_step"] != t or row["branch_step"] != i
                    or row["stage"] != ("prefix" if t < self.prefix_end else "tail")
                    or row["action_source"] != ("candidate" if i == 0 else "fixed_r4" if t < self.prefix_end else "common_tail")
                    or not math.isfinite(row["cost"]) or row["cost"] < 0
                    or row["cost"] != row["info"]["cost"] or row["raw_reward"] != -row["cost"]
                    or (i == 0 and not np.array_equal(row["action"], self.request))):
                raise ValueError("raw branch receipts disagree with plan")
        candidate = copy.copy(self)
        candidate.env = copy.deepcopy(self._initial_env)
        if state["prefix_final"] is None:
            if state["followup"]["closed"] or self.start + len(rows) >= self.prefix_end:
                raise ValueError("missing completed prefix boundary")
            candidate.env.load_state_dict(state["followup"]["environment"])
        else:
            if self.start + len(rows) < self.prefix_end:
                raise ValueError("premature prefix completion")
            candidate.env.load_state_dict(state["prefix_final"])
            candidate.env.reconstruct_followup_from_prefix(state["followup"])
        if (candidate.env.t != self.start + len(rows)
                or evidence_digest(candidate.env.state_dict()) != state["expected_token"]
                or not same_payload(candidate.env.followup_state_dict(), state["followup"])):
            raise ValueError("complete environment/RNG did not roundtrip")
        candidate.rows, candidate.prefix_final = rows, state["prefix_final"]
        candidate.expected_token = state["expected_token"]
        candidate.producer = None if candidate.env._cohort_closed else candidate.producer_factory(candidate.env)
        if state_digest(candidate.state_dict()) != state_digest(saved):
            raise ValueError("branch envelope did not roundtrip")
        self.__dict__.update(candidate.__dict__)
