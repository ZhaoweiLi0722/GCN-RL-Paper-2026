"""Trainer-only native continuations from an existing, fully observed root.

Private simulator snapshots never become learner features. The same future tape
is reused: these are dependent branches, not independent patient worlds.
"""

import copy
import math

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.baselines.capacity_policy_tail_mpc import _sha
from src.env.patient_support_capacity import PatientSupportAction
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.capacity_pilot_runner import COST_KEYS, jsonable
from src.rl.capacity_value_features import observed_features
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays
from src.rl.public_support_input import PublicSupportControlInput


def snapshot_sha256(state):
    return state_digest(encode_arrays(state))


def clone_native_environment(source, before_native):
    """Do not copy a runner through its bound debit callback or call reset."""
    if not callable(before_native) or not callable(source._before_native):
        raise ValueError("explicit independent native callback required")
    if getattr(source, "_failed", False):
        raise ValueError("failed environment cannot branch")
    clone = copy.deepcopy(source, {id(source._before_native): before_native})
    clone._before_native = before_native
    if clone is source or snapshot_sha256(clone.state_dict()) != snapshot_sha256(source.state_dict()):
        raise ValueError("clone did not preserve complete native state")
    return clone


def collect_native_tail(view, source, reference, candidate, *, value,
                        continuation_sha256, tape_sha256, before_clone,
                        before_native, before_plan, before_query, after_plan,
                        before_filter, after_filter, record_raw, checkpoint,
                        controller_factory=CapacityPlannerTailMPC,
                        capture=PublicSupportControlInput.capture,
                        feature_adapter=observed_features,
                        clone_factory=clone_native_environment, scientific=True):
    """Run one candidate H8 prefix then a frozen-parent suffix through epoch64.

Every callback is required even for a fake backend. An exception saves the
partially applied boundary and rethrows; no retry, refund or resume is provided.
    """
    root = view.common.epoch
    callbacks = (before_clone, before_native, before_plan, before_query, after_plan,
                 before_filter, after_filter, record_raw, checkpoint)
    if (type(root) is not int or not 0 <= root < 48 or source.t != root
            or type(candidate) is not int or not 0 <= candidate < 16
            or value is None or not _sha(continuation_sha256) or not _sha(tape_sha256)
            or any(not callable(cb) for cb in callbacks)):
        raise ValueError("bounded native root, frozen parent, tape and callbacks required")
    source_state, reference_state = source.state_dict(), reference.state_dict()
    if (source_state.get("world_tape_sha256") != tape_sha256
            or reference_state["filter"]["epoch"] != root
            or reference_state["lifecycle"]["epoch"] != root
            or reference_state["lifecycle"]["recorded_epoch"] >= root):
        raise ValueError("native root must be observed, tape-bound and not operated")
    source_sha, reference_sha = snapshot_sha256(source_state), snapshot_sha256(reference_state)
    rng = global_rng_state()
    env = controller = None
    records = dict(features=[], heuristics=[], costs=[])
    prefix_costs, raw_costs = [], []
    metadata = dict(format="capacity-native-tail-record-v1", root_epoch=root,
        candidate=candidate, start_epoch=root + 8, label_policy="frozen_mpc",
        continuation_sha256=continuation_sha256, source_state_sha256=source_sha,
        tape_sha256=tape_sha256, provenance="native_patient_branch")

    def boundary(status):
        return dict(format="capacity-native-tail-boundary-v1", status=status,
            metadata=metadata, environment=None if env is None else env.state_dict(),
            controller=None if controller is None else controller.state_dict(),
            global_rng=global_rng_state(), records=copy.deepcopy(records),
            prefix_costs=list(prefix_costs), raw_costs=list(raw_costs),
            automatic_resume=False, primitive_may_be_partially_applied=status == "failed")

    try:
        before_clone(dict(epoch=root, candidate=candidate))
        env = clone_factory(source, before_native)
        # Construct an unwrapped controller; the reference act wrapper owns live
        # runner closures and must never be deep-copied into a branch.
        controller = controller_factory(reference.proposal, base_control=reference.base_control,
            value=value, scientific=scientific, planning_horizon=8, capture_epochs=())
        restored = copy.deepcopy(reference_state)
        restored.update(capture_epochs=(), last_training_tails=[])
        controller.load_state_dict(restored)
        public = capture(env)
        if snapshot_sha256(jsonable(public)) != snapshot_sha256(jsonable(view)):
            raise ValueError("cloned public root differs from reference")
        candidates = controller.candidates(public)
        if len(candidates) != 16:
            raise ValueError("fixed sixteen-candidate support required")
        initial, switch = candidates[candidate]
        initial = np.array(initial, dtype=np.float64, copy=True)
        lost_before = {p.patient_id for p in public.operations.patients if p.status == "lost"}
        checkpoint(boundary("root"))
        for epoch in range(root, 65):
            if env.t != epoch or public.common.epoch != epoch:
                raise ValueError("native branch clock mismatch")
            if epoch >= root + 8:
                x, h = feature_adapter(public, controller)
                x, h = np.array(x, dtype=np.float32, copy=True), float(h)
                if x.shape != (4, 31) or not np.isfinite(x).all() or not math.isfinite(h) or epoch == 64 and h != 0.:
                    raise ValueError("invalid public suffix features or terminal base")
                records["features"].append(x)
                records["heuristics"].append(h)
            if epoch == 64:
                break
            offset = epoch - root
            if epoch >= 48:
                hours = np.zeros(4, dtype=np.float64)
            elif offset < 8:
                hours = controller.adaptive(public) if switch and offset >= 2 else initial
            else:
                before_plan(dict(epoch=epoch))
                hours = controller.act(public, role="id_mpc", before_query=before_query,
                                       before_filter=before_filter)
                after_plan(dict(epoch=epoch))
            hours = np.array(hours, dtype=np.float64, copy=True)
            base = reference.base_control(public, reference.proposal, tail=epoch >= 48)
            base = np.array(base, dtype=np.float64, copy=True)
            if hours.shape != (4,) or not np.isfinite(hours).all() or (hours < 0).any() or not np.isfinite(base).all():
                raise ValueError("invalid native request")
            controller.record_operation(public, base)
            _, reward, done, info = env.step(PatientSupportAction(tuple(base), tuple(hours)))
            cost = float(info["cost"])
            components = {key: float(info[key]) for key in COST_KEYS}
            if (not math.isfinite(cost) or cost < 0 or not all(math.isfinite(c) for c in components.values())
                    or not math.isclose(math.fsum(components.values()), cost, rel_tol=1e-12, abs_tol=1e-6)
                    or not math.isclose(-float(reward), cost, rel_tol=1e-12, abs_tol=1e-6)
                    or bool(done) != (epoch == 63)):
                raise ValueError("native branch objective or termination mismatch")
            following = capture(env)
            controller.observe(following, before_filter=before_filter)
            after_filter(dict(epoch=epoch + 1))
            losses = {p.patient_id for p in following.operations.patients if p.status == "lost"}
            if not lost_before.issubset(losses):
                raise ValueError("native branch lost patient reopened")
            row = dict(epoch=epoch, metadata=metadata, prefix=offset < 8,
                requested_hours=hours, requested_base_action=base,
                executed_hours=info["support_public_receipt"]["committed_hours"],
                cost=cost, reward=float(reward), components=components,
                public_input=public, patient_records=following.operations.patients,
                new_lost_patients=len(losses - lost_before), info=info,
                plan=copy.deepcopy(controller.last_plan) if offset >= 8 and epoch < 48 else None)
            record_raw(row)
            raw_costs.append(cost)
            (prefix_costs if offset < 8 else records["costs"]).append(cost)
            public, lost_before = following, losses
            if (epoch + 1) % 8 == 0:
                checkpoint(boundary("step_boundary"))
        settlement = env.settlement()
        if not settlement["settled"] or settlement["lost"] != len(lost_before):
            raise ValueError("native branch did not settle by fixed horizon")
        output = dict(metadata, epochs=list(range(root + 8, 65)),
            features=np.stack(records["features"]),
            heuristics=np.asarray(records["heuristics"], dtype=np.float64),
            costs=np.asarray(records["costs"], dtype=np.float64),
            prefix_costs=np.asarray(prefix_costs, dtype=np.float64),
            total_branch_cost=math.fsum(raw_costs), settlement=settlement,
            policy_is_fixed_parent_not_updated_student=True,
            feature_distribution="native_observed_not_forecast_endpoint")
        checkpoint(boundary("completed"))
        return output
    except BaseException:
        checkpoint(boundary("failed"))
        raise
    finally:
        restore_global_rng(rng)
        if (snapshot_sha256(source.state_dict()) != source_sha
                or snapshot_sha256(reference.state_dict()) != reference_sha):
            raise RuntimeError("native branch mutated reference state")
