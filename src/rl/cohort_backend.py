"""Admitted, unregistered cohort episodes over the unchanged public layout."""

from dataclasses import replace

from src.env.cohort_followup import CohortTailSpec, cohort_environment_class
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.cohort_public import CohortObservationProducer, CohortPrefixSession
from src.rl.time_baseline_backend import TimeBaselinePatientBackend


class CohortPatientBackend(TimeBaselinePatientBackend):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.parity_blocks = []

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self._admit("episode_build")
        if (block not in self.contexts or split not in ("preflight", "training", "test")
                or seed not in [int(s) for s in self.streams["environment"][str(block)][split]]
                or self.episode_builds >= self.config["totals"]["fresh_episode_builds"]):
            raise ValueError("declared cohort episode and remaining build budget required")
        self.episode_builds += 1
        before = global_rng_state()
        try:
            from src.env.patient_capacity_planning import patient_env_config_from_dict
            from src.rl.experiment import apply_graph_ablation
            from src.rl.frozen_value_probe import assert_scenario
            runtime, layout, reference = self.contexts[block]
            env_config = dict(runtime.get("env", {}))
            ablation = env_config.pop("graph_ablation", runtime.get("graph_ablation", "full_graph"))
            scenario = env_config.pop("scenario_name", runtime.get("scenario", "default"))
            typed = patient_env_config_from_dict(env_config)
            typed = replace(typed, base=apply_graph_ablation(typed.base, ablation))
            p = self.config["cohort_proposal"]
            spec = CohortTailSpec(p["enrollment_steps"], p["patient_resolution_steps"], p["accounting_steps"])
            env = cohort_environment_class()(typed, seed=seed, cohort_spec=spec, enabled=True)
            env.scenario_name, env.graph_ablation = scenario, ablation
            assert_scenario(env, runtime, self.config["objective"]["scenario"])
            producer = CohortObservationProducer(env, layout, enabled=True)
            return CohortPrefixSession(env, producer, reference, kernel, enabled=True,
                options=self.config["candidate_support"]["options"], trajectory_id=trajectory,
                split=split, selection=selection, source_id=self.streams["namespace"], environment_seed=seed)
        except BaseException as error:
            self.failure = dict(operation="cohort_session", error=repr(error))
            raise
        finally:
            restore_global_rng(before)

    def parity_environment(self, block, seed):
        self._admit("parity_environment_build")
        if (block not in self.contexts or block in self.parity_blocks
                or seed != int(self.streams["environment"][str(block)]["preflight"][0])):
            raise ValueError("one prescribed original-engine parity trace per block")
        self.parity_blocks.append(block)
        before = global_rng_state()
        try:
            from src.rl.experiment import build_env
            return build_env(self.contexts[block][0], seed)
        except BaseException as error:
            self.failure = dict(operation="prefix_parity", error=repr(error))
            raise
        finally:
            restore_global_rng(before)

    def state_dict(self):
        return super().state_dict() | dict(parity_blocks=list(self.parity_blocks))

    def assert_restore_compatible(self, saved):
        if saved.get("parity_blocks") != self.parity_blocks:
            raise ValueError("original-engine parity construction cannot rewind")
        super().assert_restore_compatible({k: v for k, v in saved.items() if k != "parity_blocks"})
