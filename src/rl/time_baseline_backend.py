"""Disabled-by-default real backend with explicit new stream/session bindings."""

import json

from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_compatibility import inspect_patient_layout, require_supported_layouts
from src.rl.candidate_pilot_driver import file_record
from src.rl.dynamic_candidate_backend import DynamicPatientBackend
from src.rl.dynamic_candidate_session import DynamicCandidateSession
from src.rl.time_baseline_collection import TimeBaselineSession
from src.rl.time_baseline_ppo import TimeBaselinePPOKernel


class TimeBaselinePatientBackend(DynamicPatientBackend):
    def prepare(self, block, seed):
        self._admit("reference_and_layout_build")
        if (block not in self.config["blocks"] or block in self.contexts
                or seed != int(self.streams["environment"][str(block)]["layout"][0])
                or self.build_attempts >= self.config["totals"]["backend_layout_builds"]):
            raise ValueError("one declared layout per block required")
        self.build_attempts += 1
        before = global_rng_state()
        try:
            spec = self.config["reference"]
            paths = {k: self.workspace / spec["directory"] / spec[k + "_template"].format(block=block)
                     for k in ("config", "policy")}
            for key, path in paths.items():
                self._admit("input_hash_verification")
                if file_record(self.workspace, path)["sha256"] != spec["locks"][str(block)][key]:
                    raise ValueError("locked reference input changed")
            runtime = json.loads(paths["config"].read_text())
            graph = self.config["candidate_message_graph"]
            if graph != "specimen_routes" or self.config["model_proposal"]["graph"] != graph:
                raise ValueError("explicit unchanged specimen graph required")
            report = inspect_patient_layout(runtime, self.config["objective"], message_graph=graph)
            require_supported_layouts(report)
            from src.rl.experiment import build_env
            from src.rl.frozen_value_probe import assert_scenario
            from src.rl.patient_replay_collector import PatientObservationProducer
            from src.rl.strict_frozen_policy import StrictFrozenPolicy
            self._admit("reference_checkpoint_load")
            reference = StrictFrozenPolicy(paths["policy"], paths["config"],
                checkpoint_sha256=spec["locks"][str(block)]["policy"],
                config_sha256=spec["locks"][str(block)]["config"], device="cpu")
            self._admit("layout_environment_build")
            env = build_env(runtime, seed)
            effective = assert_scenario(env, runtime, self.config["objective"]["scenario"])
            producer = PatientObservationProducer(env, enabled=True,
                gamma=self.config["objective"]["gamma"], reward_scale=self.config["objective"]["reward_scale"],
                message_graph=graph)
            self.contexts[block] = runtime, producer, reference
            return dict(effective_scenario=effective, static_layout=report,
                        producer_contract=producer.contract.inputs.definition_id,
                        reference_policy_sha256=reference.checkpoint_sha256)
        except BaseException as error:
            self.failure = {"operation": "prepare", "error": repr(error)}
            raise
        finally:
            restore_global_rng(before)

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self._admit("episode_build")
        if (block not in self.contexts or split not in ("preflight", "training", "test")
                or seed not in [int(s) for s in self.streams["environment"][str(block)][split]]
                or self.episode_builds >= self.config["totals"]["fresh_episode_builds"]):
            raise ValueError("declared block/split seed and remaining build budget required")
        self.episode_builds += 1
        before = global_rng_state()
        try:
            from src.rl.experiment import build_env
            runtime, producer, reference = self.contexts[block]
            env = build_env(runtime, seed)
            cls = TimeBaselineSession if type(kernel) is TimeBaselinePPOKernel else DynamicCandidateSession
            extra = {"environment_seed": seed} if cls is TimeBaselineSession else {}
            return cls(env, producer, reference, kernel, enabled=True,
                options=self.config["candidate_support"]["options"], trajectory_id=trajectory,
                split=split, selection=selection, source_id=self.streams["namespace"], **extra)
        except BaseException as error:
            self.failure = {"operation": "session", "error": repr(error)}
            raise
        finally:
            restore_global_rng(before)
