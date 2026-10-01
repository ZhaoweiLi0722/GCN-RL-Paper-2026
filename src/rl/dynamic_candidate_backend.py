"""Unregistered real-factory adapter, disabled without an execution admission hook.

All real imports/construction are behind admission. The current engineering
entrypoint never supplies that hook. Tests use invented factories only.
"""

import copy
import json
from pathlib import Path

from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_compatibility import inspect_patient_layout, require_supported_layouts
from src.rl.candidate_pilot_driver import file_record
from src.rl.dynamic_candidate_session import DynamicCandidateSession


class DynamicPatientBackend:
    def __init__(self, workspace, config, streams, *, admit_real_calls=None):
        self.workspace = Path(workspace).resolve()
        self.config, self.streams = copy.deepcopy(config), copy.deepcopy(streams)
        self.admit = admit_real_calls
        self.contexts, self.build_attempts, self.episode_builds = {}, 0, 0
        self.failure = None

    def _admit(self, operation):
        if self.failure is not None:
            raise ValueError("failed backend is terminal")
        if not callable(self.admit):
            raise PermissionError("scientific construction is disabled; no admitted execution packet")
        self.admit(operation)

    def prepare(self, block, seed):
        self._admit("reference_and_layout_build")
        if (block not in self.config["blocks"] or block in self.contexts
                or seed != self.streams["environment"][str(block)]["prototype_preflight"][0]
                or self.build_attempts >= self.config["totals"]["backend_layout_builds"]):
            raise ValueError("one declared backend preparation per block required")
        self.build_attempts += 1
        before = global_rng_state()
        try:
            reference_spec = self.config["reference"]
            paths = {key: self.workspace / reference_spec["directory"] /
                     reference_spec[key + "_template"].format(block=block) for key in ("config", "policy")}
            for key, path in paths.items():
                self._admit("input_hash_verification")
                if file_record(self.workspace, path)["sha256"] != reference_spec["locks"][str(block)][key]:
                    raise ValueError("locked reference input changed")
            runtime = json.loads(paths["config"].read_text())
            graph = self.config["candidate_message_graph"]
            if graph != "specimen_routes" or self.config["model_proposal"]["graph"] != graph:
                raise ValueError("explicit prospective message-graph binding required")
            report = inspect_patient_layout(runtime, self.config["objective"], message_graph=graph)
            require_supported_layouts(report)
            from src.rl.experiment import build_env
            from src.rl.frozen_value_probe import assert_scenario
            from src.rl.patient_replay_collector import PatientObservationProducer
            from src.rl.strict_frozen_policy import StrictFrozenPolicy
            self._admit("reference_checkpoint_load")
            reference = StrictFrozenPolicy(paths["policy"], paths["config"],
                checkpoint_sha256=reference_spec["locks"][str(block)]["policy"],
                config_sha256=reference_spec["locks"][str(block)]["config"], device="cpu")
            self._admit("layout_environment_build")
            env = build_env(runtime, seed)
            effective = assert_scenario(env, runtime, self.config["objective"]["scenario"])
            producer = PatientObservationProducer(env, enabled=True,
                gamma=self.config["objective"]["gamma"], reward_scale=self.config["objective"]["reward_scale"],
                message_graph=graph)
            self.contexts[block] = (runtime, producer, reference)
            return {"effective_scenario": effective, "static_layout": report,
                    "producer_contract": producer.contract.inputs.definition_id,
                    "reference_policy_sha256": reference.checkpoint_sha256}
        except BaseException as error:
            self.failure = {"operation": "prepare", "error": repr(error)}
            raise
        finally:
            restore_global_rng(before)

    def producer(self, block):
        if block not in self.contexts or self.failure is not None:
            raise ValueError("live prepared block required")
        return self.contexts[block][1]

    def session(self, block, kernel, seed, *, trajectory, split, selection):
        self._admit("episode_build")
        if block not in self.contexts or self.episode_builds >= self.config["totals"]["fresh_episode_builds"]:
            raise ValueError("declared prepared block and fresh episode build budget required")
        starts = self.streams["environment"][str(block)]
        allowed = (starts["prototype_preflight"] + starts["fork_preflight"] if split == "preflight"
                   else starts.get(split, []))
        if seed not in allowed:
            raise ValueError("undeclared split seed")
        self.episode_builds += 1
        before = global_rng_state()
        try:
            from src.rl.experiment import build_env
            runtime, producer, reference = self.contexts[block]
            env = build_env(runtime, seed)
            return DynamicCandidateSession(env, producer, reference, kernel, enabled=True,
                options=self.config["candidate_support"]["options"], trajectory_id=trajectory,
                split=split, selection=selection, source_id=self.streams["namespace"])
        except BaseException as error:
            self.failure = {"operation": "session", "error": repr(error)}
            raise
        finally:
            restore_global_rng(before)

    def state_dict(self):
        return {"build_attempts": self.build_attempts, "episode_builds": self.episode_builds,
                "prepared_blocks": sorted(self.contexts), "failure": copy.deepcopy(self.failure)}

    def assert_restore_compatible(self, saved):
        # Build attempts are external spend, just like the irreversible step ledger.
        live = self.state_dict()
        if (saved["failure"] is not None or live["failure"] is not None
                or saved["prepared_blocks"] != live["prepared_blocks"]
                or not 0 <= saved["build_attempts"] <= live["build_attempts"]
                or not 0 <= saved["episode_builds"] <= live["episode_builds"]):
            raise ValueError("backend restore cannot refund construction or clear failure")
