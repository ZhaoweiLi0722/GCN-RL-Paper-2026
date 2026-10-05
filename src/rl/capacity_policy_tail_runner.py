"""One serial comparison using existing native raw-record and checkpoint IO."""

import copy
import gzip
import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np

from src.baselines.capacity_planner_tail_mpc import CapacityPlannerTailMPC
from src.baselines.capacity_policy_tail_mpc import collect_policy_tail_pair
from src.rl.capacity_policy_tail_design import candidate_index
from src.rl.capacity_pilot_runner import CapacityPilotRunner, jsonable, trajectory_id
from src.rl.capacity_policy_tail_design import TRAIN_ROLES, EVAL_ROLES, capture_epochs, worlds, learner_config
from src.rl.capacity_policy_tail_learner import CapacityPolicyTailLearner
from src.rl.capacity_policy_tail_resources import SCOPE
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_features import observed_features
from src.rl.capacity_value_runner import CapacityValueRunner
from src.utils.research_clock import shared_monotonic


class CapacityPolicyTailRunner(CapacityValueRunner):
    def __init__(self,root,proposal,budget,*,admission,workspace,
                 value_factory=CapacityPolicyTailLearner,
                 frozen_factory=CapacityValueComparisonLearner,
                 controller_factory=CapacityPlannerTailMPC,
                 feature_adapter=observed_features,tail_collector=collect_policy_tail_pair,**backends):
        if admission != dict(verified=True,scope=SCOPE):
            raise PermissionError("new complete planner-tail admission required")
        self.workspace=Path(workspace)
        self.study=proposal["policy_tail_study"]
        self.training_learners,self.ancestors,self.final={},{},{}
        self.initial={}
        self.episode_data=None
        self.reference_tails=[]
        self.reference_files=[]
        self.role=None
        self.started=False
        self.continuation=None
        self.branch_failure=None

        def controller(*args,**kwargs):
            kwargs["value"]=self.learner if self.role not in ("plain_h8","plain_h16") else None
            kwargs.update(planning_horizon=16 if self.role=="plain_h16" else 8,
                capture_epochs=())
            out=controller_factory(*args,**kwargs)
            act=out.act
            def timed(*a,**k):
                start=shared_monotonic()
                result=act(*a,**k)
                elapsed=shared_monotonic()-start
                out.last_plan["decision_wall_seconds"]=elapsed
                collecting=(self.active["phase"]=="reference"
                    and a[0].common.epoch in capture_epochs(self.active["index"]))
                out.last_plan["includes_training_tail_capture"]=collecting
                if collecting:
                    pairs=tail_collector(a[0],out,candidate_index(self.active["index"],a[0].common.epoch),
                        value=self.continuation,continuation_sha256=self.ancestors[self.active["block"]][1],
                        on_failure=self._save_branch_failure,
                        before_clone=self._branch_clone,
                        before_prefix=lambda p:self.budget.debit(dict(prefix_model_epochs=p["model_epochs"])),
                        before_tail=lambda p:self.budget.debit(dict(training_tail_model_epochs=p["model_epochs"])),
                        before_query=self._branch_query,before_filter=self._branch_filter,
                        before_plan=self._branch_plan,
                        after_plan=lambda p:self.budget.finish_chunk("branch_planner_model_epochs"),
                        after_filter=lambda p:self.budget.finish_chunk("branch_filter_transitions"))
                    self.reference_tails.extend(pairs["adaptive"])
                    self.reference_tails.extend(pairs["frozen_mpc"])
                out.last_plan["decision_wall_seconds"]=shared_monotonic()-start
                return result
            out.act=timed
            return out

        CapacityPilotRunner.__init__(self,root,proposal,budget,
            admission=dict(verified=True,scope="dynamic-capacity-pilot-v1"),
            controller_factory=controller,**backends)
        self.value_factory,self.frozen_factory=value_factory,frozen_factory
        self.feature_adapter=feature_adapter
        (self.root/"training-tails").mkdir(exist_ok=False)

    def _query(self,payload):
        width=768 if self.role=="plain_h16" else 384
        self.budget.chunk_call("planner_total_model_epochs",width,payload["model_epochs"])

    def _branch_clone(self,payload):
        self.budget.debit({"forecast_root_constructions" if payload["kind"]=="prefix" else "forecast_tail_clones":1})

    def _save_branch_failure(self,state):
        self.branch_failure=state

    def _branch_plan(self,payload):
        self.budget.debit(dict(branch_planner_decisions=1,branch_candidate_rollouts=48))

    def _branch_query(self,payload):
        self.budget.chunk_call("branch_planner_model_epochs",384,payload["model_epochs"])

    def _branch_filter(self,payload):
        self.budget.chunk_call("branch_filter_transitions",100,payload["hypothesis_transitions"])

    def _optimizer(self,name,batch):
        if self.budget.phase!="value_fitting" or name!="value" or self.role not in TRAIN_ROLES:
            raise PermissionError("optimizer only in declared training arm")
        self.budget.debit(dict(value_optimizer_steps=1,total_optimizer_steps=1,
                               optimizer_example_presentations=batch))

    def state_dict(self):
        state=super().state_dict()
        state.update(format="capacity-policy-tail-runner-v1",role=self.role,
            training_learners={k:v.state_dict() for k,v in self.training_learners.items()},
            ancestors={k:v[1] for k,v in self.ancestors.items()},
            reference_tails=self.reference_tails,reference_files=self.reference_files,
            continuation=None if self.continuation is None else self.continuation.state_dict(),
            branch_failure=self.branch_failure,automatic_resume=False)
        return state

    def _episode(self,world,role,tape,*,seal=None):
        if role not in EVAL_ROLES:
            raise ValueError("unknown six-arm role")
        self.role=role
        self.reference_tails=[]
        # Parent job uses 'world'; translate only that owner kind, not caps.
        original=self.budget.job
        kind="reference_world_with_tails" if world["phase"]=="reference" else "evaluation_h16_world" if role=="plain_h16" else "evaluation_h8_world"
        self.budget.job=lambda identifier,tag:original(identifier,kind if tag=="world" else tag)
        try:
            return super()._episode(world,role,tape,seal=seal)
        finally:
            self.budget.job=original

    def _write_tails(self,world):
        if len(self.reference_tails)!=12:
            raise ValueError("reference world lacks exactly12 paired tails")
        relative="training-tails/"+trajectory_id(world,"plain_h8")+".pkl.gz"
        with gzip.open(self.root/relative,"xb") as f:
            pickle.dump(self.reference_tails,f,protocol=5)
        self.budget.check(storage=True)
        self.reference_files.append(relative)
        return relative

    def _fit(self,world,role,data,tails_path):
        self.budget.enter("value_fitting","value_fitting")
        self.role=role
        self.active=dict(world,role=role)
        name=trajectory_id(world,role)
        self.budget.job(name,"fit")
        self.learner=self.training_learners[role]
        policy="adaptive" if role=="adaptive_tail_td" else "frozen_mpc"
        self.learner.admit_tails([r for r in self.reference_tails if r["label_policy"]==policy])
        self._json("updates/"+name+"-targets.json",dict(
            targets=self.learner.pending["targets"],source_hashes=self.learner.pending["source_hashes"],
            source_native=trajectory_id(world,"plain_h8"),source_tails=tails_path,
            ancestor_sha256=self.learner.ancestor_sha256,method=role))
        with (self.root/"updates"/(name+".jsonl")).open("x") as f:
            for _ in range(32):
                receipt=self.learner.update()
                f.write(json.dumps(jsonable(receipt),allow_nan=False)+"\n")
                f.flush()
                os.fsync(f.fileno())
        self._save_state(name+"-after-fit")
        self.status("value_fit_completed",updates=self.learner.updates,method=role)

    def _seal(self,block,role):
        learner=self.training_learners[role]
        if learner.updates!=768 or learner.pending is not None:
            raise ValueError("incomplete final learner")
        key=f"block{block}-{role}"
        self.budget.job(key,"seal")
        raw,sha=learner.snapshot()
        self.budget.debit(dict(final_seals=1))
        with (self.root/"models"/(key+"-final.pt")).open("xb") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        self._json("models/"+key+"-final.json",dict(sha256=sha,updates=768,method=role,
            config=learner.config,ancestor_sha256=learner.ancestor_sha256))
        self.final[key]=(raw,sha)
        self.status("final_sealed",block=block,role=role,sha256=sha)

    def _load_value(self,block,role):
        if role in ("plain_h8","plain_h16"):
            return None,None
        if role=="existing_frozen":
            raw,sha=self.ancestors[block]
            factory=self.frozen_factory
        else:
            raw,sha=self.final[f"block{block}-{role}"]
            factory=self.value_factory
        return factory.from_bytes(raw,before_forward=self._forward,before_optimizer=self._optimizer),sha

    def run(self):
        if self.started:
            raise RuntimeError("single attempt; no relaunch")
        self.started=True
        for block in range(5):
            source=self.study["initial_models"]
            path=self.workspace/source["root"]/source["names"][block]
            raw=path.read_bytes()
            sha=hashlib.sha256(raw).hexdigest()
            if sha!=source["sha256"][block]:
                raise ValueError("ancestor hash mismatch")
            self.ancestors[block]=(raw,sha)
            self._json(f"models/block{block}-ancestor.json",dict(sha256=sha,updates=1536,
                source_path=str(path.relative_to(self.workspace)),historical_provenance="retained_mixed_predictor_block0"))
            with (self.root/"models"/f"block{block}-ancestor.pt").open("xb") as f:
                f.write(raw)
            self.training_learners={}
            for role in TRAIN_ROLES:
                config=learner_config(self.study,role,sha)
                self.training_learners[role]=self.value_factory.fork_weights(raw,config,
                    seed=self.study["streams"]["sampler_seeds"][block],expected_sha256=sha,
                    before_forward=self._forward,before_optimizer=self._optimizer)
            self.continuation=self.frozen_factory.from_bytes(raw,before_forward=self._forward,before_optimizer=self._optimizer)
            for world in worlds(self.study,"reference",block):
                self.budget.enter("reference_and_tails","reference_and_tails")
                self.learner=None
                data=self._episode(world,"plain_h8",self._tape(world))
                tails_path=self._write_tails(world)
                for role in TRAIN_ROLES:
                    self._fit(world,role,data,tails_path)
            for role in TRAIN_ROLES:
                self._seal(block,role)
            self.training_learners={}
        if len(self.final)!=15 or len(self.ancestors)!=5:
            raise RuntimeError("all models must seal before test access")
        self._json("models/all-sealed.json",dict(final_models={k:v[1] for k,v in self.final.items()},
            ancestors={f"block{k}":v[1] for k,v in self.ancestors.items()}))
        self.status("all_models_sealed")
        self.budget.enter("frozen_evaluation","frozen_evaluation")
        for block in range(5):
            for world in worlds(self.study,"evaluation",block):
                tape=self._tape(world)
                for role in EVAL_ROLES:
                    self.learner,sha=self._load_value(block,role)
                    self._episode(world,role,tape,seal=sha)
        self.budget.enter("analysis_archive","analysis_archive")
        if self.budget.counts!=self.budget.limits:
            raise ValueError("full experiment counters mismatch")
        self.finished=True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed),budget=self.budget.snapshot(),
                    final_models={k:v[1] for k,v in self.final.items()})
