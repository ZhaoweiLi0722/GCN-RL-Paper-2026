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
from src.rl.capacity_pilot_runner import CapacityPilotRunner, jsonable, trajectory_id
from src.rl.capacity_planner_tail_design import TRAIN_ROLES, EVAL_ROLES, capture_epochs, worlds
from src.rl.capacity_planner_tail_learner import CapacityPlannerTailLearner
from src.rl.capacity_planner_tail_resources import SCOPE
from src.rl.capacity_value_comparison_learner import CapacityValueComparisonLearner
from src.rl.capacity_value_features import observed_features
from src.rl.capacity_value_runner import CapacityValueRunner
from src.utils.research_clock import shared_monotonic


class CapacityPlannerTailRunner(CapacityValueRunner):
    def __init__(self,root,proposal,budget,*,admission,workspace,
                 value_factory=CapacityPlannerTailLearner,
                 frozen_factory=CapacityValueComparisonLearner,
                 controller_factory=CapacityPlannerTailMPC,
                 feature_adapter=observed_features,**backends):
        if admission != dict(verified=True,scope=SCOPE):
            raise PermissionError("new complete planner-tail admission required")
        self.workspace=Path(workspace)
        self.study=proposal["planner_tail_study"]
        self.training_learners,self.ancestors,self.final={},{},{}
        self.initial={}
        self.episode_data=None
        self.reference_tails=[]
        self.reference_files=[]
        self.role=None
        self.started=False
        self.pending_tail_length=None

        def controller(*args,**kwargs):
            kwargs["value"]=self.learner if self.role not in ("plain_h8","plain_h16") else None
            kwargs.update(planning_horizon=16 if self.role=="plain_h16" else 8,
                capture_epochs=capture_epochs(self.active["index"]) if self.active["phase"]=="reference" else (),
                before_tail_clone=self._tail_clone,before_tail_step=self._tail_step)
            out=controller_factory(*args,**kwargs)
            act=out.act
            def timed(*a,**k):
                start=shared_monotonic()
                result=act(*a,**k)
                if self.pending_tail_length is not None:
                    self._finish_tail()
                elapsed=shared_monotonic()-start
                out.last_plan["decision_wall_seconds"]=elapsed
                out.last_plan["includes_training_tail_capture"]=bool(out.last_training_tails)
                self.reference_tails.extend(out.last_training_tails)
                out.last_training_tails=[]
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

    def _tail_clone(self,payload):
        if self.budget.phase!="reference_and_tails":
            raise ValueError("tail outside reference phase")
        if self.pending_tail_length is not None:
            self._finish_tail()
        self.pending_tail_length=64-payload["start_epoch"]
        self.budget.debit(dict(forecast_clones=1))

    def _tail_step(self,payload):
        if self.pending_tail_length is None:
            raise ValueError("tail primitive has no declared clone")
        self.budget.chunk_call("training_tail_model_epochs",self.pending_tail_length,1)

    def _finish_tail(self):
        # Called only after the preceding model.step returned, never before it.
        self.budget.finish_chunk("training_tail_model_epochs")
        self.pending_tail_length=None

    def _optimizer(self,name,batch):
        if self.budget.phase!="value_fitting" or name!="value" or self.role not in TRAIN_ROLES:
            raise PermissionError("optimizer only in declared training arm")
        self.budget.debit(dict(value_optimizer_steps=1,total_optimizer_steps=1,
                               optimizer_example_presentations=batch))

    def state_dict(self):
        state=super().state_dict()
        state.update(format="capacity-planner-tail-runner-v1",role=self.role,
            training_learners={k:v.state_dict() for k,v in self.training_learners.items()},
            ancestors={k:v[1] for k,v in self.ancestors.items()},
            reference_tails=self.reference_tails,reference_files=self.reference_files,
            pending_tail_length=self.pending_tail_length,automatic_resume=False)
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
        if len(self.reference_tails)!=96:
            raise ValueError("reference world lacks exactly96 shared tails")
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
        if role=="observed_td":
            self.learner.admit_observed(data)
        else:
            self.learner.admit_tails(self.reference_tails)
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

    def _diagnostics(self):
        self.budget.enter("analysis_archive","analysis_archive")
        for relative in self.reference_files:
            with gzip.open(self.root/relative,"rb") as f:
                tails=pickle.load(f)
            block=int(relative.split("-b",1)[1].split("-",1)[0])
            reports=[]
            for role in ("existing_frozen",*TRAIN_ROLES):
                value,sha=self._load_value(block,role)
                for offset in (0,48):
                    group=tails[offset:offset+48]
                    features=np.stack([r["features"][0] for r in group])
                    residual=value.residuals(features)
                    prefix=np.array([r["prefix_cost"] for r in group],dtype=np.float64)
                    h=np.array([r["heuristics"][0] for r in group],dtype=np.float64)
                    truth=prefix+np.array([sum(r["costs"]) for r in group],dtype=np.float64)
                    scores=(prefix+h+residual).reshape(16,3)@np.array([.25,.5,.25])
                    targets=truth.reshape(16,3)@np.array([.25,.5,.25])
                    chosen=int(np.argmin(scores))
                    reports.append(dict(role=role,model_sha256=sha,decision_epoch=group[0]["decision_epoch"],
                        predicted_scores=scores,public_model_targets=targets,chosen=chosen,
                        in_model_regret=float(targets[chosen]-targets.min()),
                        diagnostic_only_training_roots=True))
            self._json("training-tails/"+Path(relative).name+"-diagnostics.json",reports)

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
                config=dict(copy.deepcopy(self.study["value"]),architecture="graph",method=role,max_new_updates=768)
                self.training_learners[role]=self.value_factory.fork_weights(raw,config,
                    seed=self.study["streams"]["sampler_seeds"][block],expected_sha256=sha,
                    before_forward=self._forward,before_optimizer=self._optimizer)
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
        self._diagnostics()
        if self.budget.counts!=self.budget.limits:
            raise ValueError("full experiment counters mismatch")
        self.finished=True
        self.status("all_trajectories_complete")
        return dict(trajectories=len(self.completed),budget=self.budget.snapshot(),
                    final_models={k:v[1] for k,v in self.final.items()})
