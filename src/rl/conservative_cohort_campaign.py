"""Serial two-round orchestration; native launch authority is supplied separately.

Each boundary has reference-only lineage plus the existing full per-episode,
branch and optimizer snapshots. Immutable datasets are not copied into every
launcher checkpoint. A closed/failed instance can never be rerun.
"""

import copy
from pathlib import Path

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.conservative_cohort_binding import bind_inputs
from src.rl.conservative_cohort_plan import budget_plan, cohort_ids, schedule
from src.rl.conservative_cohort_training import collect_round, fit_round
from src.rl.conservative_cohort_verification import assemble_round, branch_reader
from src.rl.paired_cohort_binding import frozen_owner
from src.rl.paired_cohort_episode import run_episode, recycle_prefix_template
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import decode_arrays
from src.utils.research_archive import inventory, create_archive, copy_verified


class ConservativeCohortCampaign:
    def __init__(self, workspace, root, packet, budget, admission, *, engineering=False,
                 binder=bind_inputs, episode=run_episode, collect=collect_round, fit=fit_round,
                 assemble=assemble_round, reader=branch_reader, wrapper=frozen_owner,
                 recycle=recycle_prefix_template, verifier=None, archive=create_archive,
                 final_lock_check=lambda: None):
        if engineering:
            if getattr(admission, "engineering_fixture", False) is not True:
                raise PermissionError("explicit fake admission required")
        else:
            from src.rl.conservative_cohort_execution import ConservativeAdmission
            if type(admission) is not ConservativeAdmission:
                raise PermissionError("native committed single-attempt admission required")
            admission.check()
            if Path(workspace).resolve() != admission.workspace or Path(root).resolve() != admission.root:
                raise PermissionError("campaign differs from its exclusive workspace/run claim")
        if budget.plan != budget_plan(packet["config"]) or not callable(verifier):
            raise ValueError("exact budget and independent raw comparison required")
        self.workspace, self.root = Path(workspace).resolve(), Path(root).resolve()
        self.packet, self.config = copy.deepcopy(packet), copy.deepcopy(packet["config"])
        self.budget, self.admit, self.engineering = budget, admission, engineering
        self.binder, self.episode, self.collect, self.fit = binder, episode, collect, fit
        self.assemble, self.reader, self.wrapper, self.recycle = assemble, reader, wrapper, recycle
        self.verifier, self.archive = verifier, archive
        self.final_lock_check = final_lock_check
        self.jobs, self.done, self.failure, self.started = schedule(self.config), [], None, False
        self.backend, self.initializers, self.models, self.runtime = None, {}, {}, None
        self.contexts, self.templates, self.datasets, self.continuations = {}, {}, {}, {}
        self.branch_indexes, self.evaluation_indexes, self.model_paths, self.seals = [], [], {}, {}

    def boundary(self, job):
        record = dict(format="conservative-campaign-boundary-v1", job=job, complete_jobs=list(self.done),
            packet_sha256=self.packet.get("packet_sha256"), budget=self.budget.snapshot(),
            models={k: state_digest(v.state_dict()) for k, v in self.models.items()},
            model_files=self.model_paths, datasets={str(k): v["dataset_sha256"] for k, v in self.datasets.items()},
            contexts=list(map(list, self.contexts)), branches=len(self.branch_indexes),
            evaluations=len(self.evaluation_indexes), failure=self.failure,
            full_state_location="per_episode_after_context_and_branch_and_actor_snapshots",
            automatic_resume_authorized=False)
        write_json_once(self.root / f"launcher/boundaries/{len(self.done):03d}.json", record)

    def _episode(self, block, role, world, split, *, clone=False):
        self.admit("episode_dispatch")
        model_role = "own_frozen" if role in ("r4", "full_mdl2") else role
        selection = "reference" if role == "r4" else "anchor" if role == "full_mdl2" else "greedy"
        seeds = self.runtime["environment"][str(block)]["context" if split == "training" else split]
        return self.episode(self.root, self.packet["inherited"]["backend_config"], self.config,
            self.runtime, self.backend, self.models[f"block{block}/{model_role}"], self.budget,
            admit=self.admit, block=block, role=role, world=world, seed=seeds[world], split=split,
            selection=selection, name=f"{split}/block{block}/{role}/world{world:03d}", clone=clone)

    def perform(self, job):
        p, b, rnd = job["phase"], job.get("block"), job.get("round")
        if p == "binding":
            self.backend, self.initializers, self.models, self.runtime = self.binder(
                self.workspace, self.root, self.packet, self.budget, self.admit)
        elif p == "preflight":
            self._episode(b, "paired_cost", 0, "preflight", clone=True)
        elif p == "contexts":
            key = (rnd, b)
            owner = self.models[f"block{b}/paired_cost"]
            self.continuations[key] = owner
            save_envelope(self.root / f"payload/round-start/round{rnd}/block{b}.pt", owner.state_dict())
            last = None
            for world in cohort_ids(rnd):
                last = self._episode(b, "paired_cost", world, "training")
                for (c, t), context in last["contexts"].items():
                    self.contexts[b, c, t] = context
            self.templates[key] = self.recycle(last["environment"], next(iter(last["contexts"].values())))
        elif p == "branches":
            key = (rnd, b)
            contexts = {(c, t): value for (block, c, t), value in self.contexts.items()
                        if block == b and c in cohort_ids(rnd)}
            continuation = self.continuations[key]
            owner_hash = state_digest(continuation.state_dict())
            entries = self.collect(self.root, self.config, self.runtime, b, rnd, contexts,
                self.templates[key], lambda env: self.backend.branch_producer(b, env), self.backend.contexts[b][2],
                continuation, self.budget, admit=self.admit,
                verifier=self.reader(self.config, self.runtime, b, rnd, owner_hash))
            self.branch_indexes.extend(entries)
            dataset = self.assemble({k: decode_arrays(v) for k, v in contexts.items()}, entries,
                                   self.config, self.runtime, b, rnd, owner_hash)
            self.datasets[key] = dataset
            write_json_once(self.root / f"payload/datasets/round{rnd}/block{b}.json", dataset)
        elif p in ("paired_actor", "bc_actor"):
            arm = "paired_cost" if p == "paired_actor" else "bc_continue"
            key = f"block{b}/{arm}"
            prototype = self.models[key]
            if rnd == 0 and prototype.policy.snapshot_sha256() != self.models[f"block{b}/own_frozen"].policy.snapshot_sha256():
                raise ValueError("first-round arms are not identical to their frozen control")
            owner, record = self.fit(self.root, self.config, b, rnd, arm, prototype.policy,
                prototype.contract, self.datasets[rnd, b], self.budget, admit=self.admit)
            self.models[key] = self.wrapper(owner, self.runtime["neural"][key])
            self.model_paths[f"round{rnd}/{key}"] = record
        elif p == "seal":
            for key, owner in self.models.items():
                path = self.root / f"payload/sealed/{key}.pt"
                save_envelope(path, owner.state_dict())
                self.seals[key] = file_record(self.root, path)
            if len(self.seals) != 9 or len(self.model_paths) != 12:
                raise ValueError("exact nine final models and twelve round fits required")
            write_json_once(self.root / "payload/model-seals.json", self.seals)
        elif p == "evaluation":
            if len(self.seals) != 9:
                raise PermissionError("all models must seal before first test access")
            for record in self.seals.values():
                if file_record(self.root, self.root / record["path"]) != record:
                    raise ValueError("sealed model changed")
            for world in range(12):
                self.evaluation_indexes.append(self._episode(b, job["role"], world, "test")["index"])
        elif p == "raw_verification":
            if len(self.evaluation_indexes) != 180 or self.budget.counts["optimizer"] != 768:
                raise ValueError("full planned comparison and all fits required")
            write_json_once(self.root / "payload/evaluation-index.json", self.evaluation_indexes)
            write_json_once(self.root / "payload/branch-index.json", self.branch_indexes)
            result = self.verifier(self.root, self.evaluation_indexes, self.config,
                                  self.packet["inherited_scientific_config"], self.packet["streams"])
            write_json_once(self.root / "payload/independent-comparison.json", result)
            self.final_lock_check()
        elif p == "archive":
            self.payload_seal = inventory(self.root / "payload")
            receipt = self.archive(self.root / "payload", self.root / "archives/payload.tar.gz")
            write_json_once(self.root / "launcher/payload-archive.json", receipt)
        elif p == "closure":
            self.final_lock_check()
            if inventory(self.root / "payload") != self.payload_seal:
                raise ValueError("payload changed after archive")
            # Archive a fixed launcher snapshot while the child watchdog and
            # closure budget are still active. Final closure/terminal receipts
            # deliberately remain outside this immutable snapshot.
            snapshot = self.root / "launcher-snapshot"
            for path in sorted((self.root / "launcher").rglob("*")):
                if path.is_file():
                    self.budget.check()
                    copy_verified(path, snapshot / path.relative_to(self.root / "launcher"))
            receipt = self.archive(snapshot, self.root / "archives/launcher.tar.gz")
            self.budget.check()
            write_json_once(self.root / "launcher/launcher-archive.json", dict(archive=receipt,
                snapshot_scope="before closure and final supervisor/terminal receipts",
                dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
            write_json_once(self.root / "launcher/closure.json", dict(complete=True,
                engineering_fixture=self.engineering, models=len(self.seals), round_fits=len(self.model_paths),
                evaluations=len(self.evaluation_indexes), branches=len(self.branch_indexes),
                contexts=len(self.contexts), counts=self.budget.counts, automatic_followon=False))
        else:
            raise ValueError("unknown serial phase")

    def run(self):
        if self.started:
            raise ValueError("single attempt already consumed; never rerun")
        self.started = True
        try:
            for job in self.jobs:
                self.admit(job["id"])
                self.budget.begin(job["id"])
                self.perform(job)
                self.budget.finish()
                self.done.append(job["id"])
                self.boundary(job["id"])
            return dict(completed=True, jobs=self.done)
        except BaseException as error:
            self.failure = dict(error=repr(error), job=self.budget.active,
                                refunded=False, automatic_retry=False)
            write_json_once(self.root / "launcher/campaign-failure.json", self.failure)
            raise
