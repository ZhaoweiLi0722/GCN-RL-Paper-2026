"""Single finite native campaign, with explicit fake-backend integration seams."""

import copy
import json
from pathlib import Path

import torch

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_campaign import global_rng_state, restore_global_rng
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.candidate_pilot_resources import digest
from src.rl.paired_cohort_binding import bind_inputs, frozen_owner
from src.rl.paired_cohort_episode import recycle_prefix_template, run_episode
from src.rl.paired_cohort_resources import budget_plan
from src.rl.paired_cohort_sequence import PairedCohortSequence, dispatch_serial
from src.rl.paired_cohort_training import block_dataset, branch_verifier, collect_block, fit_block
from src.rl.paired_cohort_verification import assemble_paired_cohort_labels, raw_json
from src.rl.prospective_ddpg_kernel import state_digest
from src.rl.prospective_patient_session import encode_arrays, decode_arrays
from src.utils.research_archive import create_archive, inventory


def read_branch_payloads(root, indexes):
    result = []
    for index in indexes:
        files = index["files"]
        for record in files.values():
            if file_record(root, record["path"]) != record:
                raise ValueError("completed branch bytes changed")
        def read(name):
            return json.loads((Path(root) / files[name]["path"]).read_text())
        header, states = read("header.json"), read("states.json")
        rows = [json.loads(line) for line in (Path(root) / files["events.jsonl"]["path"]).read_text().splitlines()]
        result.append(dict(manifest=header["manifest"], initial_state=header["initial_state"],
            prefix_final=states["prefix_final"], final_state=states["final_state"], rows=rows, receipt=read("receipt.json")))
    return result


class PairedCohortCampaign:
    def __init__(self, workspace, root, packet, budget, admission, *, engineering=False,
                 binder=bind_inputs, episode=run_episode, collect=collect_block, fit=fit_block,
                 assemble=assemble_paired_cohort_labels, branch_reader=read_branch_payloads,
                 wrapper=frozen_owner, recycle=recycle_prefix_template, verifier=None,
                 archive=create_archive, final_lock_check=lambda: None):
        if engineering is not True:
            from src.rl.paired_cohort_execution import PairedAdmission
            if type(admission) is not PairedAdmission:
                raise PermissionError("native campaign requires the live packet/claim admission")
        elif not getattr(admission, "engineering_fixture", False):
            raise PermissionError("explicit fake admission required for artificial integration")
        if budget.plan != budget_plan(packet["config"]):
            raise ValueError("complete prescribed campaign budget required")
        self.workspace, self.root = Path(workspace).resolve(), Path(root).resolve()
        self.packet, self.cfg, self.budget, self.admit = packet, packet["config"], budget, admission
        self.engineering = engineering
        self.binder, self.episode, self.collect, self.fit = binder, episode, collect, fit
        self.assemble, self.branch_reader, self.wrapper, self.recycle = assemble, branch_reader, wrapper, recycle
        if verifier is None:
            from src.rl.paired_cohort_comparison import verify_paired_cohort_bundle
            verifier = verify_paired_cohort_bundle
        self.verifier, self.archive, self.final_lock_check = verifier, archive, final_lock_check
        self.sequence = PairedCohortSequence(self.cfg, self.root, enabled=True)
        self.backend = self.streams = self.labels = self.payload_seal = None
        self.initializers, self.models, self.templates, self.contexts = {}, {}, {}, {}
        self.branch_indexes, self.evaluation_indexes, self.model_paths, self.fitted = [], [], {}, {}
        self.serial, self.failure, self.live = 0, None, None

    def checkpoint(self):
        return encode_arrays(dict(format="paired-cohort-campaign-v1", packet_sha256=digest(self.packet),
            budget=self.budget.snapshot(), sequence=self.sequence.state_dict(),
            backend=None if self.backend is None else self.backend.state_dict(),
            initializers={k: v.state_dict() for k, v in self.initializers.items()},
            models={k: v.state_dict() for k, v in self.models.items()},
            fitted={k: v.state_dict() for k, v in self.fitted.items()},
            templates={k: v.state_dict() for k, v in self.templates.items()},
            contexts=[dict(key=list(k), value=v) for k, v in sorted(self.contexts.items())],
            labels=self.labels, branch_indexes=self.branch_indexes,
            evaluation_indexes=self.evaluation_indexes, model_paths=self.model_paths,
            global_rng=global_rng_state(), failure=self.failure, payload_seal=self.payload_seal))

    def restore_boundary(self, saved):
        """Restore numerical owners at the exact owned durable boundary only.

        This does not authorize a terminated process retry, reconstruct extra
        simulators or roll back the external ledger/recorded bytes.
        """
        old, current = decode_arrays(saved), decode_arrays(self.checkpoint())
        if self.failure is not None or self.live is not None or old["failure"] is not None:
            raise ValueError("terminal/active episode cannot restore a campaign boundary")
        mutable = {"models", "fitted", "templates", "global_rng"}
        if (set(old) != set(current) or any(state_digest(encode_arrays(old[k])) != state_digest(encode_arrays(current[k]))
                                           for k in set(old) - mutable)):
            raise ValueError("durable phase/budget/evidence boundary differs; no rewind")
        replacements = {}
        for field in ("models", "fitted", "templates"):
            owners = getattr(self, field)
            if set(owners) != set(old[field]):
                raise ValueError("owned restoration inventory differs")
            replacements[field] = copy.deepcopy(owners)
            for key, owner in replacements[field].items():
                owner.load_state_dict(encode_arrays(old[field][key]) if field == "fitted" else old[field][key])
                if state_digest(encode_arrays(owner.state_dict())) != state_digest(encode_arrays(old[field][key])):
                    raise ValueError("owned numerical restore did not roundtrip")
        before = global_rng_state()
        try:
            restore_global_rng(old["global_rng"])
        except BaseException:
            restore_global_rng(before)
            raise
        self.__dict__.update(replacements)

    def persist(self, stage, sequence, budget):
        self.serial += 1
        name = f"launcher/boundaries/{self.serial:05d}-{stage}"
        save_envelope(self.root / (name + ".pt"), self.checkpoint())
        write_json_once(self.root / (name + ".json"), dict(stage=stage, sequence=sequence,
            budget=budget, evaluation_cohorts=len(self.evaluation_indexes), conditional_branches=len(self.branch_indexes),
            engineering_fixture=self.engineering, automatic_retry=False))

    def _episode(self, block, role, world, split, *, clone=False):
        self.admit("episode_dispatch")
        name = f"{split}/block{block}/{role}/world{world:02d}"
        seeds = self.streams["environment"][str(block)]["context" if split == "training" else split]
        model = self.models[f"block{block}/{role if role in self.cfg['evaluation_controllers'][:4] else 'own_frozen'}"]
        self.live = name
        try:
            result = self.episode(self.root, self.packet["inherited"]["backend_config"], self.cfg, self.streams,
                self.backend, model, self.budget, admit=self.admit, block=block, role=role, world=world,
                seed=seeds[world], split=split, selection="reference" if role == "r4" else
                "anchor" if role == "full_mdl2" else "greedy", name=name, clone=clone)
            self.live = None
            return result
        except BaseException:
            # Episode dispatcher has already saved partial numerical/raw state.
            raise

    def perform(self, job, sequence, budget):
        phase = job.split("/")[0]
        block = int(job.split("/")[1][5:]) if "/block" in job else None
        evidence = []
        if phase == "binding":
            self.backend, self.initializers, self.models, self.streams = self.binder(
                self.workspace, self.root, self.packet, budget, self.admit)
            evidence.append("payload/binding/models.json")
        elif phase == "same_start_preflight":
            self._episode(block, "r4", 0, "preflight", clone=True)
        elif phase == "reference_contexts":
            for world in range(self.cfg["context_cohorts_per_block"]):
                result = self._episode(block, "r4", world, "training")
                for key, context in result["contexts"].items():
                    if (block, *key) in self.contexts:
                        raise ValueError("duplicate reference context")
                    self.contexts[(block,) + key] = context
                if world == self.cfg["context_cohorts_per_block"] - 1:
                    last = result["contexts"][world, self.cfg["context_after_prefix_steps"][-1]]
                    self.templates[block] = self.recycle(result["environment"], last)
        elif phase == "paired_branches":
            reference = self.backend.contexts[block][2]
            self.branch_indexes.extend(self.collect(self.root, self.cfg, self.streams, block,
                {(c, t): value for (b, c, t), value in self.contexts.items() if b == block}, self.templates[block],
                lambda env: self.backend.branch_producer(block, env), reference, budget,
                admit=self.admit, verifier=branch_verifier(self.cfg, self.streams)))
            if block == self.cfg["blocks"][-1]:
                self.labels = self.assemble([decode_arrays(value) for value in self.contexts.values()],
                    self.branch_reader(self.root, self.branch_indexes), config=self.cfg,
                    context_seeds={str(b): self.streams["environment"][str(b)]["context"] for b in self.cfg["blocks"]},
                    future_seeds=self.streams["conditional_future"])
                self.labels = raw_json(self.labels)
                write_json_once(self.root / "payload/labels.json", self.labels)
                evidence.append("payload/labels.json")
        elif phase in ("paired_actor", "bc_actor"):
            if self.labels is None:
                raise ValueError("complete branch matrix required before any fitting")
            arm = "paired_cost" if phase == "paired_actor" else "bc_continue"
            seed = self.streams["neural"][f"block{block}/{arm}"]
            rng = global_rng_state()
            try:
                torch.manual_seed(seed)
                view = self.initializers[block]
                if view.policy.snapshot_sha256() != self.models[f"block{block}/own_frozen"].policy.snapshot_sha256():
                    raise ValueError("same-start immutable initializer changed before fitting")
                owner, record = self.fit(self.root, self.cfg, block, arm, view.policy, view.contract,
                    block_dataset(self.labels, block), budget, admit=self.admit)
            finally:
                restore_global_rng(rng)
            key = f"block{block}/{arm}"
            self.fitted[key] = owner
            self.models[key] = self.wrapper(owner, seed)
            evidence.append(record["path"])
        elif phase == "seal":
            for key in sequence.models:
                path = self.root / f"payload/sealed/{key}.pt"
                save_envelope(path, self.models[key].state_dict())
                self.model_paths[key] = path.relative_to(self.root).as_posix()
            sequence.seal_models(self.model_paths)
            write_json_once(self.root / "payload/model-seals.json", sequence.seals)
            evidence.append("payload/model-seals.json")
        elif phase == "evaluation":
            sequence.check_seals()
            role = job.split("/")[2]
            for world in range(self.cfg["test_worlds_per_block"]):
                self.evaluation_indexes.append(self._episode(block, role, world, "test")["index"])
        elif phase == "raw_verification":
            write_json_once(self.root / "payload/evaluation-index.json", self.evaluation_indexes)
            report = self.verifier(self.root, self.evaluation_indexes, self.cfg,
                self.packet["inherited_scientific_config"], self.packet["streams"])
            write_json_once(self.root / "payload/independent-comparison.json", report)
            self.final_lock_check()
            evidence.append("payload/independent-comparison.json")
        elif phase == "archive":
            write_json_once(self.root / "payload/artifact-inventory.json", inventory(self.root / "payload"))
            receipt = self.archive(self.root / "payload", self.root / "archives/completed-payload.tar.gz")
            self.payload_seal = receipt["files"]
            write_json_once(self.root / "launcher/archive-receipt.json", dict(archive=receipt,
                local_archive_verified=True, dropbox_exported=False, cloud_sync_verified=False, howard_access_verified=False))
            evidence.append("launcher/archive-receipt.json")
        elif phase == "closure":
            if inventory(self.root / "payload") != self.payload_seal:
                raise ValueError("payload changed after archive")
            self.final_lock_check()
            write_json_once(self.root / "launcher/closure.json", dict(complete=True,
                evaluation_cohorts=len(self.evaluation_indexes), conditional_branches=len(self.branch_indexes),
                contexts=len(self.contexts), new_fits=len(self.fitted), operations=self.admit.counts,
                counts=budget.counts, engineering_fixture=self.engineering, automatic_followon=False))
            evidence.append("launcher/closure.json")
        else:
            raise ValueError("undeclared serial phase")
        path = f"launcher/completed-jobs/{job}.json"
        write_json_once(self.root / path, dict(job=job, budget=budget.snapshot(), models=list(self.models),
            evaluations=len(self.evaluation_indexes), branches=len(self.branch_indexes)))
        return evidence + [path]

    def run(self):
        try:
            return dispatch_serial(self.sequence, self.budget, self.perform, self.persist, admit=self.admit)
        except BaseException as error:
            self.failure = dict(error=repr(error), automatic_retry=False)
            save_envelope(self.root / "launcher/campaign-failure.pt", self.checkpoint())
            raise
