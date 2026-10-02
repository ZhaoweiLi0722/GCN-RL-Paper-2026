"""Additive remaining-work campaign; old attempt and scientific settings stay fixed."""

from pathlib import Path

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_recording import write_json_once
from src.rl.paired_cohort_binding import bind_inputs, frozen_owner
from src.rl.paired_cohort_campaign import PairedCohortCampaign, read_branch_payloads
from src.rl.paired_cohort_episode import run_episode
from src.rl.paired_cohort_recovery_data import import_saved_data
from src.rl.paired_cohort_recovery_training import collect_remaining, restore_template
from src.rl.paired_cohort_training import fit_block, branch_verifier
from src.rl.paired_cohort_verification import assemble_paired_cohort_labels, raw_json
from src.rl.prospective_patient_session import decode_arrays
from src.utils.research_archive import create_archive


class PairedCohortRecoveryCampaign(PairedCohortCampaign):
    def __init__(self, workspace, root, packet, budget, admission, *, engineering=False,
                 binder=bind_inputs, importer=import_saved_data, template=restore_template,
                 collect=collect_remaining, episode=run_episode, fit=fit_block,
                 assemble=assemble_paired_cohort_labels, branch_reader=read_branch_payloads,
                 wrapper=frozen_owner, verifier=None, archive=create_archive,
                 final_lock_check=lambda: None):
        from src.rl.paired_cohort_recovery_resources import budget_plan
        from src.rl.paired_cohort_recovery_sequence import PairedCohortRecoverySequence
        if engineering is not True:
            from src.rl.paired_cohort_recovery_execution import RecoveryAdmission
            if type(admission) is not RecoveryAdmission:
                raise PermissionError("live committed recovery admission required")
        elif not getattr(admission, "engineering_fixture", False):
            raise PermissionError("explicit fake-only recovery admission required")
        if budget.plan != budget_plan(packet["config"], packet["recovery_proposal"]):
            raise ValueError("exact approved remaining-work budget required")
        self.workspace, self.root = Path(workspace).resolve(), Path(root).resolve()
        self.packet, self.cfg, self.budget, self.admit = packet, packet["config"], budget, admission
        self.engineering = engineering
        self.binder, self.importer, self.template = binder, importer, template
        self.episode, self.collect, self.fit = episode, collect, fit
        self.assemble, self.branch_reader, self.wrapper = assemble, branch_reader, wrapper
        if verifier is None:
            from src.rl.paired_cohort_comparison import verify_paired_cohort_bundle
            verifier = verify_paired_cohort_bundle
        self.verifier, self.archive, self.final_lock_check = verifier, archive, final_lock_check
        self.sequence = PairedCohortRecoverySequence(self.cfg, packet["recovery_proposal"], self.root, enabled=True)
        self.backend = self.streams = self.labels = self.payload_seal = None
        self.initializers, self.models, self.templates, self.contexts = {}, {}, {}, {}
        self.branch_indexes, self.evaluation_indexes, self.model_paths, self.fitted = [], [], {}, {}
        self.serial, self.failure, self.live = 0, None, None

    def perform(self, job, sequence, budget):
        if job != "binding" and not job.startswith("paired_branches/"):
            return super().perform(job, sequence, budget)
        evidence = []
        if job == "binding":
            self.backend, self.initializers, self.models, self.streams = self.binder(
                self.workspace, self.root, self.packet, budget, self.admit)
            self.contexts, self.branch_indexes = self.importer(self.workspace, self.root,
                self.packet["recovery_manifest"], admit=self.admit, budget=budget)
            for block in self.cfg["blocks"]:
                self.templates[block] = self.template(self.backend, block,
                    self.contexts[block, 0, self.cfg["context_after_prefix_steps"][0]], admit=self.admit)
            evidence += ["payload/binding/models.json", "payload/reused-data.json"]
        else:
            block = int(job.split("/")[1][5:])
            remaining = [r for r in self.packet["recovery_manifest"]["remaining"] if r["block"] == block]
            self.branch_indexes.extend(self.collect(self.root, self.cfg, self.streams, block,
                {(c, t): value for (b, c, t), value in self.contexts.items() if b == block},
                self.templates[block], lambda env: self.backend.branch_producer(block, env),
                self.backend.contexts[block][2], budget, remaining=remaining, admit=self.admit,
                verifier=branch_verifier(self.cfg, self.streams)))
            if block == self.cfg["blocks"][-1]:
                self.labels = raw_json(self.assemble([decode_arrays(v) for v in self.contexts.values()],
                    self.branch_reader(self.root, self.branch_indexes), config=self.cfg,
                    context_seeds={str(b): self.streams["environment"][str(b)]["context"] for b in self.cfg["blocks"]},
                    future_seeds=self.streams["conditional_future"]))
                write_json_once(self.root / "payload/labels.json", self.labels)
                evidence.append("payload/labels.json")
        path = f"launcher/completed-jobs/{job}.json"
        write_json_once(self.root / path, dict(job=job, budget=budget.snapshot(), models=list(self.models),
            evaluations=len(self.evaluation_indexes), branches=len(self.branch_indexes),
            reused_branches=len(self.packet["recovery_manifest"]["completed"])))
        return evidence + [path]

    def run(self):
        from src.rl.paired_cohort_recovery_sequence import dispatch_serial
        try:
            return dispatch_serial(self.sequence, self.budget, self.perform, self.persist, admit=self.admit)
        except BaseException as error:
            self.failure = dict(error=repr(error), automatic_retry=False)
            save_envelope(self.root / "launcher/campaign-failure.pt", self.checkpoint())
            raise
