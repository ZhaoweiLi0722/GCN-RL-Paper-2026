"""Exclusive branch records and full restore envelopes; no scientific calls."""

import json
import os
from pathlib import Path

from src.rl.candidate_patient_session import save_envelope
from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import normalized, write_json_once
from src.rl.patient_replay_collector import evidence_digest


class PairedBranchRecorder:
    def __init__(self, root, relative, branch):
        self.root = Path(root).resolve()
        self.directory = self.root / relative
        if (self.directory.resolve() != self.directory or not self.directory.is_relative_to(self.root)
                or branch.rows or branch.failure is not None):
            raise ValueError("new in-root branch and zero-step source required")
        self.directory.mkdir(parents=True, exist_ok=False)
        self.count, self.failed, self.closed = 0, False, False
        self.manifest = normalized(branch.manifest)
        write_json_once(self.directory / "header.json", dict(format="paired-branch-raw-v1",
            manifest=self.manifest, initial_state=branch.env.state_dict(), context=branch.context))
        save_envelope(self.directory / "initial.pt", branch.state_dict())
        self.handle = (self.directory / "events.jsonl").open("x", encoding="utf-8")

    def append(self, branch, event):
        if (self.failed or self.closed or normalized(branch.manifest) != self.manifest
                or event["branch_step"] != self.count or len(branch.rows) != self.count):
            raise ValueError("foreign, skipped or duplicate branch row")
        try:
            self.handle.write(json.dumps(normalized(event), sort_keys=True, allow_nan=False) + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
            self.count += 1
        except BaseException:
            self.failed = True
            raise

    def snapshot(self):
        if self.failed:
            raise ValueError("failed recorder cannot resume")
        if not self.handle.closed:
            self.handle.flush()
        return dict(rows=self.count, closed=self.closed,
                    files={p.name: file_record(self.root, p) for p in sorted(self.directory.iterdir()) if p.is_file()})

    def assert_boundary(self, saved):
        if saved != self.snapshot():
            raise ValueError("persisted branch boundary differs; no truncation or budget refund")

    def finish(self, branch, *, verifier):
        if (self.failed or self.closed or not callable(verifier)
                or not branch.closed or branch.failure is not None or len(branch.rows) != self.count):
            raise ValueError("complete branch and independent verifier required")
        self.handle.close()
        try:
            receipt = normalized(branch.receipt())
            rows = [json.loads(line) for line in (self.directory / "events.jsonl").read_text().splitlines()]
            if evidence_digest(rows) != receipt["raw_rows_sha256"]:
                raise ValueError("disk rows differ from collector")
            states = dict(prefix_final=normalized(branch.prefix_final), final_state=normalized(branch.env.state_dict()))
            write_json_once(self.directory / "states.json", states)
            write_json_once(self.directory / "receipt.json", receipt)
            save_envelope(self.directory / "final.pt", branch.state_dict())
            header = json.loads((self.directory / "header.json").read_text())
            result = verifier(header=header, rows=rows, states=states, receipt=receipt)
            write_json_once(self.directory / "verified.json", result)
            self.closed = True
            return dict(files={p.name: file_record(self.root, p) for p in sorted(self.directory.iterdir())},
                        result=result)
        except BaseException:
            self.failed = True
            raise

    def close_partial(self):
        self.failed = True
        self.handle.close()
