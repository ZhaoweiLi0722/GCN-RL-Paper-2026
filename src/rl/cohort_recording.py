"""Two-part, non-overwriting evidence writer; no model or environment execution."""

from dataclasses import asdict
import json
import os
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_recording import EpisodeRecorder, normalized, write_json_once
from src.rl.cohort_verification import verify_cohort_tail
from src.rl.dynamic_candidate_session import save_envelope
from src.rl.patient_replay_collector import evidence_digest


class CohortRecorder:
    def __init__(self, root, relative, prefix, config, *, prefix_recorder_type=EpisodeRecorder, **metadata):
        self.root = Path(root).resolve()
        self.directory = self.root / relative
        if self.directory.resolve() != self.directory or not self.directory.is_relative_to(self.root):
            raise ValueError("exclusive cohort evidence must stay inside the root")
        self.prefix = prefix_recorder_type(self.root, str(Path(relative) / "prefix"), prefix, config, **metadata)
        self.config, self.metadata = config, metadata
        self.tail_dir = self.directory / "tail"
        self.tail_dir.mkdir(exist_ok=False)
        self.handle = None
        self.count, self.failed, self.closed = 0, False, False
        self.prefix_index = self.tail_header = self.result = None

    def record_prefix(self, session, event):
        if self.failed or self.closed or self.prefix_index is not None:
            raise ValueError("prefix is closed or recorder failed")
        self.prefix.session = session
        self.prefix.append(event)

    def finish_prefix(self, session):
        if self.failed or self.closed or self.prefix_index is not None or session.env._cohort_closed:
            raise ValueError("prefix finalization must precede enrollment closure")
        try:
            self.prefix.session = session
            self.prefix_index = self.prefix.finish()
            self.tail_header = normalized(dict(format="cohort-tail-header-v1", **self.metadata,
                trajectory_id=session.trajectory_id, source_id=session.source_id, split=session.split,
                contract=asdict(session.env.cohort_spec), prefix=self.prefix_index,
                prefix_final_sha256=evidence_digest(session.env.state_dict()),
                anchor_config=session.producer.anchor_config))
            write_json_once(self.tail_dir / "header.json", self.tail_header)
            self.handle = (self.tail_dir / "events.jsonl").open("x", encoding="utf-8")
            return self.prefix_index
        except BaseException:
            self.failed = True
            raise

    def record_tail(self, collection, event):
        if (self.failed or self.closed or self.handle is None
                or collection.trajectory_id != self.tail_header["trajectory_id"]
                or event["index"] != self.count + 1):
            raise ValueError("missing/duplicate/foreign tail event")
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
        if self.handle is not None and not self.handle.closed:
            self.handle.flush()
        prefix = self.prefix.snapshot() if self.prefix_index is None else {
            "closed": True, "files": {k: file_record(self.root, v["path"]) for k, v in self.prefix_index.items()}}
        files = {p.name: file_record(self.root, p) for p in self.tail_dir.iterdir() if p.is_file()}
        return dict(format="cohort-recorder-v1", prefix=prefix, tail_files=files,
                    tail_rows=self.count, closed=self.closed)

    def assert_prefix(self, saved):
        # Same persisted boundary only. Never truncate rows or refund work to
        # make an older checkpoint appear current.
        if saved != self.snapshot():
            raise ValueError("persisted prefix/tail bytes or progress changed")

    def finish(self, collection):
        if (self.failed or self.closed or not collection.closed or collection.failure is not None
                or self.count != collection.env.cohort_spec.accounting_steps
                or collection.prefix_receipt != self.prefix_index):
            raise ValueError("only a fully recorded, live closed cohort can finalize")
        self.handle.close()
        try:
            final = normalized(collection.env.state_dict())
            write_json_once(self.tail_dir / "final_state.json", final)
            prefix_final = json.loads((self.root / self.prefix_index["final_state"]["path"]).read_text())
            rows = [json.loads(line) for line in (self.tail_dir / "events.jsonl").read_text().splitlines()]
            if evidence_digest(rows) != evidence_digest(collection.tail_events):
                raise ValueError("persisted tail differs from collected events")
            receipt = dict(prefix_final_sha256=self.tail_header["prefix_final_sha256"],
                           final_sha256=evidence_digest(final))
            spec = collection.env.cohort_spec
            result = verify_cohort_tail(prefix_final, final, rows, receipt,
                enrollment_steps=spec.enrollment_steps, patient_resolution_steps=spec.patient_resolution_steps,
                accounting_steps=spec.accounting_steps, num_facilities=collection.env.config.num_facilities)
            write_json_once(self.tail_dir / "outcome.json", result)
            if collection.objective != "none":
                write_json_once(self.tail_dir / "training-target.json", collection.target_receipt())
                write_json_once(self.tail_dir / "training-lineage.json", collection.closure_receipt())
            save_envelope(self.tail_dir / "collector.pt", collection.state_dict())
            self.result = dict(prefix=self.prefix_index,
                tail={name: file_record(self.root, self.tail_dir / filename) for name, filename in
                      (("header", "header.json"), ("events", "events.jsonl"), ("final_state", "final_state.json"))})
            self.closed = True
            return self.result
        except BaseException:
            self.failed = True
            raise

    def close_partial(self):
        self.failed = True
        if self.handle is not None:
            self.handle.close()
        if self.prefix_index is None:
            self.prefix.close_partial()
