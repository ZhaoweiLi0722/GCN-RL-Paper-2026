"""Exclusive, step-flushed raw evidence for the bounded P1 runner.

Does not create/reset/step an environment, fit a model or resume a failed run.
Partial traces remain evidence, never silently upgraded to completed outcomes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from src.rl.candidate_pilot_driver import file_record
from src.rl.candidate_pilot_verification import verify_episode
from src.rl.patient_replay_collector import json_value


def normalized(data):
    return json.loads(json.dumps(data, default=json_value, sort_keys=True, allow_nan=False))


def write_json_once(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, default=json_value, sort_keys=True, indent=2, allow_nan=False) + "\n"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


class EpisodeRecorder:
    def __init__(self, root, relative, session, config, *, block, representation, role, world_index, seed):
        self.root, self.directory = Path(root).resolve(), Path(root).resolve() / relative
        if not self.directory.resolve().is_relative_to(self.root) or self.directory.resolve() != self.directory:
            raise ValueError("episode evidence must stay in the declared root")
        self.directory.mkdir(parents=True, exist_ok=False)
        if session.index != 0 or session.env.t != 0:
            raise ValueError("fresh raw recorder must start before the first step")
        self.session, self.config = session, config
        self.count, self.failed, self.closed = 0, False, False
        self.header = normalized({"format": "candidate-pilot-episode-v1", "block": block,
            "representation": representation, "role": role, "world_index": world_index, "seed": seed,
            "trajectory_id": session.trajectory_id, "source_id": session.source_id, "split": session.split,
            "selection": session.selection, "initial_state": session.env.state_dict(),
            "policy_sha256": session.learner.policy.snapshot_sha256(), "session_manifest": session.manifest,
            "inference_timing": "raw observation, reference/options, candidate scores and checked submission; excludes step and I/O"})
        write_json_once(self.directory / "header.json", self.header)
        self.path = self.directory / "events.jsonl"
        self.handle = self.path.open("x", encoding="utf-8")

    def append(self, event):
        if self.failed or self.closed or self.session.index != self.count + 1:
            raise ValueError("closed/failed recorder or skipped/duplicate raw step")
        record = event["audit"]["record"]
        if (record["trajectory_id"] != self.header["trajectory_id"] or record["source_id"] != self.header["source_id"]
                or record["step_index"] != self.count):
            raise ValueError("raw recorder lineage mismatch")
        try:
            row = normalized({"event": event, "inference_seconds": self.session.last_inference_seconds})
            self.handle.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            self.handle.flush()
            os.fsync(self.handle.fileno())
        except BaseException:
            self.failed = True
            raise
        self.count += 1

    def snapshot(self):
        if self.failed or self.closed:
            raise ValueError("recorder failed/closed")
        self.handle.flush()
        return {"rows": self.count, "file": file_record(self.root, self.path),
                "header": file_record(self.root, self.directory / "header.json")}

    def assert_prefix(self, saved):
        """Read-only recovery check; never truncate existing later rows."""
        if saved["header"] != file_record(self.root, self.directory / "header.json"):
            raise ValueError("recorder header changed")
        raw = self.path.read_bytes()
        prefix = raw[:saved["file"]["bytes"]]
        if (saved["file"]["path"] != self.path.relative_to(self.root).as_posix()
                or hashlib.sha256(prefix).hexdigest() != saved["file"]["sha256"]
                or len(prefix.splitlines()) != saved["rows"] or saved["rows"] > self.count):
            raise ValueError("raw checkpoint is not a preserved trace prefix")

    def finish(self):
        if self.failed or self.closed or not self.session.closed or self.count != self.config["objective"]["horizon"]:
            raise ValueError("incomplete/failed episode cannot publish an outcome")
        self.handle.close()
        try:
            header = json.loads((self.directory / "header.json").read_text())
            rows = [json.loads(line) for line in self.path.read_text().splitlines()]
            final = normalized(self.session.env.state_dict())
            # Persist the raw endpoint before auditing it, including on audit failure.
            write_json_once(self.directory / "final_state.json", final)
            result = verify_episode(header, rows, final, self.config)
            self.session.save(self.directory / "collector.pt")
            write_json_once(self.directory / "outcome.json", result)
        except BaseException:
            self.failed = True
            raise
        self.closed = True
        return {"header": file_record(self.root, self.directory / "header.json"),
                "events": file_record(self.root, self.path),
                "final_state": file_record(self.root, self.directory / "final_state.json")}

    def close_partial(self):
        """Close a failure's file handles; keep every partial byte and no outcome."""
        self.handle.close()
        self.failed = True
