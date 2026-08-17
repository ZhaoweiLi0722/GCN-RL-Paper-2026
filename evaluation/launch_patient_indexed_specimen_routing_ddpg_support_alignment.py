"""Detached launcher for the locked DDPG support-alignment experiment."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_LAUNCHER_ROOT = Path(
    "results/patient_indexed_specimen_routing_ddpg_support_alignment_"
    "development/launcher"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument(
        "--spec",
        default=(
            "experiments/configs/"
            "patient_indexed_specimen_routing_ddpg_support_alignment_"
            "execution.json"
        ),
    )
    args = parser.parse_args()

    spec_path = Path(args.spec)
    spec = json.loads(spec_path.read_text(encoding="utf-8-sig"))
    launcher_root = Path(spec.get("launcher_root", DEFAULT_LAUNCHER_ROOT))
    launcher_root.mkdir(parents=True, exist_ok=False)
    stdout_path = launcher_root / "detached.stdout.log"
    stderr_path = launcher_root / "detached.stderr.log"
    command = [
        sys.executable,
        "-m",
        "evaluation.run_patient_indexed_specimen_routing_ddpg_support_alignment",
        "--expected-commit",
        str(args.expected_commit),
        "--spec",
        str(spec_path),
    ]
    environment = os.environ.copy()
    environment["PYTHONPATH"] = "."
    environment["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    popen_kwargs = {
        "stdin": subprocess.DEVNULL,
        "cwd": str(Path.cwd()),
        "env": environment,
        "close_fds": True,
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = (
            subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
        )
    else:
        popen_kwargs["start_new_session"] = True
    with stdout_path.open("xb") as stdout_handle, stderr_path.open(
        "xb"
    ) as stderr_handle:
        process = subprocess.Popen(
            command,
            stdout=stdout_handle,
            stderr=stderr_handle,
            **popen_kwargs,
        )
    launch = {
        "launched_at": datetime.now(timezone.utc).isoformat(),
        "launcher_pid": os.getpid(),
        "detached_pid": process.pid,
        "command": command,
        "cwd": str(Path.cwd().resolve()),
        "python": sys.executable,
        "expected_commit": str(args.expected_commit),
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }
    (launcher_root / "launch.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"DDPG support-alignment detached PID: {process.pid}", flush=True)


if __name__ == "__main__":
    main()
