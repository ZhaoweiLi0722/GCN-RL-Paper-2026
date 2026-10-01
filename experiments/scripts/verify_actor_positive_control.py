"""Independently reconcile artificial actor-control receipts without fitting."""

import json
from pathlib import Path

from src.rl.actor_control_verification import verify


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[2])))
