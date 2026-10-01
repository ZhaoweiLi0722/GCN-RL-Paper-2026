"""Verify saved artificial receipts; never call a model or patient environment."""

import json
from pathlib import Path

from src.rl.sampled_return_packet_verification import verify


if __name__ == "__main__":
    print(json.dumps(verify(Path(__file__).resolve().parents[2])))
