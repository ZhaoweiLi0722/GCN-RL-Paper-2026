"""Run only the frozen, single-attempt artificial calibration acceptance."""

from pathlib import Path

from src.rl.candidate_calibration_engineering import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
