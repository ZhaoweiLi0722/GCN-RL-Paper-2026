"""Execute the single frozen artificial actor-only packet, never patient RL."""

from pathlib import Path

from src.rl.actor_positive_control import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
