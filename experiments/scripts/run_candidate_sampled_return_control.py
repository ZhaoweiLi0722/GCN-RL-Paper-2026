"""Launch only the separately approved artificial diagnostic packet."""

from pathlib import Path

from src.rl.sampled_return_campaign import run


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[2])
