"""Experiment config loading without a hard PyYAML dependency."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a JSON/YAML config.

    The checked-in ``.yaml`` files use JSON-compatible YAML syntax so this loader
    can parse them with the standard library. If richer YAML is needed later,
    add PyYAML and extend this function.
    """

    config_path = Path(path)
    text = config_path.read_text()
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{config_path} is not JSON-compatible YAML. Use JSON syntax or add PyYAML support."
        ) from exc


def save_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(config, indent=2, sort_keys=True))
