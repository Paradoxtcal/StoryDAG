"""Read causally verifiable script YAML artifacts."""

from __future__ import annotations

from pathlib import Path

import yaml

from storydag.serialization.schema import ScriptYAML, script_from_dict


def read_script(path: str | Path) -> ScriptYAML:
    """Load and validate a script YAML file."""
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("script YAML root must be a mapping")
    return script_from_dict(payload)
