"""YAML serialization and causal closure metrics (CCR)."""

from storydag.serialization.schema import (
    ScriptAct,
    ScriptBeat,
    ScriptCharacterLine,
    ScriptMetadata,
    ScriptScene,
    ScriptValidationError,
    ScriptYAML,
    script_from_dict,
    script_to_dict,
    validate_script,
)
from storydag.serialization.yaml_reader import read_script
from storydag.serialization.yaml_writer import build_scene, build_script, write_script

__all__ = [
    "ScriptAct",
    "ScriptBeat",
    "ScriptCharacterLine",
    "ScriptMetadata",
    "ScriptScene",
    "ScriptValidationError",
    "ScriptYAML",
    "build_scene",
    "build_script",
    "read_script",
    "script_from_dict",
    "script_to_dict",
    "validate_script",
    "write_script",
]
