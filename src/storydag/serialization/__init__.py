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
from storydag.serialization.metrics import (
    BacklinkViolation,
    MetricsReport,
    compute_causal_density,
    compute_ccr,
    compute_character_consistency,
    compute_metrics,
    compute_metrics_from_files,
    detect_plot_holes,
    verify_backlinks,
    write_metrics_report,
)
from storydag.serialization.yaml_reader import read_script
from storydag.serialization.yaml_writer import build_scene, build_script, write_script

__all__ = [
    "BacklinkViolation",
    "MetricsReport",
    "ScriptAct",
    "ScriptBeat",
    "ScriptCharacterLine",
    "ScriptMetadata",
    "ScriptScene",
    "ScriptValidationError",
    "ScriptYAML",
    "build_scene",
    "build_script",
    "compute_causal_density",
    "compute_ccr",
    "compute_character_consistency",
    "compute_metrics",
    "compute_metrics_from_files",
    "detect_plot_holes",
    "read_script",
    "verify_backlinks",
    "write_metrics_report",
    "script_from_dict",
    "script_to_dict",
    "validate_script",
    "write_script",
]
