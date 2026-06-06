"""Dataclass schema for verifiable StoryDAG script YAML output.

Design goals (see ``schemas/script_output.yaml`` for field-level rationale):

1. **Causal verifiability** — every scene and line can be traced back to CNGE edges.
2. **Production readiness** — acts / scenes / beats mirror screenplay structure.
3. **Metric computability** — ``satisfied_edges`` and ``causal_backlink`` feed PR #14 CCR.
4. **Reproducibility** — metadata preserves optimizer score and graph provenance.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

LineType = Literal["dialogue", "action"]
VALID_LINE_TYPES = frozenset({"dialogue", "action"})


@dataclass
class ScriptCharacterLine:
    """One dialogue or action line with per-line causal traceability."""

    character: str
    type: str
    text: str
    causal_backlink: List[str] = field(default_factory=list)


@dataclass
class ScriptBeat:
    """A narrative beat grouping related character lines within a scene."""

    description: str
    characters: List[ScriptCharacterLine] = field(default_factory=list)


@dataclass
class ScriptScene:
    """A production scene linked to CausOpt ordering and CNGE nodes."""

    scene_id: str
    setting: str
    act: int
    satisfied_edges: List[str] = field(default_factory=list)
    assigned_node_ids: List[str] = field(default_factory=list)
    narrative_time_clue: Dict[str, int] = field(default_factory=dict)
    beats: List[ScriptBeat] = field(default_factory=list)


@dataclass
class ScriptAct:
    """One dramatic act containing an ordered list of scenes."""

    act: int
    scenes: List[ScriptScene] = field(default_factory=list)


@dataclass
class ScriptMetadata:
    """Reproducibility and evaluation metadata carried with the script artifact."""

    causopt_score: Optional[float] = None
    source_graph: Optional[str] = None
    schema_version: str = "1.0"


@dataclass
class ScriptYAML:
    """Root document for a causally verifiable adapted script."""

    title: str
    acts: List[ScriptAct] = field(default_factory=list)
    metadata: ScriptMetadata = field(default_factory=ScriptMetadata)


class ScriptValidationError(ValueError):
    """Raised when script YAML fails schema validation."""


def validate_script(script: ScriptYAML) -> None:
    """Validate required fields and enum constraints."""
    if not script.title.strip():
        raise ScriptValidationError("title must be non-empty")

    seen_scene_ids: set[str] = set()
    for act in script.acts:
        if act.act < 1:
            raise ScriptValidationError(f"act number must be >= 1, got {act.act}")
        for scene in act.scenes:
            if not scene.scene_id:
                raise ScriptValidationError("scene_id must be non-empty")
            if scene.scene_id in seen_scene_ids:
                raise ScriptValidationError(f"duplicate scene_id: {scene.scene_id}")
            seen_scene_ids.add(scene.scene_id)
            for beat in scene.beats:
                for line in beat.characters:
                    if line.type not in VALID_LINE_TYPES:
                        raise ScriptValidationError(
                            f"invalid line type '{line.type}' for {line.character}"
                        )


def script_to_dict(script: ScriptYAML) -> Dict[str, Any]:
    """Convert a script dataclass tree to a plain dict for YAML dumping."""
    return {
        "title": script.title,
        "metadata": {
            "schema_version": script.metadata.schema_version,
            "causopt_score": script.metadata.causopt_score,
            "source_graph": script.metadata.source_graph,
        },
        "acts": [
            {
                "act": act.act,
                "scenes": [
                    {
                        "scene_id": scene.scene_id,
                        "setting": scene.setting,
                        "act": scene.act,
                        "satisfied_edges": list(scene.satisfied_edges),
                        "assigned_node_ids": list(scene.assigned_node_ids),
                        "narrative_time_clue": dict(scene.narrative_time_clue),
                        "beats": [
                            {
                                "description": beat.description,
                                "characters": [
                                    {
                                        "character": line.character,
                                        "type": line.type,
                                        "text": line.text,
                                        "causal_backlink": list(line.causal_backlink),
                                    }
                                    for line in beat.characters
                                ],
                            }
                            for beat in scene.beats
                        ],
                    }
                    for scene in act.scenes
                ],
            }
            for act in script.acts
        ],
    }


def script_from_dict(payload: Dict[str, Any]) -> ScriptYAML:
    """Parse a plain dict (from YAML) into script dataclasses."""
    metadata_payload = payload.get("metadata", {}) or {}
    acts: List[ScriptAct] = []

    for act_payload in payload.get("acts", []) or []:
        scenes: List[ScriptScene] = []
        for scene_payload in act_payload.get("scenes", []) or []:
            beats: List[ScriptBeat] = []
            for beat_payload in scene_payload.get("beats", []) or []:
                characters = [
                    ScriptCharacterLine(
                        character=str(item["character"]),
                        type=str(item["type"]),
                        text=str(item["text"]),
                        causal_backlink=list(item.get("causal_backlink", [])),
                    )
                    for item in beat_payload.get("characters", []) or []
                ]
                beats.append(
                    ScriptBeat(
                        description=str(beat_payload.get("description", "")),
                        characters=characters,
                    )
                )
            scenes.append(
                ScriptScene(
                    scene_id=str(scene_payload["scene_id"]),
                    setting=str(scene_payload.get("setting", "")),
                    act=int(scene_payload.get("act", act_payload.get("act", 1))),
                    satisfied_edges=list(scene_payload.get("satisfied_edges", [])),
                    assigned_node_ids=list(scene_payload.get("assigned_node_ids", [])),
                    narrative_time_clue=dict(scene_payload.get("narrative_time_clue", {})),
                    beats=beats,
                )
            )
        acts.append(ScriptAct(act=int(act_payload["act"]), scenes=scenes))

    script = ScriptYAML(
        title=str(payload.get("title", "")),
        acts=acts,
        metadata=ScriptMetadata(
            causopt_score=metadata_payload.get("causopt_score"),
            source_graph=metadata_payload.get("source_graph"),
            schema_version=str(metadata_payload.get("schema_version", "1.0")),
        ),
    )
    validate_script(script)
    return script
