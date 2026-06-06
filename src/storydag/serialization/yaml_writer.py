"""Write causally verifiable script YAML artifacts."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import yaml

from storydag.causopt.optimize import SceneRecord, SceneSequence
from storydag.cgca.generator import CharacterLine
from storydag.cnge.graph import CausalGraph
from storydag.serialization.schema import (
    ScriptAct,
    ScriptBeat,
    ScriptCharacterLine,
    ScriptMetadata,
    ScriptScene,
    ScriptYAML,
    script_to_dict,
    validate_script,
)


def _satisfied_edges_for_scene(sequence: SceneSequence, scene_id: str) -> List[str]:
    return sorted(
        edge_id
        for edge_id, scene_index in sequence.satisfied_edges.items()
        if sequence.scenes[scene_index].scene_id == scene_id
    )


def _lines_to_beat(lines: Sequence[CharacterLine], description: str = "") -> ScriptBeat:
    return ScriptBeat(
        description=description,
        characters=[
            ScriptCharacterLine(
                character=line.character,
                type=line.type,
                text=line.text,
                causal_backlink=list(line.causal_backlink),
            )
            for line in lines
        ],
    )


def build_scene(
    scene_record: SceneRecord,
    *,
    setting: str,
    beats: Sequence[ScriptBeat],
    sequence: SceneSequence,
) -> ScriptScene:
    """Build one ``ScriptScene`` from CausOpt output plus CGCA beats."""
    return ScriptScene(
        scene_id=scene_record.scene_id,
        setting=setting,
        act=scene_record.act,
        satisfied_edges=_satisfied_edges_for_scene(sequence, scene_record.scene_id),
        assigned_node_ids=list(scene_record.assigned_node_ids),
        narrative_time_clue=dict(scene_record.narrative_time_clue),
        beats=list(beats),
    )


def build_script(
    title: str,
    sequence: SceneSequence,
    *,
    settings: Mapping[str, str],
    beats_by_scene: Mapping[str, Sequence[ScriptBeat]],
    source_graph: Optional[str] = None,
) -> ScriptYAML:
    """Assemble a full script document from CausOpt and CGCA outputs."""
    acts_map: Dict[int, List[ScriptScene]] = defaultdict(list)
    for scene_record in sequence.scenes:
        scene_beats = beats_by_scene.get(scene_record.scene_id, [])
        acts_map[scene_record.act].append(
            build_scene(
                scene_record,
                setting=settings.get(scene_record.scene_id, ""),
                beats=scene_beats,
                sequence=sequence,
            )
        )

    acts = [
        ScriptAct(act=act_number, scenes=scenes)
        for act_number, scenes in sorted(acts_map.items())
    ]
    script = ScriptYAML(
        title=title,
        acts=acts,
        metadata=ScriptMetadata(
            causopt_score=sequence.score,
            source_graph=source_graph,
        ),
    )
    validate_script(script)
    return script


def write_script(script: ScriptYAML, path: str | Path) -> None:
    """Validate and write a script YAML file."""
    validate_script(script)
    target = Path(path)
    payload = script_to_dict(script)
    target.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
