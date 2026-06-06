"""Tests for verifiable script YAML serialization."""

from pathlib import Path

import pytest

from storydag.causopt.optimize import SceneRecord, SceneSequence
from storydag.cgca.generator import CharacterLine
from storydag.serialization import (
    ScriptBeat,
    ScriptCharacterLine,
    ScriptValidationError,
    build_script,
    read_script,
    validate_script,
    write_script,
)


def _sample_sequence() -> SceneSequence:
    return SceneSequence(
        scenes=[
            SceneRecord(
                scene_id="S01",
                act=1,
                assigned_node_ids=["n1"],
                narrative_time_clue={"earliest": 0, "latest": 0},
            ),
            SceneRecord(
                scene_id="S02",
                act=2,
                assigned_node_ids=["n2", "n3"],
                narrative_time_clue={"earliest": 1, "latest": 2},
            ),
        ],
        satisfied_edges={"e1": 1, "e2": 1},
        score=0.85,
    )


def test_build_script_groups_acts_and_satisfied_edges():
    sequence = _sample_sequence()
    beats = {
        "S01": [
            ScriptBeat(
                description="夜袭前奏",
                characters=[
                    ScriptCharacterLine(
                        character="林远",
                        type="dialogue",
                        text="今夜必救人。",
                        causal_backlink=["e1"],
                    )
                ],
            )
        ],
        "S02": [
            ScriptBeat(
                description="潜入",
                characters=[
                    ScriptCharacterLine(
                        character="林远",
                        type="action",
                        text="林远翻过后墙。",
                        causal_backlink=["e2"],
                    )
                ],
            )
        ],
    }
    script = build_script(
        "测试剧本",
        sequence,
        settings={"S01": "城外驿站", "S02": "后山牢房"},
        beats_by_scene=beats,
        source_graph="outputs/test/causal_graph.json",
    )

    assert script.title == "测试剧本"
    assert len(script.acts) == 2
    assert script.acts[0].scenes[0].satisfied_edges == []
    assert script.acts[1].scenes[0].satisfied_edges == ["e1", "e2"]
    assert script.acts[1].scenes[0].beats[0].characters[0].type == "action"
    assert script.metadata.causopt_score == 0.85


def test_write_and_read_script_roundtrip(tmp_path: Path):
    sequence = _sample_sequence()
    script = build_script(
        "测试剧本",
        sequence,
        settings={"S01": "城外", "S02": "后山"},
        beats_by_scene={
            "S01": [ScriptBeat(description="开端", characters=[])],
            "S02": [ScriptBeat(description="高潮", characters=[])],
        },
    )
    path = tmp_path / "script.yaml"
    write_script(script, path)
    restored = read_script(path)

    assert restored.title == script.title
    assert restored.acts[1].scenes[0].scene_id == "S02"
    assert restored.acts[1].scenes[0].narrative_time_clue["latest"] == 2


def test_validate_script_rejects_invalid_line_type():
    with pytest.raises(ScriptValidationError, match="line type"):
        build_script(
            "坏剧本",
            _sample_sequence(),
            settings={"S01": "a", "S02": "b"},
            beats_by_scene={
                "S01": [
                    ScriptBeat(
                        description="x",
                        characters=[
                            ScriptCharacterLine(character="甲", type="invalid", text="...")
                        ],
                    )
                ],
                "S02": [],
            },
        )
