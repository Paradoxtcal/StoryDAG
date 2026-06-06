"""Novel chapter and scene segmentation for CNGE input preprocessing."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple

# 第[一二三四五六七八九十百千0-9]+章 — algorithm-specified chapter header pattern.
CHAPTER_HEADER_RE = re.compile(
    r"第[一二三四五六七八九十百千0-9]+章[^\n\r]*",
    re.MULTILINE,
)

# Scene break delimiters: asterisk lines, horizontal rules, double blank lines,
# or time-skip marker lines at the start of a line.
TIME_SKIP_MARKERS = (
    "三年后",
    "翌日",
    "次日",
    "与此同时",
    "数日后",
    "片刻后",
    "半个时辰后",
    "一炷香后",
    "转眼",
    "多年后",
    "几日后",
    "当天夜里",
    "翌晨",
    "数年之后",
    "数月后",
    "数年后",
)
_TIME_SKIP_PATTERN = "|".join(re.escape(marker) for marker in TIME_SKIP_MARKERS)

SCENE_SPLIT_RE = re.compile(
    rf"""
    (?:
        ^\s*\*+\s*$
      | ^\s*[-—=]{{3,}}\s*$
      | \n\s*\n\s*\n
      | (?=^(?:{_TIME_SKIP_PATTERN})[^\n\r]*$)
    )
    """,
    re.MULTILINE | re.VERBOSE,
)


@dataclass
class Scene:
    """A narrative scene chunk within a chapter, processed independently by CNGE."""

    scene_id: str
    index: int
    text: str


@dataclass
class Chapter:
    """A novel chapter containing one or more scenes."""

    chapter_id: str
    index: int
    title: str
    scenes: List[Scene] = field(default_factory=list)

    @property
    def text(self) -> str:
        """Full chapter text reconstructed from scene bodies."""
        return "\n\n".join(scene.text for scene in self.scenes)


def segment_into_chapters(text: str) -> List[Tuple[str, str, str]]:
    """Split novel text into ``(chapter_id, title, body)`` tuples.

    When no chapter headers are found the entire input is returned as a single
    chapter ``ch1`` titled ``正文``.
    """
    normalized = text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    matches = list(CHAPTER_HEADER_RE.finditer(normalized))
    if not matches:
        return [("ch1", "正文", normalized)]

    chapters: List[Tuple[str, str, str]] = []
    for index, match in enumerate(matches):
        chapter_id = f"ch{index + 1}"
        title = match.group(0).strip()
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        body = normalized[body_start:body_end].strip()
        chapters.append((chapter_id, title, body))
    return chapters


def segment_scenes(chapter_id: str, chapter_text: str) -> List[Scene]:
    """Split a chapter body into scene chunks.

    Scene boundaries are detected via asterisk lines, horizontal rules,
    double blank lines, or time-skip marker lines. If no boundary is found the
    whole chapter becomes a single scene.
    """
    normalized = chapter_text.replace("\r\n", "\n").strip()
    if not normalized:
        return []

    raw_parts = SCENE_SPLIT_RE.split(normalized)
    scenes: List[Scene] = []
    for part in raw_parts:
        cleaned = part.strip()
        if not cleaned:
            continue
        scenes.append(
            Scene(
                scene_id=f"{chapter_id}_s{len(scenes) + 1}",
                index=len(scenes),
                text=cleaned,
            )
        )

    if not scenes:
        return [Scene(scene_id=f"{chapter_id}_s1", index=0, text=normalized)]

    for index, scene in enumerate(scenes):
        scene.index = index
        scene.scene_id = f"{chapter_id}_s{index + 1}"
    return scenes


def segment_novel(text: str) -> List[Chapter]:
    """Segment a full novel into chapters and scenes for CNGE chunking."""
    chapters: List[Chapter] = []
    for index, (chapter_id, title, body) in enumerate(segment_into_chapters(text)):
        chapters.append(
            Chapter(
                chapter_id=chapter_id,
                index=index,
                title=title,
                scenes=segment_scenes(chapter_id, body),
            )
        )
    return chapters
