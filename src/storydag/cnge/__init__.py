"""CNGE: Causal Narrative Graph Extraction."""

from storydag.cnge.segmentation import Chapter, Scene, segment_novel, segment_scenes, segment_into_chapters

__all__ = [
    "Chapter",
    "Scene",
    "segment_novel",
    "segment_scenes",
    "segment_into_chapters",
]
