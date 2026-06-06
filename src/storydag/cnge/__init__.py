"""CNGE: Causal Narrative Graph Extraction."""

from storydag.cnge.coref import resolve_coreferences
from storydag.cnge.graph import CausalGraph, RemovedCycleEdge, build_dag
from storydag.cnge.extractor import (
    CausalExtractionError,
    extract_novel_triples,
    extract_triples,
    extract_triples_from_text,
    normalize_triples,
    parse_triples_response,
)
from storydag.cnge.segmentation import Chapter, Scene, segment_into_chapters, segment_novel, segment_scenes
from storydag.cnge.types import CausalTriple, EDGE_TYPES, GraphEdge, GraphNode, NODE_TYPES

__all__ = [
    "CausalExtractionError",
    "CausalGraph",
    "CausalTriple",
    "Chapter",
    "EDGE_TYPES",
    "GraphEdge",
    "GraphNode",
    "NODE_TYPES",
    "RemovedCycleEdge",
    "Scene",
    "build_dag",
    "resolve_coreferences",
    "extract_novel_triples",
    "extract_triples",
    "extract_triples_from_text",
    "normalize_triples",
    "parse_triples_response",
    "segment_into_chapters",
    "segment_novel",
    "segment_scenes",
]
