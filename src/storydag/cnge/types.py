"""Data types for CNGE causal triple extraction and graph assembly."""

from dataclasses import dataclass, field
from typing import List, Literal

EdgeType = Literal["motivates", "informs", "emotionally_causes"]
NodeType = Literal["event", "intention", "revelation", "emotional_state"]

EDGE_TYPES = frozenset({"motivates", "informs", "emotionally_causes"})
NODE_TYPES = frozenset({"event", "intention", "revelation", "emotional_state"})


@dataclass(frozen=True)
class CausalTriple:
    """A directed causal edge between two narrative nodes within a scene chunk."""

    source_id: str
    source_label: str
    source_type: str
    edge_type: str
    target_id: str
    target_label: str
    target_type: str


@dataclass
class GraphNode:
    """A canonical narrative node after coreference resolution."""

    node_id: str
    label: str
    type: str
    source_ids: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class GraphEdge:
    """A deduplicated causal edge between canonical nodes."""

    edge_id: str
    source: str
    target: str
    type: str
    strength: float = 1.0
