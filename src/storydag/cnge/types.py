"""Data types for CNGE causal triple extraction."""

from dataclasses import dataclass
from typing import Literal

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
