from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .schemas import MemoryType


@dataclass(frozen=True)
class ExtractedMemory:
    memory_type: MemoryType
    category: str
    key: str
    value: str
    evidence: str
    confidence: float = 0.75
    attributes: dict[str, Any] = field(default_factory=dict)
