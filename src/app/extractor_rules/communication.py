from __future__ import annotations

from ..memory_types import ExtractedMemory
from .common import _revision_attributes


def _extract_communication_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    lowered = text.lower()
    revision_attributes = _revision_attributes(text)

    if any(
        phrase in lowered
        for phrase in (
            "concise",
            "direct answers",
            "brief answers",
            "brief and direct",
            "keep it brief",
            "keep answers brief",
            "short answers",
        )
    ):
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="communication",
                key="answer_style",
                value="Prefers concise, direct answers",
                evidence=text,
                confidence=0.78,
                attributes=dict(revision_attributes),
            )
        )

    if (
        any(
            phrase in lowered
            for phrase in (
                "detailed answers",
                "thorough answers",
                "step-by-step",
                "step by step",
                "explain the reasoning",
                "more detail",
            )
        )
        and not any(
            phrase in lowered
            for phrase in ("concise", "brief", "short answers", "keep it brief")
        )
    ):
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="communication",
                key="answer_style",
                value="Prefers detailed, step-by-step answers",
                evidence=text,
                confidence=0.76,
                attributes=dict(revision_attributes),
            )
        )

    return memories
