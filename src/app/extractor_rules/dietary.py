from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import _clean_capture, _pluralize_food, _revision_attributes


def _extract_dietary_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    lowered = text.lower()
    revision_attributes = _revision_attributes(text)

    if "vegetarian" in lowered or "vegan" in lowered:
        dietary_value = "Vegan" if "vegan" in lowered else "Vegetarian"
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="personal_context",
                key="dietary_preference",
                value=dietary_value,
                evidence=text,
                confidence=0.82,
                attributes=dict(revision_attributes),
            )
        )

    if "pescatarian" in lowered:
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="personal_context",
                key="dietary_preference",
                value="Pescatarian",
                evidence=text,
                confidence=0.82,
                attributes=dict(revision_attributes),
            )
        )

    if any(phrase in lowered for phrase in ("don't eat meat", "do not eat meat", "no meat")):
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="personal_context",
                key="dietary_preference",
                value="Vegetarian",
                evidence=text,
                confidence=0.78,
                attributes=dict(revision_attributes),
            )
        )

    dietary_avoidance_match = re.search(
        r"\b(?:avoid|don't eat|do not eat|can't eat|cannot eat)\s+(?P<food>[a-zA-Z0-9 ,&'-]+?)(?:[.!?,;]| and I\b| but\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if dietary_avoidance_match:
        avoided_food = _clean_capture(dietary_avoidance_match.group("food")).lower()
        if avoided_food and avoided_food != "meat":
            memories.append(
                ExtractedMemory(
                    memory_type="preference",
                    category="personal_context",
                    key="dietary_preference",
                    value=f"Avoids {avoided_food}",
                    evidence=text,
                    confidence=0.76,
                    attributes=dict(revision_attributes),
                )
            )

    allergy_match = re.search(
        r"\b(?:allergic to|allergy to)\s+(?P<allergy>[a-zA-Z0-9 ,&'-]+?)(?:[.!?,;]| and I\b| but\b|$)",
        text,
        flags=re.IGNORECASE,
    )
    if allergy_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="allergy",
                value=f"Allergic to {_clean_capture(allergy_match.group('allergy')).lower()}",
                evidence=text,
                confidence=0.86,
                attributes=dict(revision_attributes),
            )
        )

    allergy_noun_match = re.search(
        r"\b(?:have|has|with)\s+(?:a|an)?\s*(?P<allergy>[a-zA-Z0-9 ,&'-]+?)\s+allerg(?:y|ies)\b",
        text,
        flags=re.IGNORECASE,
    )
    if allergy_noun_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="allergy",
                value=f"Allergic to {_pluralize_food(_clean_capture(allergy_noun_match.group('allergy')).lower())}",
                evidence=text,
                confidence=0.78,
                attributes=dict(revision_attributes),
            )
        )

    if "lactose intolerant" in lowered:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="allergy",
                value="Lactose intolerant",
                evidence=text,
                confidence=0.78,
                attributes=dict(revision_attributes),
            )
        )

    return memories
