from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import _normalize_person_name, _normalize_relationship


def _extract_pet_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    patterns = (
        re.compile(
            r"\b(?:my|our)?\s*(?P<animal>dog|cat)\s+(?:is\s+)?(?:named|called)\s+(?P<pet>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+have\s+(?:a|an)\s+(?P<animal>dog|cat)\s+(?:named|called)\s+(?P<pet>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bmy\s+(?P<animal>dog|cat)'?s\s+name\s+is\s+(?P<pet>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?P<pet>[A-Z][A-Za-z'-]+)\s+is\s+(?:my|our)\s+(?P<animal>dog|cat)\b",
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        animal = match.group("animal").title()
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="pet",
                value=f"{animal} named {_normalize_person_name(match.group('pet'))}",
                evidence=text,
                confidence=0.82,
            )
        )

    walking_match = re.search(
        r"\b(?:walking|walked|taking|took)\s+(?P<pet>[A-Za-z][A-Za-z'-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if walking_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="pet",
                value=f"Dog named {_normalize_person_name(walking_match.group('pet'))}",
                evidence=text,
                confidence=0.74,
                attributes={"inferred_from": "walking"},
            )
        )

    return memories


def _extract_family_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    patterns = (
        re.compile(
            r"\bmy\s+(?P<relation>son|daughter|child|kid)(?:'s name)?\s+(?:is\s+)?(?:named\s+|called\s+)?(?P<name>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\b(?:i|we)\s+have\s+(?:a|an)\s+(?P<relation>son|daughter|child|kid)\s+(?:named|called)\s+(?P<name>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
        re.compile(
            r"\bmy\s+(?P<relation>wife|husband|partner|spouse)\s+(?:is\s+)?(?:named\s+|called\s+)?(?P<name>[A-Za-z][A-Za-z'-]+)\b",
            flags=re.IGNORECASE,
        ),
    )
    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue
        relation = _normalize_relationship(match.group("relation"))
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="family",
                value=f"Has a {relation} named {_normalize_person_name(match.group('name'))}",
                evidence=text,
                confidence=0.78,
            )
        )
    return memories
