from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import (
    _normalize_location,
    _normalize_person_name,
    _revision_attributes,
)


def _extract_identity_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    revision_attributes = _revision_attributes(text)

    name_match = re.search(
        r"\b(?:my name is|call me)\s+(?P<name>[A-Za-z][A-Za-z'-]{1,40})(?:[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if name_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="name",
                value=_normalize_person_name(name_match.group("name")),
                evidence=text,
                confidence=0.82,
                attributes=dict(revision_attributes),
            )
        )

    corrected_location_match = re.search(
        r"\b(?:actually|sorry)[, ]+\s*not\s+[A-Za-z][A-Za-z0-9\s.'-]+?\s*-\s*(?:(?:i\s+)?(?:live in|am based in|i'm based in)\s+)?(?P<city>[A-Za-z][A-Za-z0-9\s.'-]+?)(?: now\b| these days\b| currently\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if corrected_location_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value=_normalize_location(corrected_location_match.group("city")),
                evidence=text,
                confidence=0.86,
                attributes=dict(revision_attributes),
            )
        )

    moved_match = re.search(
        r"\b(?:i\s+)?(?:just\s+)?(?:moved|relocated) to (?P<city>[A-Za-z][A-Za-z0-9\s.'-]+?)(?: from (?P<from>[A-Za-z][A-Za-z0-9\s.'-]+?))?(?: last\b| now\b| these days\b| recently\b|\sand\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if moved_match:
        city = _normalize_location(moved_match.group("city"))
        attributes = dict(revision_attributes)
        if moved_match.group("from"):
            attributes["previous_location"] = _normalize_location(
                moved_match.group("from")
            )
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value=city,
                evidence=text,
                confidence=0.9,
                attributes=attributes,
            )
        )

    location_match = re.search(
        r"\b(?:actually\s+|sorry\s+)?(?:i\s+)?(?:live in|am based in|i'm based in|based out of|currently live in|currently based in|home base is|call home)\s+(?P<city>[A-Za-z][A-Za-z0-9\s.'-]+?)(?: now\b| these days\b| currently\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if location_match and not moved_match and not corrected_location_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value=_normalize_location(location_match.group("city")),
                evidence=text,
                confidence=0.84,
                attributes=dict(revision_attributes),
            )
        )

    currently_in_match = re.search(
        r"\b(?:i'm|i am|currently)\s+in\s+(?P<city>[A-Z][A-Za-z0-9\s.'-]+?)(?: now\b| these days\b| currently\b|[.!?,;]|$)",
        text,
    )
    if (
        currently_in_match
        and not moved_match
        and not corrected_location_match
        and not location_match
    ):
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value=_normalize_location(currently_in_match.group("city")),
                evidence=text,
                confidence=0.76,
                attributes=dict(revision_attributes),
            )
        )

    return memories
