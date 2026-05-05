from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import _normalize_location, _sentence_fragment


def _extract_events(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    event_boundary = r"(?=\s+(?:next|this|tomorrow|today|on|in)\b|[.!?,;]|$)"
    preparing_match = re.search(
        rf"\b(?:i'm|i am|we're|we are)?\s*(?:preparing|studying|getting ready)\s+for\s+(?P<event>[A-Za-z0-9&+/\s.'-]+?){event_boundary}",
        text,
        flags=re.IGNORECASE,
    )
    if preparing_match:
        memories.append(
            ExtractedMemory(
                memory_type="event",
                category="project_goal",
                key="upcoming_focus",
                value=f"Preparing for {_sentence_fragment(preparing_match.group('event'))}",
                evidence=text,
                confidence=0.74,
            )
        )

    interview_match = re.search(
        rf"\b(?:i\s+have|i've got|i got)\s+(?:an?|the)?\s*(?P<event>[A-Za-z0-9&+/\s.'-]*interview[A-Za-z0-9&+/\s.'-]*?){event_boundary}",
        text,
        flags=re.IGNORECASE,
    )
    if interview_match:
        memories.append(
            ExtractedMemory(
                memory_type="event",
                category="project_goal",
                key="upcoming_focus",
                value=_sentence_fragment(interview_match.group("event")),
                evidence=text,
                confidence=0.72,
            )
        )

    travel_match = re.search(
        r"\b(?:planning|booking|taking)\s+(?:a\s+)?trip\s+to\s+(?P<place>[A-Za-z][A-Za-z0-9\s.'-]+?)(?:\s+next\b|\s+this\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if travel_match:
        memories.append(
            ExtractedMemory(
                memory_type="event",
                category="project_goal",
                key="travel_plan",
                value=f"Planning a trip to {_normalize_location(travel_match.group('place'))}",
                evidence=text,
                confidence=0.72,
            )
        )

    return memories


