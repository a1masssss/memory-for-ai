from __future__ import annotations

from ..memory_types import ExtractedMemory
from ..schemas import Message
from .common import _dedupe, _normalize_text
from .creative import _extract_creative_preferences, _extract_generation_feedback
from .events import _extract_events
from .personal import _extract_personal_facts


def extract_rule_based_memories(messages: list[Message]) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    user_texts = [
        _normalize_text(message.content)
        for message in messages
        if message.role == "user"
    ]

    for text in user_texts:
        memories.extend(_extract_personal_facts(text))
        memories.extend(_extract_events(text))
        memories.extend(_extract_creative_preferences(text))
        memories.extend(_extract_generation_feedback(text))

    return _dedupe(memories)
