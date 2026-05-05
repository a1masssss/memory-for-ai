from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import (
    _clean_capture,
    _normalize_topic,
    _revision_attributes,
    _slug,
    _split_clauses,
)


def _extract_opinions(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    for clause in _split_clauses(text):
        revision_attributes = _revision_attributes(clause)

        explicit_match = re.search(
            r"\bI\s+(?P<verb>love|like|enjoy|prefer|hate|dislike)\s+(?P<topic>[A-Za-z0-9+#][A-Za-z0-9+#./\s-]+?)(?:[.!?,;]|$)",
            clause,
            flags=re.IGNORECASE,
        )
        if explicit_match:
            verb = explicit_match.group("verb").lower()
            topic = _normalize_topic(explicit_match.group("topic"))
            stance = {
                "love": "positive",
                "like": "positive",
                "enjoy": "positive",
                "prefer": "positive",
                "hate": "negative",
                "dislike": "negative",
            }[verb]
            value = (
                f"Dislikes {topic}"
                if stance == "negative"
                else f"Likes {topic}"
            )
            memories.append(
                _opinion_memory(
                    topic=topic,
                    value=value,
                    evidence=clause,
                    confidence=0.74,
                    stance=stance,
                    attributes=revision_attributes,
                )
            )
            continue

        conditional_match = re.search(
            r"\b(?P<topic>[A-Za-z][A-Za-z0-9+#./\s-]+?)\s+is\s+fine\s+for\s+(?P<context>[A-Za-z0-9\s-]+?)(?:[.!?,;]|$)",
            clause,
            flags=re.IGNORECASE,
        )
        if conditional_match:
            topic = _normalize_topic(conditional_match.group("topic"))
            context = _clean_capture(conditional_match.group("context")).lower()
            memories.append(
                _opinion_memory(
                    topic=topic,
                    value=f"{topic} is fine for {context}",
                    evidence=clause,
                    confidence=0.76,
                    stance="conditional",
                    attributes={**revision_attributes, "revision_kind": "conditional"},
                )
            )
            continue

        nuanced_match = re.search(
            r"\b(?P<topic>[A-Za-z][A-Za-z0-9+#./\s-]{1,40}?)(?:\s+(?P<subtopic>generics|syntax|tooling|ecosystem))?\s+(?:(?:is|are)\s+getting|is|are|feels?|gets?|getting)\s+(?P<descriptor>annoying|frustrating|great|solid|fine|bad|rough|noisy)\b",
            clause,
            flags=re.IGNORECASE,
        )
        if nuanced_match:
            topic = _normalize_topic(nuanced_match.group("topic"))
            descriptor = nuanced_match.group("descriptor").lower()
            stance = (
                "negative"
                if descriptor in {"annoying", "frustrating", "bad", "rough", "noisy"}
                else "positive"
            )
            attributes = dict(revision_attributes)
            if nuanced_match.group("subtopic"):
                attributes["subtopic"] = nuanced_match.group("subtopic").lower()
            memories.append(
                _opinion_memory(
                    topic=topic,
                    value=_clean_capture(clause),
                    evidence=clause,
                    confidence=0.73,
                    stance=stance,
                    attributes=attributes,
                )
            )
            continue

        usage_match = re.search(
            r"\bI(?:'d| would)\s+use\s+(?P<topic>[A-Za-z0-9+#][A-Za-z0-9+#./\s-]+?)\s+for\s+(?P<context>[A-Za-z0-9\s-]+?)(?:[.!?,;]|$)",
            clause,
            flags=re.IGNORECASE,
        )
        if usage_match:
            topic = _normalize_topic(usage_match.group("topic"))
            context = _clean_capture(usage_match.group("context")).lower()
            memories.append(
                _opinion_memory(
                    topic=topic,
                    value=f"Would use {topic} for {context}",
                    evidence=clause,
                    confidence=0.72,
                    stance="preference",
                    attributes=revision_attributes,
                )
            )
    return memories


def _opinion_memory(
    *,
    topic: str,
    value: str,
    evidence: str,
    confidence: float,
    stance: str,
    attributes: dict[str, str],
) -> ExtractedMemory:
    return ExtractedMemory(
        memory_type="opinion",
        category="opinions",
        key=_slug(topic),
        value=value,
        evidence=evidence,
        confidence=confidence,
        attributes={
            "topic": topic,
            "stance": stance,
            **attributes,
        },
    )

