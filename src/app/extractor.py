from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .schemas import Message, MemoryType


@dataclass(frozen=True)
class ExtractedMemory:
    memory_type: MemoryType
    category: str
    key: str
    value: str
    evidence: str
    confidence: float = 0.75
    attributes: dict[str, Any] = field(default_factory=dict)


def extract_memories(messages: list[Message]) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    user_texts = [message.content for message in messages if message.role == "user"]

    for text in user_texts:
        memories.extend(_extract_personal_facts(text))
        memories.extend(_extract_creative_preferences(text))
        memories.extend(_extract_generation_feedback(text))

    return _dedupe(memories)


def _extract_personal_facts(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []

    moved_match = re.search(
        r"\b(?:i\s+)?(?:just\s+)?moved to (?P<city>[A-Z][A-Za-z\s.-]+?)(?: from (?P<from>[A-Z][A-Za-z\s.-]+?))?(?: last|\sand|\.|,|$)",
        text,
    )
    if moved_match:
        city = moved_match.group("city").strip()
        attributes = {}
        if moved_match.group("from"):
            attributes["previous_location"] = moved_match.group("from").strip()
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

    work_match = re.search(
        r"\bI (?:work at|work for|joined|started at|just joined) (?P<company>[A-Z][A-Za-z0-9&.\s-]+)",
        text,
    )
    if work_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="employment",
                value=work_match.group("company").strip().rstrip("."),
                evidence=text,
                confidence=0.85,
            )
        )

    pet_match = re.search(
        r"\b(?:my dog|dog named|walking) (?P<pet>[A-Z][A-Za-z-]+)\b",
        text,
    )
    if pet_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="pet",
                value=f"Dog named {pet_match.group('pet')}",
                evidence=text,
                confidence=0.8,
            )
        )

    return memories


def _extract_creative_preferences(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    lowered = text.lower()

    style_terms = {
        "raw": "raw documentary realism",
        "documentary": "raw documentary realism",
        "cinematic": "cinematic visuals",
        "fashion editorial": "fashion editorial visuals",
        "editorial": "editorial visuals",
        "hyperreal": "hyperreal visuals",
        "anime": "anime-inspired visuals",
        "analog": "analog film texture",
        "surreal": "surreal visuals",
    }
    for term, value in style_terms.items():
        if term in lowered and not _is_negated(lowered, term):
            memories.append(
                ExtractedMemory(
                    memory_type="preference",
                    category="creative_style",
                    key="visual_style",
                    value=value,
                    evidence=text,
                    confidence=0.78,
                    attributes={"source_term": term},
                )
            )
            break

    negative_pairs = {
        "not glossy": "avoid glossy polish",
        "too glossy": "avoid glossy polish",
        "not cinematic": "avoid cinematic polish",
        "too artificial": "avoid artificial-looking results",
        "too cgi": "avoid CGI-looking results",
        "plastic skin": "avoid plastic skin",
        "over-smoothed": "avoid over-smoothed faces and skin",
        "extra fingers": "avoid hand/finger artifacts",
        "no text": "avoid text artifacts",
        "avoid fast camera": "avoid fast camera motion",
    }
    for needle, value in negative_pairs.items():
        if needle in lowered:
            memories.append(
                ExtractedMemory(
                    memory_type="constraint",
                    category="negative_constraints",
                    key=_slug(value),
                    value=value,
                    evidence=text,
                    confidence=0.82,
                )
            )

    camera_terms = {
        "close-up": "close-up framing",
        "close up": "close-up framing",
        "wide shot": "wide shot framing",
        "dolly": "dolly camera movement",
        "handheld": "handheld camera feel",
        "static camera": "static camera",
        "top-down": "top-down camera angle",
    }
    for term, value in camera_terms.items():
        if term in lowered:
            memories.append(
                ExtractedMemory(
                    memory_type="preference",
                    category="camera_language",
                    key="camera_direction",
                    value=value,
                    evidence=text,
                    confidence=0.76,
                )
            )

    motion_terms = {
        "slow-motion": "slow-motion movement",
        "slow motion": "slow-motion movement",
        "subtle motion": "subtle motion",
        "high-energy": "high-energy movement",
        "fast-paced": "fast-paced movement",
    }
    for term, value in motion_terms.items():
        if term in lowered:
            memories.append(
                ExtractedMemory(
                    memory_type="preference",
                    category="motion_language",
                    key="motion_style",
                    value=value,
                    evidence=text,
                    confidence=0.76,
                )
            )

    lighting_terms = {
        "golden hour": "golden hour lighting",
        "soft daylight": "soft daylight",
        "neon": "neon lighting",
        "moody": "moody contrast",
        "soft contrast": "soft contrast",
    }
    for term, value in lighting_terms.items():
        if term in lowered:
            memories.append(
                ExtractedMemory(
                    memory_type="preference",
                    category="lighting",
                    key="lighting_style",
                    value=value,
                    evidence=text,
                    confidence=0.74,
                )
            )

    return memories


def _extract_generation_feedback(text: str) -> list[ExtractedMemory]:
    lowered = text.lower()
    memories: list[ExtractedMemory] = []

    if any(term in lowered for term in ("last one", "previous one", "that version")):
        if any(term in lowered for term in ("great", "worked", "love", "liked")):
            memories.append(
                ExtractedMemory(
                    memory_type="generation_feedback",
                    category="generation_feedback",
                    key="prior_success",
                    value=text,
                    evidence=text,
                    confidence=0.72,
                    attributes={"sentiment": "positive"},
                )
            )
        if any(term in lowered for term in ("bad", "rejected", "too", "wrong", "failed")):
            memories.append(
                ExtractedMemory(
                    memory_type="generation_feedback",
                    category="generation_feedback",
                    key="prior_failure",
                    value=text,
                    evidence=text,
                    confidence=0.72,
                    attributes={"sentiment": "negative"},
                )
            )

    return memories


def _is_negated(text: str, term: str) -> bool:
    return (
        f"not {term}" in text
        or f"avoid {term}" in text
        or f"too {term}" in text
        or f"away from {term}" in text
        or f"move away from {term}" in text
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _dedupe(memories: list[ExtractedMemory]) -> list[ExtractedMemory]:
    seen: set[tuple[str, str, str, str]] = set()
    unique: list[ExtractedMemory] = []
    for memory in memories:
        fingerprint = (
            memory.memory_type,
            memory.category,
            memory.key,
            memory.value.lower(),
        )
        if fingerprint not in seen:
            seen.add(fingerprint)
            unique.append(memory)
    return unique
