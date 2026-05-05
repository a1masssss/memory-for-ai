from __future__ import annotations

from ..memory_types import ExtractedMemory
from .common import _slug


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
        "away from cinematic": "avoid cinematic polish",
        "move away from cinematic": "avoid cinematic polish",
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


