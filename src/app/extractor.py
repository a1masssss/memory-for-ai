from __future__ import annotations

import re

from .config import get_settings
from .memory_types import ExtractedMemory
from .openai_extractor import extract_with_openai
from .schemas import Message


async def extract_memories(messages: list[Message]) -> list[ExtractedMemory]:
    rule_memories = extract_rule_based_memories(messages)
    settings = get_settings()
    if not settings.openai_extraction_enabled or not settings.openai_api_key:
        return rule_memories

    llm_memories = await extract_with_openai(
        messages=messages,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    return _dedupe(llm_memories + rule_memories)


def extract_rule_based_memories(messages: list[Message]) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    user_texts = [message.content for message in messages if message.role == "user"]

    for text in user_texts:
        memories.extend(_extract_personal_facts(text))
        memories.extend(_extract_creative_preferences(text))
        memories.extend(_extract_generation_feedback(text))

    return _dedupe(memories)


def _extract_personal_facts(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    lowered = text.lower()

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

    location_match = re.search(
        r"\b(?:i\s+)?(?:live in|am based in|i'm based in|currently live in|currently based in) (?P<city>[A-Z][A-Za-z\s.-]+?)(?: now| these days|\.|,|$)",
        text,
        flags=re.IGNORECASE,
    )
    if location_match and not moved_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="current_location",
                value=_clean_capture(location_match.group("city")),
                evidence=text,
                confidence=0.84,
            )
        )

    work_match = re.search(
        r"\bI (?:work at|work for|joined|started at|just joined) (?P<company>[A-Z][A-Za-z0-9&.\s-]+?)(?: as | now| recently| last|\.|,|$)",
        text,
    )
    if work_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="employment",
                value=_clean_capture(work_match.group("company")),
                evidence=text,
                confidence=0.85,
            )
        )

    pet_match = re.search(
        r"\b(?:my dog(?: is named)?|dog named|walking) (?P<pet>[A-Z][A-Za-z-]+)\b",
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

    if "vegetarian" in lowered:
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="personal_context",
                key="dietary_preference",
                value="Vegetarian",
                evidence=text,
                confidence=0.82,
            )
        )

    allergy_match = re.search(
        r"\b(?:allergic to|allergy to) (?P<allergy>[a-zA-Z ,&-]+?)(?:\.|,| and I| but |$)",
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
            )
        )

    child_match = re.search(
        r"\b(?:my son|my daughter|my child|kid named|child named) (?P<name>[A-Z][A-Za-z-]+)\b",
        text,
    )
    if child_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="family",
                value=f"Has a child named {child_match.group('name')}",
                evidence=text,
                confidence=0.78,
            )
        )

    if "concise" in lowered or "direct answers" in lowered:
        memories.append(
            ExtractedMemory(
                memory_type="preference",
                category="communication",
                key="answer_style",
                value="Prefers concise, direct answers",
                evidence=text,
                confidence=0.78,
            )
        )

    memories.extend(_extract_opinions(text))
    return memories


def _extract_opinions(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    patterns = [
        (
            r"\bI (?:love|like) (?P<topic>[A-Z][A-Za-z0-9+#.\s-]+?)(?:\.|,| but |$)",
            "likes",
        ),
        (
            r"\bI (?:hate|dislike) (?P<topic>[A-Z][A-Za-z0-9+#.\s-]+?)(?:\.|,| but |$)",
            "dislikes",
        ),
        (
            r"\b(?P<topic>[A-Z][A-Za-z0-9+#.\s-]+?) is fine for (?P<context>[a-zA-Z0-9\s-]+?)(?:\.|,| but |$)",
            "conditional",
        ),
    ]
    for pattern, stance in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        topic = _clean_capture(match.group("topic"))
        if stance == "conditional":
            value = f"{topic} is fine for {_clean_capture(match.group('context')).lower()}"
        else:
            value = f"{stance.capitalize()} {topic}"
        memories.append(
            ExtractedMemory(
                memory_type="opinion",
                category="opinions",
                key=_slug(topic),
                value=value,
                evidence=text,
                confidence=0.7,
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


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _clean_capture(value: str) -> str:
    cleaned = re.sub(r"\s+", " ", value).strip()
    return cleaned.rstrip(" .,!?:;")


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
