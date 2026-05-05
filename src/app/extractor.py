from __future__ import annotations

import re

from .config import get_settings
from .memory_types import ExtractedMemory
from .openai_extractor import extract_with_openai
from .schemas import Message


async def extract_memories(messages: list[Message]) -> list[ExtractedMemory]:
    settings = get_settings()
    if not settings.openai_extraction_enabled or not settings.openai_api_key:
        return extract_rule_based_memories(messages)

    llm_memories = await extract_with_openai(
        messages=messages,
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url,
        timeout_seconds=settings.openai_timeout_seconds,
    )
    if llm_memories is not None:
        return _dedupe(llm_memories)

    return extract_rule_based_memories(messages)


def extract_rule_based_memories(messages: list[Message]) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    user_texts = [
        _normalize_text(message.content)
        for message in messages
        if message.role == "user"
    ]

    for text in user_texts:
        memories.extend(_extract_personal_facts(text))
        memories.extend(_extract_creative_preferences(text))
        memories.extend(_extract_generation_feedback(text))

    return _dedupe(memories)


def _extract_personal_facts(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    lowered = text.lower()
    revision_attributes = _revision_attributes(text)

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
        r"\b(?:i\s+)?(?:just\s+)?moved to (?P<city>[A-Za-z][A-Za-z0-9\s.'-]+?)(?: from (?P<from>[A-Za-z][A-Za-z0-9\s.'-]+?))?(?: last\b| now\b| these days\b| recently\b|\sand\b|[.!?,;]|$)",
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
        r"\b(?:actually\s+|sorry\s+)?(?:i\s+)?(?:live in|am based in|i'm based in|currently live in|currently based in)\s+(?P<city>[A-Za-z][A-Za-z0-9\s.'-]+?)(?: now\b| these days\b| currently\b|[.!?,;]|$)",
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

    corrected_work_match = re.search(
        r"\b(?:actually|sorry)[, ]+\s*not\s+[A-Za-z0-9&.'\s-]+?\s*-\s*(?:i\s+)?(?:work at|work for|joined|started at|just joined|now at)\s+(?P<company>[A-Za-z0-9&.'\s-]+?)(?:\s+as\b|\s+now\b|\s+recently\b|\s+these days\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if corrected_work_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="employment",
                value=_normalize_company(corrected_work_match.group("company")),
                evidence=text,
                confidence=0.86,
                attributes=dict(revision_attributes),
            )
        )

    work_match = re.search(
        r"\b(?:actually\s+|sorry\s+)?(?:i\s+)?(?:work at|work for|joined|started at|just joined|now at)\s+(?P<company>[A-Za-z0-9&.'\s-]+?)(?:\s+as\b|\s+now\b|\s+recently\b|\s+these days\b|[.!?,;]|$)",
        text,
        flags=re.IGNORECASE,
    )
    if work_match and not corrected_work_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="employment",
                value=_normalize_company(work_match.group("company")),
                evidence=text,
                confidence=0.85,
                attributes=dict(revision_attributes),
            )
        )

    pet_match = re.search(
        r"\b(?:my dog(?: is named)?|dog named|walking|walked|taking|took)\s+(?P<pet>[A-Za-z][A-Za-z'-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if pet_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="pet",
                value=f"Dog named {_normalize_person_name(pet_match.group('pet'))}",
                evidence=text,
                confidence=0.8,
            )
        )

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

    child_match = re.search(
        r"\b(?:my son|my daughter|my child|kid named|child named)\s+(?P<name>[A-Za-z][A-Za-z'-]+)\b",
        text,
        flags=re.IGNORECASE,
    )
    if child_match:
        memories.append(
            ExtractedMemory(
                memory_type="fact",
                category="personal_context",
                key="family",
                value=f"Has a child named {_normalize_person_name(child_match.group('name'))}",
                evidence=text,
                confidence=0.78,
            )
        )

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

    memories.extend(_extract_opinions(text))
    return memories


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


def _normalize_text(value: str) -> str:
    normalized = (
        value.replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("\u00a0", " ")
    )
    return re.sub(r"\s+", " ", normalized).strip()


def _normalize_location(value: str) -> str:
    cleaned = _clean_capture(value)
    return cleaned.title() if cleaned.islower() else cleaned


def _normalize_company(value: str) -> str:
    cleaned = _clean_capture(value)
    if cleaned.islower():
        return " ".join(part.capitalize() for part in cleaned.split())
    return cleaned


def _normalize_person_name(value: str) -> str:
    cleaned = _clean_capture(value)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _normalize_topic(value: str) -> str:
    cleaned = _clean_capture(value)
    cleaned = re.sub(r"\b(my|the|a|an)\b\s+", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def _split_clauses(text: str) -> list[str]:
    parts = re.split(r"(?i)\bbut\b|[.;!?]", text)
    return [_clean_capture(part) for part in parts if _clean_capture(part)]


def _revision_attributes(text: str) -> dict[str, str]:
    lowered = text.lower()
    if (
        "actually" in lowered
        or "sorry" in lowered
        or re.search(r"\bnot\b.+(?:-|,).+", lowered)
    ):
        return {"revision_kind": "correction"}
    if "fine for" in lowered:
        return {"revision_kind": "conditional"}
    if any(term in lowered for term in ("but i'd", "but i would", "however", "these days")):
        return {"revision_kind": "refinement"}
    return {}


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
