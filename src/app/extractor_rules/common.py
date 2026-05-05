from __future__ import annotations

import re

from ..memory_types import ExtractedMemory


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


def _normalize_role(value: str) -> str:
    cleaned = _clean_capture(value)
    cleaned = re.sub(r"^(?:a|an|the)\s+", "", cleaned, flags=re.IGNORECASE)
    lower = cleaned.lower()
    if lower in {"pm", "swe", "ml", "ai"}:
        return lower.upper()
    return cleaned.title() if cleaned.islower() else cleaned


def _looks_like_role(value: str) -> bool:
    lowered = _clean_capture(value).lower()
    if not lowered:
        return False
    if lowered.startswith(("preparing", "studying", "getting ready", "planning")):
        return False
    return " for " not in lowered


def _looks_like_company(value: str) -> bool:
    lowered = _clean_capture(value).lower()
    if not lowered:
        return False
    time_words = {"next", "tomorrow", "today", "month", "week", "summer", "winter"}
    if time_words & set(lowered.split()):
        return False
    return not lowered.startswith(("a ", "an "))


def _normalize_person_name(value: str) -> str:
    cleaned = _clean_capture(value)
    return cleaned[:1].upper() + cleaned[1:] if cleaned else cleaned


def _normalize_relationship(value: str) -> str:
    relation = _clean_capture(value).lower()
    return "child" if relation == "kid" else relation


def _pluralize_food(value: str) -> str:
    cleaned = re.sub(r"^(?:severe|mild|bad)\s+", "", _clean_capture(value).lower())
    if cleaned in {"peanut", "tree nut", "nut"}:
        return f"{cleaned}s"
    return cleaned


def _sentence_fragment(value: str) -> str:
    return _clean_capture(value)


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
