from __future__ import annotations

import re

from ..memory_types import ExtractedMemory
from .common import (
    _looks_like_company,
    _looks_like_role,
    _normalize_company,
    _normalize_role,
    _revision_attributes,
)


def _extract_employment_memories(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    revision_attributes = _revision_attributes(text)

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

    memories.extend(_extract_employment_details(text, revision_attributes))

    return memories


def _extract_employment_details(
    text: str,
    revision_attributes: dict[str, str],
) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    terminator = r"(?=\s+now\b|\s+these days\b|\s+recently\b|[.!?,;]|$)"
    patterns = (
        re.compile(
            rf"\b(?:i'm|i am)\s+(?:an?|the)?\s*(?P<role>[A-Za-z][A-Za-z0-9&+/\s.'-]{{1,60}}?)\s+at\s+(?P<company>[A-Za-z0-9&.'\s-]+?){terminator}",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:i\s+)?work\s+as\s+(?:an?|the)?\s*(?P<role>[A-Za-z][A-Za-z0-9&+/\s.'-]{{1,60}}?)\s+(?:at|for)\s+(?P<company>[A-Za-z0-9&.'\s-]+?){terminator}",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:i\s+)?(?:work at|work for|joined|started at|just joined|now at)\s+(?P<company>[A-Za-z0-9&.'\s-]+?)\s+as\s+(?:an?|the)?\s*(?P<role>[A-Za-z][A-Za-z0-9&+/\s.'-]{{1,60}}?){terminator}",
            flags=re.IGNORECASE,
        ),
        re.compile(
            rf"\b(?:my employer is|employer is|new job at|started a new job at|took a role at)\s+(?P<company>[A-Za-z0-9&.'\s-]+?){terminator}",
            flags=re.IGNORECASE,
        ),
    )

    for pattern in patterns:
        match = pattern.search(text)
        if match is None:
            continue

        role = match.groupdict().get("role")
        if role and not _looks_like_role(role):
            continue

        company = _normalize_company(match.group("company"))
        if not _looks_like_company(company):
            continue

        if company:
            memories.append(
                ExtractedMemory(
                    memory_type="fact",
                    category="personal_context",
                    key="employment",
                    value=company,
                    evidence=text,
                    confidence=0.82,
                    attributes=dict(revision_attributes),
                )
            )

        if role:
            memories.append(
                ExtractedMemory(
                    memory_type="fact",
                    category="personal_context",
                    key="current_role",
                    value=_normalize_role(role),
                    evidence=text,
                    confidence=0.78,
                    attributes=dict(revision_attributes),
                )
            )
        break

    return memories
