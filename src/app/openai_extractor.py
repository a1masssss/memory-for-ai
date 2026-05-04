from __future__ import annotations

import asyncio
import json
import logging
import urllib.error
import urllib.request
from typing import Any

from .memory_types import ExtractedMemory
from .schemas import Message, MemoryType


logger = logging.getLogger(__name__)

ALLOWED_MEMORY_TYPES: set[str] = {
    "fact",
    "preference",
    "opinion",
    "event",
    "constraint",
    "style_profile",
    "generation_feedback",
    "continuity_anchor",
}

MEMORY_EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "memories": {
            "type": "array",
            "maxItems": 24,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "memory_type": {
                        "type": "string",
                        "enum": sorted(ALLOWED_MEMORY_TYPES),
                    },
                    "category": {"type": "string"},
                    "key": {"type": "string"},
                    "value": {"type": "string"},
                    "evidence": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": [
                    "memory_type",
                    "category",
                    "key",
                    "value",
                    "evidence",
                    "confidence",
                ],
            },
        }
    },
    "required": ["memories"],
}

EXTRACTION_INSTRUCTIONS = """
You extract durable structured memories from completed conversation turns.

Return only facts/preferences/opinions/events/constraints that will help a future AI
agent. Do not store transient small talk. Prefer normalized, human-readable values.

Important domains:
- Generic user facts: employment, location, family, pets, diet, allergies.
- Preferences and opinions, including communication style.
- Corrections and fact evolution signals such as "actually", "not X, Y", "moved from X to Y".
- Video-generation memory: visual style, camera language, motion, lighting, negative constraints,
  prompt preferences, character/subject continuity, prior generation feedback.

Use stable categories such as:
personal_context, communication, opinions, creative_style, camera_language,
motion_language, lighting, negative_constraints, generation_feedback,
character_continuity, project_goal.

Use stable keys such as:
employment, current_location, pet, dietary_preference, allergy, family,
answer_style, visual_style, camera_direction, motion_style, lighting_style,
prior_success, prior_failure.

Evidence must be a short quote or paraphrase from the user message. Confidence must be 0 to 1.
If there are no durable memories, return {"memories": []}.
""".strip()


async def extract_with_openai(
    *,
    messages: list[Message],
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: float,
) -> list[ExtractedMemory]:
    return await asyncio.to_thread(
        _extract_with_openai_sync,
        messages=messages,
        api_key=api_key,
        model=model,
        base_url=base_url,
        timeout_seconds=timeout_seconds,
    )


def _extract_with_openai_sync(
    *,
    messages: list[Message],
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: float,
) -> list[ExtractedMemory]:
    payload = {
        "model": model,
        "instructions": EXTRACTION_INSTRUCTIONS,
        "input": [
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "messages": [
                            {
                                "role": message.role,
                                "name": message.name,
                                "content": message.content,
                            }
                            for message in messages
                        ]
                    },
                    ensure_ascii=False,
                ),
            }
        ],
        "text": {
            "format": {
                "type": "json_schema",
                "name": "memory_extraction",
                "strict": True,
                "schema": MEMORY_EXTRACTION_SCHEMA,
            }
        },
        "max_output_tokens": 2000,
        "store": False,
    }

    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/responses",
        data=json.dumps(payload).encode("utf-8"),
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except (TimeoutError, urllib.error.URLError, json.JSONDecodeError) as exc:
        logger.warning("OpenAI extraction failed; falling back to rules: %s", exc)
        return []

    output_text = _extract_output_text(body)
    if not output_text:
        return []

    try:
        parsed = json.loads(output_text)
    except json.JSONDecodeError as exc:
        logger.warning("OpenAI extraction returned invalid JSON: %s", exc)
        return []

    return _validated_memories(parsed)


def _extract_output_text(response_body: dict[str, Any]) -> str:
    if isinstance(response_body.get("output_text"), str):
        return response_body["output_text"]

    chunks: list[str] = []
    for item in response_body.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "".join(chunks).strip()


def _validated_memories(payload: Any) -> list[ExtractedMemory]:
    if not isinstance(payload, dict):
        return []
    raw_memories = payload.get("memories")
    if not isinstance(raw_memories, list):
        return []

    memories: list[ExtractedMemory] = []
    for raw in raw_memories[:24]:
        if not isinstance(raw, dict):
            continue
        memory_type = str(raw.get("memory_type", "")).strip()
        category = _clean_token(raw.get("category"))
        key = _clean_token(raw.get("key"))
        value = _clean_text(raw.get("value"))
        evidence = _clean_text(raw.get("evidence"))

        if memory_type not in ALLOWED_MEMORY_TYPES:
            continue
        if not category or not key or not value or not evidence:
            continue

        memories.append(
            ExtractedMemory(
                memory_type=memory_type,  # type: ignore[arg-type]
                category=category,
                key=key,
                value=value,
                evidence=evidence,
                confidence=_confidence(raw.get("confidence")),
                attributes={"source": "openai"},
            )
        )
    return memories


def _clean_token(value: Any) -> str:
    return str(value or "").strip().lower().replace(" ", "_")[:80]


def _clean_text(value: Any) -> str:
    return str(value or "").strip()[:500]


def _confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.7
    return max(0.0, min(1.0, confidence))
