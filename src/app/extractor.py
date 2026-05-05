from __future__ import annotations

from .config import get_settings
from .extractor_rules import extract_rule_based_memories
from .extractor_rules.common import _dedupe
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
