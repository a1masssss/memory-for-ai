from __future__ import annotations

from ..memory_types import ExtractedMemory
from .communication import _extract_communication_memories
from .dietary import _extract_dietary_memories
from .employment import _extract_employment_memories
from .identity import _extract_identity_memories
from .opinions import _extract_opinions
from .relationships import _extract_family_memories, _extract_pet_memories


def _extract_personal_facts(text: str) -> list[ExtractedMemory]:
    memories: list[ExtractedMemory] = []
    memories.extend(_extract_identity_memories(text))
    memories.extend(_extract_employment_memories(text))
    memories.extend(_extract_pet_memories(text))
    memories.extend(_extract_dietary_memories(text))
    memories.extend(_extract_family_memories(text))
    memories.extend(_extract_communication_memories(text))
    memories.extend(_extract_opinions(text))
    return memories
