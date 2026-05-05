from __future__ import annotations

from collections import defaultdict

import asyncpg

from .constants import (
    MAX_MEMORIES_PER_SECTION,
    QUERY_ALIASES,
    RECENT_CONTEXT_TITLE,
    SECTION_TITLES,
)
from .models import QueryProfile
from .text import _approx_tokens, _format_recent_turn, _terms


def _assemble_context(
    ranked_memories: list[tuple[float, asyncpg.Record]],
    ranked_turns: list[tuple[float, asyncpg.Record]],
    *,
    query_profile: QueryProfile,
    max_tokens: int,
) -> tuple[
    str,
    list[tuple[float, asyncpg.Record]],
    list[tuple[float, asyncpg.Record]],
]:
    grouped, used_memories, used_tokens = _select_ranked_memories(
        ranked_memories,
        budget=max(16, max_tokens),
    )
    lines = _render_memory_sections(grouped)
    used_turns: list[tuple[float, asyncpg.Record]] = []
    budget = max(16, max_tokens)

    used_turn_ids = {str(row["source_turn"]) for _, row in used_memories}
    recent_lines: list[str] = []
    for score, row in ranked_turns:
        if str(row["id"]) in used_turn_ids:
            continue
        if _turn_is_redundant_with_selected_memories(
            row["content_text"],
            used_memories=used_memories,
            query_profile=query_profile,
        ):
            continue
        turn_summary = _format_recent_turn(row["content_text"])
        item_cost = _approx_tokens(f"- {turn_summary}")
        if not recent_lines:
            item_cost += _approx_tokens(f"## {RECENT_CONTEXT_TITLE}") + 1
        if used_tokens + item_cost > budget:
            continue
        if not recent_lines:
            recent_lines.append(f"## {RECENT_CONTEXT_TITLE}")
        recent_lines.append(f"- {turn_summary}")
        used_turns.append((score, row))
        used_tokens += item_cost

    if recent_lines:
        lines.extend(recent_lines)
        lines.append("")

    return "\n".join(lines).strip(), used_memories, used_turns


def _select_ranked_memories(
    ranked_memories: list[tuple[float, asyncpg.Record]],
    *,
    budget: int,
) -> tuple[
    dict[str, list[tuple[float, asyncpg.Record]]],
    list[tuple[float, asyncpg.Record]],
    int,
]:
    grouped: dict[str, list[tuple[float, asyncpg.Record]]] = defaultdict(list)
    used_memories: list[tuple[float, asyncpg.Record]] = []
    used_tokens = 0

    for score, row in ranked_memories:
        category = str(row["category"])
        if len(grouped[category]) >= MAX_MEMORIES_PER_SECTION:
            continue

        title = SECTION_TITLES.get(category, category.replace("_", " ").title())
        item_cost = _approx_tokens(f"- {row['value']}")
        if not grouped[category]:
            item_cost += _approx_tokens(f"## {title}") + 1

        if used_tokens + item_cost > budget:
            continue

        grouped[category].append((score, row))
        used_memories.append((score, row))
        used_tokens += item_cost

    return grouped, used_memories, used_tokens


def _render_memory_sections(
    grouped: dict[str, list[tuple[float, asyncpg.Record]]],
) -> list[str]:
    lines: list[str] = []
    for category in _section_order(grouped):
        title = SECTION_TITLES.get(category, category.replace("_", " ").title())
        lines.append(f"## {title}")
        for _, row in grouped[category]:
            lines.append(f"- {row['value']}")
        lines.append("")
    return lines


def _section_order(grouped: dict[str, list[tuple[float, asyncpg.Record]]]) -> list[str]:
    preferred = [
        "personal_context",
        "creative_style",
        "negative_constraints",
        "camera_language",
        "motion_language",
        "lighting",
        "character_continuity",
        "communication",
        "opinions",
        "project_goal",
        "generation_feedback",
    ]
    remaining = [category for category in grouped if category not in preferred]
    return [category for category in preferred if category in grouped] + sorted(remaining)


def _turn_is_redundant_with_selected_memories(
    content_text: str,
    *,
    used_memories: list[tuple[float, asyncpg.Record]],
    query_profile: QueryProfile,
) -> bool:
    if not used_memories:
        return False

    turn_terms = _terms(content_text)
    if not turn_terms:
        return False

    for _, row in used_memories:
        category = str(row["category"])
        key = str(row["key"])
        if (
            (category, key) not in query_profile.target_keys
            and category not in query_profile.target_categories
        ):
            continue
        alias = QUERY_ALIASES.get((category, key), "")
        slot_terms = _terms(f"{category} {key} {alias}")
        if turn_terms & slot_terms:
            return True
    return False
