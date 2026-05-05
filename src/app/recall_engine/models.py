from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class QueryProfile:
    raw_query: str
    retrieval_query: str
    overlap_terms: set[str]
    target_keys: set[tuple[str, str]]
    target_categories: set[str]
