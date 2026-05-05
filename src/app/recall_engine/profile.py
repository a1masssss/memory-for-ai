from __future__ import annotations

from .intent import QUERY_INTENT_RULES
from .models import QueryProfile
from .text import _terms


def _build_query_profile(query: str) -> QueryProfile:
    expanded_terms: list[str] = []
    seen_terms: set[str] = set()
    target_keys: set[tuple[str, str]] = set()
    target_categories: set[str] = set()

    for rule in QUERY_INTENT_RULES:
        if not rule["pattern"].search(query):
            continue
        for term in rule["terms"]:
            if term not in seen_terms:
                seen_terms.add(term)
                expanded_terms.append(term)
        target_keys.update(rule["keys"])
        target_categories.update(rule["categories"])

    retrieval_query = query
    if expanded_terms:
        retrieval_query = f"{query} {' '.join(expanded_terms)}"

    return QueryProfile(
        raw_query=query,
        retrieval_query=retrieval_query,
        overlap_terms=_terms(retrieval_query),
        target_keys=target_keys,
        target_categories=target_categories,
    )


def _focus_boost(
    *,
    category: str,
    key: str,
    query_profile: QueryProfile,
) -> float:
    if (category, key) in query_profile.target_keys:
        return 0.55
    if category in query_profile.target_categories:
        return 0.2
    return 0.0
