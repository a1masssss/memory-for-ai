from __future__ import annotations

import re
from typing import Any

from .constants import STOPWORDS


def _keyword_overlap(query_terms_or_text: set[str] | str, document: str) -> float:
    query_terms = (
        query_terms_or_text
        if isinstance(query_terms_or_text, set)
        else _terms(query_terms_or_text)
    )
    if not query_terms:
        return 0.0
    document_terms = _terms(document)
    return len(query_terms & document_terms) / len(query_terms)


def _terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.lower())
        if len(term) > 2 and term not in STOPWORDS
    }


def _approx_tokens(text: str) -> int:
    return max(1, int(len(text.split()) * 1.25))


def _snippet(text: Any) -> str:
    value = str(text).strip()
    return value if len(value) <= 220 else value[:217].rstrip() + "..."


def _format_recent_turn(content_text: str) -> str:
    normalized = re.sub(r"\s+", " ", content_text).strip()
    return normalized if len(normalized) <= 280 else normalized[:277].rstrip() + "..."
