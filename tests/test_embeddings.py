from __future__ import annotations

import asyncio

from app.config import get_settings
from app.embeddings import embed_texts, vector_literal


def test_local_embeddings_are_deterministic_and_sized(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_EMBEDDING_ENABLED", "false")
    get_settings.cache_clear()

    first = asyncio.run(embed_texts(["Berlin memory", "Berlin memory"]))
    second = asyncio.run(embed_texts(["Berlin memory"]))

    assert len(first) == 2
    assert len(first[0]) == 64
    assert first[0] == first[1]
    assert first[0] == second[0]
    assert abs(sum(value * value for value in first[0]) - 1.0) < 1e-6


def test_vector_literal_formats_pgvector_input() -> None:
    literal = vector_literal([0.5, -0.25, 0.0])

    assert literal == "[0.50000000,-0.25000000,0.00000000]"


def test_embeddings_normalize_whitespace_consistently(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_EMBEDDING_ENABLED", "false")
    get_settings.cache_clear()

    compact = asyncio.run(embed_texts(["Berlin memory"]))
    spaced = asyncio.run(embed_texts(["  Berlin   memory \n\n"]))

    assert compact[0] == spaced[0]


def test_empty_text_embedding_is_zero_vector(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_EMBEDDING_ENABLED", "false")
    get_settings.cache_clear()

    embedding = asyncio.run(embed_texts(["   "]))[0]

    assert len(embedding) == 64
    assert all(value == 0.0 for value in embedding)
