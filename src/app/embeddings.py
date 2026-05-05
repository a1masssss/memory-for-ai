from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import re
import urllib.error
import urllib.request

from .config import get_settings


logger = logging.getLogger(__name__)


async def embed_text(text: str) -> list[float]:
    embeddings = await embed_texts([text])
    return embeddings[0]


async def embed_texts(texts: list[str]) -> list[list[float]]:
    settings = get_settings()
    cleaned = [_clean_text(text) for text in texts]
    if (
        settings.openai_embedding_enabled
        and settings.openai_api_key
        and cleaned
    ):
        remote = await asyncio.to_thread(
            _embed_with_openai_sync,
            texts=cleaned,
            api_key=settings.openai_api_key,
            model=settings.openai_embedding_model,
            base_url=settings.openai_base_url,
            timeout_seconds=settings.openai_timeout_seconds,
            dimensions=settings.embedding_dimensions,
        )
        if remote is not None:
            return remote

    return [
        _local_embedding(text, dimensions=settings.embedding_dimensions)
        for text in cleaned
    ]


def vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def _embed_with_openai_sync(
    *,
    texts: list[str],
    api_key: str,
    model: str,
    base_url: str,
    timeout_seconds: float,
    dimensions: int,
) -> list[list[float]] | None:
    payload = {
        "model": model,
        "input": texts,
        "encoding_format": "float",
        "dimensions": dimensions,
    }
    request = urllib.request.Request(
        url=f"{base_url.rstrip('/')}/embeddings",
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
        logger.warning("OpenAI embedding request failed; falling back to local embeddings: %s", exc)
        return None

    data = body.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        return None

    embeddings: list[list[float]] = []
    for item in data:
        embedding = item.get("embedding") if isinstance(item, dict) else None
        if not isinstance(embedding, list):
            return None
        try:
            vector = [float(value) for value in embedding[:dimensions]]
        except (TypeError, ValueError):
            return None
        embeddings.append(_normalize(vector, dimensions=dimensions))
    return embeddings


def _local_embedding(text: str, *, dimensions: int) -> list[float]:
    vector = [0.0] * dimensions
    for token in _tokens(text):
        _accumulate(vector, f"tok:{token}", 1.0)
    for gram in _char_ngrams(text):
        _accumulate(vector, f"chr:{gram}", 0.35)
    return _normalize(vector, dimensions=dimensions)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _char_ngrams(text: str) -> list[str]:
    normalized = re.sub(r"\s+", " ", text.lower()).strip()
    if len(normalized) < 3:
        return [normalized] if normalized else []
    return [normalized[index : index + 3] for index in range(len(normalized) - 2)]


def _accumulate(vector: list[float], feature: str, weight: float) -> None:
    digest = hashlib.sha256(feature.encode("utf-8")).digest()
    index = int.from_bytes(digest[:4], "big") % len(vector)
    sign = 1.0 if digest[4] % 2 == 0 else -1.0
    vector[index] += weight * sign


def _normalize(values: list[float], *, dimensions: int) -> list[float]:
    if not values:
        return [0.0] * dimensions
    norm = math.sqrt(sum(value * value for value in values))
    if norm <= 0:
        return [0.0] * dimensions
    return [value / norm for value in values[:dimensions]]


def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:2000]
