from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, AsyncIterator

import asyncpg

from .config import get_settings


_pool: asyncpg.Pool | None = None


async def connect() -> None:
    global _pool
    if _pool is None:
        settings = get_settings()
        _pool = await asyncpg.create_pool(
            dsn=settings.database_url,
            min_size=1,
            max_size=10,
            command_timeout=30,
        )


async def close() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


def pool() -> asyncpg.Pool:
    if _pool is None:
        raise RuntimeError("Database pool is not initialized")
    return _pool


async def init_schema() -> None:
    schema_path = Path(__file__).with_name("schema.sql")
    statements = schema_path.read_text(encoding="utf-8")
    async with pool().acquire() as connection:
        await connection.execute(statements)


@asynccontextmanager
async def transaction() -> AsyncIterator[asyncpg.Connection]:
    async with pool().acquire() as connection:
        async with connection.transaction():
            yield connection


async def fetch(query: str, *args: Any) -> list[asyncpg.Record]:
    async with pool().acquire() as connection:
        return await connection.fetch(query, *args)


async def fetchrow(query: str, *args: Any) -> asyncpg.Record | None:
    async with pool().acquire() as connection:
        return await connection.fetchrow(query, *args)


async def execute(query: str, *args: Any) -> str:
    async with pool().acquire() as connection:
        return await connection.execute(query, *args)
