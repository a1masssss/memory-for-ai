# Memory Service

Small memory service for an AI agent. It stores conversation turns, extracts structured memories, keeps fact history, and returns compact recall context for later prompts.

## What It Does

- `POST /turns`: store a turn and extract memories synchronously
- `POST /recall`: build prompt-ready context from stored memory
- `POST /search`: search structured memories and raw turns
- `GET /users/{user_id}/memories`: inspect memory state and history
- `DELETE /users/{user_id}`: delete a user's stored data

## Architecture

```text
client
  -> FastAPI app
     -> turn storage
     -> memory extraction
     -> recall ranking + assembly
  -> Postgres + pgvector
```

Main pieces:

- FastAPI for the HTTP API
- Postgres as the only backing store
- `pgvector` for vector similarity
- full-text indexes for lexical retrieval
- synchronous ingestion so a new memory is immediately available to recall

## Why This Design

The service is intentionally a small monolith.

- One database keeps deployment simple.
- Postgres gives persistence, filtering, joins, JSONB, and full-text search.
- `pgvector` adds vector retrieval without introducing a second data system.
- Synchronous `POST /turns` avoids eventual-consistency surprises during evaluation.

This is not the most horizontally scalable design, but it is easy to inspect, test, and reason about.

## Memory Model

Each turn is stored raw, then converted into structured memories when possible.

Typical memory fields:

- `memory_type`: fact, preference, constraint, opinion, generation_feedback
- `category`: personal context, creative style, motion, lighting, negative constraints, etc.
- `key` / `value`: normalized slot plus readable value
- `evidence`: source text
- `attributes`: extra metadata

Fact updates are versioned, not overwritten:

- old memory becomes inactive
- new memory points to the previous one with `supersedes`

That keeps `/recall` clean while letting `/users/{user_id}/memories` show history.

## Extraction

Extraction is LLM-first with a deterministic fallback.

- If `OPENAI_API_KEY` is set and extraction is enabled, the service calls the OpenAI Responses API with a strict schema.
- If that is unavailable or invalid, it falls back to local rule-based extraction.

This choice keeps production extraction broad while still allowing offline tests and local runs.

## Recall

`POST /recall` combines:

- lexical search
- vector similarity
- type-based boosts
- session/user relevance
- lightweight memory links

The result is returned as short prompt-ready sections such as:

- known facts
- creative profile
- negative constraints
- recent relevant context

If extraction misses something important, recent raw turns can still be included as fallback context.

## Tradeoffs

- Strong on inspectability, correctness, and simple deployment
- Weaker than a fully learned memory system on subtle extraction edge cases
- Synchronous ingestion is simpler, but slower than an async pipeline under heavy load

## Running

Start the stack:

```bash
docker compose up --build
```

Health check:

```bash
curl -s http://localhost:8080/health
```

The API listens on `localhost:8080`.

## Auth

Optional bearer auth:

```bash
cp .env.example .env
# set MEMORY_AUTH_TOKEN in .env
docker compose up --build
```

`GET /health` stays open for health checks. Other endpoints require:

```bash
Authorization: Bearer <token>
```

## OpenAI Mode

To enable OpenAI extraction and embeddings:

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env
docker compose up --build
```

Without API keys, the service still works with deterministic local extraction and local embeddings.

## Tests

Run the test stack:

```bash
docker compose run --build --rm test
```

Optional restart persistence test:

```bash
ENABLE_DOCKER_RESTART_TEST=1 pytest tests/test_restart_persistence.py
```

## Notes

- Different users are isolated from each other.
- Sessions share memory only when they use the same non-null `user_id`.
- The recall quality fixture lives in `fixtures/recall_quality.json`.
