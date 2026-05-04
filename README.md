# Memory Service

Dockerized memory service for an AI agent, optimized around video-generation workflows: creative preferences, prompt style, negative constraints, prior generation feedback, and continuity cues.

## Architecture

```text
HTTP client / eval harness
        |
        v
FastAPI app
  |-- Pydantic contract schemas
  |-- synchronous turn ingestion
  |-- rule-based extractor
  |-- recall ranker + context assembler
        |
        v
Postgres + pgvector image
  |-- raw turns
  |-- structured memories
  |-- tsvector full-text indexes
  |-- JSONB attributes
  |-- supersession links
```

The service is a small FastAPI monolith backed by Postgres. `POST /turns` stores the raw conversation turn and extracts structured memories in the same database transaction, so memories are immediately available to `/recall` and `/users/{user_id}/memories` when the endpoint returns.

The product angle is video generation rather than generic chatbot memory. The extractor promotes creative memory types such as visual style, camera language, motion preference, lighting, negative prompt constraints, and prior generation feedback.

## Backing Store

The backing store is Postgres using the `pgvector/pgvector:pg16` Docker image. The current implementation uses Postgres tables, JSONB, foreign keys, GIN indexes, and full-text search. The image includes pgvector so embeddings can be added without changing infrastructure.

This choice keeps the deployment simple while still supporting the challenge requirements: persistence via a named Docker volume, structured inspectable memories, lexical retrieval, contradiction history, and future vector retrieval.

## Extraction Pipeline

Extraction is currently local and rule-based. It reads user messages from each turn and emits structured memories with:

- `memory_type`: fact, preference, constraint, generation_feedback, etc.
- `category`: personal_context, creative_style, camera_language, motion_language, lighting, negative_constraints.
- `key` and `value`: normalized memory slot and human-readable value.
- `evidence`: original user text.
- `confidence`: simple rule confidence.
- `attributes`: JSONB metadata for future richer extraction.

Examples:

- "raw and documentary, not glossy" becomes `creative_style/visual_style = raw documentary realism` and `negative_constraints/avoid_glossy_polish = avoid glossy polish`.
- "slow-motion close-ups and soft daylight" becomes motion, camera, and lighting preferences.
- "I just moved to Berlin from NYC" becomes `personal_context/current_location = Berlin` with previous location in attributes.

Current limitations: the extractor is deliberately conservative and pattern-based. It misses many implicit facts and nuanced opinions that an LLM-assisted extractor could capture.

## Recall Strategy

`POST /recall` fetches active memories scoped to the same session or same user. It ranks candidates with a hybrid score:

- Postgres full-text lexical score via `ts_rank_cd`.
- Keyword overlap between query and memory text.
- Memory type boost for facts, preferences, and constraints.
- Same-session boost.
- Stable-memory boost for user facts and creative style.

The response is assembled into prompt-readable sections:

- `Known Facts About This User`
- `Stable Creative Profile`
- `Negative Prompt Memory`
- `Camera And Composition Preferences`
- `Motion Preferences`
- `Lighting Preferences`
- `Relevant Prior Generation Feedback`

The token budget is approximate: the assembler estimates tokens from word count and stops adding sections before exceeding `max_tokens`.

## Fact Evolution

Mutable memory slots are superseded instead of overwritten. For example:

- First turn: `creative_style/visual_style = cinematic visuals`
- Later turn: `creative_style/visual_style = raw documentary realism`

The older memory is marked `active=false`, the newer memory remains active, and the newer row stores `supersedes=<old_id>`. `/recall` only uses active memories, while `/users/{user_id}/memories` preserves the history for inspection.

Currently supported mutable keys include current location, employment, visual style, camera direction, motion style, and lighting style.

## Tradeoffs

The implementation optimizes for contract correctness, inspectability, and a strong video-generation memory story under a short time box. It avoids async job orchestration so `/turns` is synchronously correct.

The main tradeoff is extraction breadth. A rule-based extractor is predictable and easy to test, but less capable than an LLM extractor for subtle preferences, implicit facts, and gradual opinion arcs.

## Failure Modes

- Empty store: `/recall` returns `{"context": "", "citations": []}`.
- Malformed input: FastAPI/Pydantic returns 4xx validation errors.
- Missing API keys: no API keys are required for the current local extractor.
- Slow or unavailable Postgres: startup fails health readiness, and requests fail rather than returning stale data.
- Oversized context budget: `max_tokens` is bounded by request validation.

## Running

```bash
docker compose up --build
```

The API listens on port `8080`.

```bash
curl -s http://localhost:8080/health
```

## Tests

Run the black-box contract and recall-quality tests against the Compose stack:

```bash
docker compose run --rm test
```

The recall fixture lives in `fixtures/recall_quality.json` and covers style evolution, prompt DNA, and negative generation feedback.
