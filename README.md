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
  |-- OpenAI structured extractor, optional
  |-- rule-based fallback extractor
  |-- recall ranker + context assembler
        |
        v
Postgres + pgvector image
  |-- raw turns
  |-- structured memories
  |-- pgvector embeddings for hybrid retrieval
  |-- tsvector full-text indexes
  |-- JSONB attributes
  |-- supersession links
```

The service is a small FastAPI monolith backed by Postgres. `POST /turns` stores the raw conversation turn and extracts structured memories in the same database transaction, so memories are immediately available to `/recall` and `/users/{user_id}/memories` when the endpoint returns.

The product angle is video generation rather than generic chatbot memory. The extractor promotes creative memory types such as visual style, camera language, motion preference, lighting, negative prompt constraints, and prior generation feedback, while still covering generic user facts from the task such as location, employment, pets, diet, allergies, communication preferences, and lightweight opinions.

## Backing Store

The backing store is Postgres using the `pgvector/pgvector:pg16` Docker image. The current implementation uses Postgres tables, JSONB, foreign keys, GIN indexes, full-text search, and `vector(64)` embedding columns on both turns and memories.

This choice keeps the deployment simple while still supporting the challenge requirements: persistence via a named Docker volume, structured inspectable memories, lexical retrieval, contradiction history, and actual vector retrieval without adding another service.

## Extraction Pipeline

Extraction is LLM-first. When `OPENAI_API_KEY` is set and `OPENAI_EXTRACTION_ENABLED=true`, the service calls the OpenAI Responses API with a strict JSON schema and asks the model to return normalized memory objects. When no key is present, the model call fails, or the response is invalid, the service falls back to deterministic local extraction so the API remains usable in offline evaluation.

The production path trusts the model output after local validation instead of merging it with rule-based heuristics. The rule engine is now a safety net for offline operation and extraction failures, not a co-equal primary source.

Both paths emit structured memories with:

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
- "I work at Stripe" followed later by "I just joined Notion" supersedes the active employment memory.
- "walking Biscuit this morning" becomes `personal_context/pet = Dog named Biscuit`.
- "I am vegetarian and allergic to shellfish" becomes separate dietary and allergy memories.
- "i work at notion now, i'm based in berlin these days" is handled even with lowercase phrasing.
- "Actually I live in Munich now" is marked as a correction via attributes and supersedes the old location.
- "I love TypeScript" -> "TypeScript generics are getting annoying" -> "TypeScript is fine for big projects" is stored as an opinion arc with stance and revision metadata instead of collapsing to one row.

Embeddings are generated synchronously during `POST /turns`. By default the service uses deterministic local hashed embeddings so offline startup and tests remain self-contained. If `OPENAI_API_KEY` is present and `OPENAI_EMBEDDING_ENABLED=true`, it upgrades to OpenAI embeddings while keeping the same pgvector storage and retrieval path.

Current limitations: the fallback extractor is still deliberately pattern-based even though it now handles noisier phrasing, lowercase facts, explicit corrections, and several opinion-evolution templates. The OpenAI path is broader and now supports richer structured attributes, but it still validates model output against the local schema and drops malformed memories rather than trusting free-form text.

## Recall Strategy

`POST /recall` fetches active memories scoped to the same session or same user. Different users are isolated; sessions share memory only when they have the same non-null `user_id`. It ranks candidates with a hybrid score:

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
- `Relevant From Recent Conversations`

The assembler now blends two sources. It ranks structured memories first, then adds a recent-turn fallback section for query-relevant raw conversation snippets that were not captured as structured memories. This reduces recall misses when extraction is conservative or the user mentions something important only once.

Structured memories are now selected globally by score before rendering. The service no longer fills one section and then stops on the first budget overflow; instead, it keeps scanning the ranked list and admits whichever items still fit, then renders the selected items back into stable reviewer-friendly sections.

There is also a lightweight multi-hop pass. Query-matched memories act as anchors, and the service expands to linked active memories via `memory_links` when they were co-mentioned in the same turn or belong to the same user profile cluster of personal facts. This is intentionally modest rather than a full graph retriever, but it improves questions that require joining two memories such as pet name -> city.

Before scoring, the service builds a lightweight query profile for common paraphrases. It expands questions like "What city does this user call home?", "Any food restrictions I should keep in mind?", and "How chatty should my replies be?" into intent-aware lexical hints and target slots such as `current_location`, `dietary_preference`, `allergy`, and `answer_style`. This keeps the system deterministic while reducing dependence on exact wording.

Retrieval is now hybrid on two axes:

- lexical: Postgres full-text rank plus keyword overlap
- vector: cosine similarity over pgvector embeddings for both memories and turns

These scores are blended with the existing type boosts, same-session bias, stable-memory priority, and lightweight multi-hop expansion over memory links.

The token budget is approximate: the assembler estimates tokens from word count and admits items while they fit within `max_tokens`. Priority is stable facts first, then query-relevant structured memories, then recent raw context.

`POST /search` searches both structured memories and raw turns. Structured memory hits receive a small boost so agent tool calls prefer normalized facts when available, while still exposing raw turn text for auditability.

## Fact Evolution

Mutable memory slots are superseded instead of overwritten. For example:

- First turn: `creative_style/visual_style = cinematic visuals`
- Later turn: `creative_style/visual_style = raw documentary realism`

The older memory is marked `active=false`, the newer memory remains active, and the newer row stores `supersedes=<old_id>`. `/recall` only uses active memories, while `/users/{user_id}/memories` preserves the history for inspection.

For opinions, the service no longer treats every same-key update as a hard overwrite. Stable facts still use supersession, but opinion memories keep their history by default and receive link relations plus JSONB attributes such as `stance`, `topic`, `subtopic`, and `revision_kind`. Explicit opinion corrections can still supersede a prior value, while gradual changes are preserved as an inspectable arc.

`GET /users/{user_id}/memories` includes these attributes so reviewers can inspect why a memory was treated as a correction, conditional preference, or ordinary opinion update.

## Tradeoffs

The implementation optimizes for contract correctness, inspectability, and a strong video-generation memory story under a short time box. It avoids async job orchestration so `/turns` is synchronously correct.

The main tradeoff is extraction breadth. A rule-based extractor is predictable and easy to test, but less capable than an LLM extractor for subtle preferences, implicit facts, and gradual opinion arcs.

## Failure Modes

- Empty store: `/recall` returns `{"context": "", "citations": []}`.
- Malformed input: FastAPI/Pydantic returns 4xx validation errors.
- Missing API keys: the service uses deterministic fallback extraction.
- OpenAI timeout or malformed model output: the service logs a warning and falls back to local extraction.
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

Optional OpenAI extraction:

```bash
cp .env.example .env
# set OPENAI_API_KEY in .env
docker compose up --build
```

## Tests

Run the black-box contract and recall-quality tests against the Compose stack:

```bash
docker compose run --rm test
```

The recall fixture lives in `fixtures/recall_quality.json` and covers style evolution, prompt DNA, negative generation feedback, and generic fact evolution.

An optional restart-persistence test is included because it needs access to the host Docker daemon:

```bash
ENABLE_DOCKER_RESTART_TEST=1 pytest tests/test_restart_persistence.py
```
