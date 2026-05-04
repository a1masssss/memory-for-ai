# Changelog

## v0.1 - Contract skeleton with Postgres

**What changed:** Added a FastAPI app, Pydantic request/response schemas, Postgres connection pooling, and all required HTTP endpoints.

**Why:** The private eval calls the HTTP contract directly, so the first milestone was exact endpoint shape and Docker-compatible startup.

**Result:** The service could start, initialize Postgres, accept turns, and expose empty recall/search scaffolding.

**Next:** Add structured extraction so `/users/{user_id}/memories` is not just raw message storage.

## v0.2 - Postgres schema and FTS indexing

**What changed:** Added `turns`, `memories`, and `memory_links` tables, JSONB attributes, full-text `tsvector` columns, GIN indexes, and pgvector-ready infrastructure.

**Why:** The challenge rewards real structured memory and non-trivial retrieval. Postgres gives us persistence, inspectable data, lexical search, and a clean path to vector search.

**Result:** Startup initially failed because generated `tsvector` expressions were not accepted as immutable. Replaced generated columns with trigger-maintained search vectors.

**Next:** Wire memory extraction into `POST /turns`.

## v0.3 - Rule-based video-generation extraction

**What changed:** Added local extraction for creative style, negative constraints, camera language, motion language, lighting, generation feedback, and some personal facts.

**Why:** Raw message chunks are not enough. A video-generation assistant benefits most from remembering prompt DNA: what visual style the user wants, what to avoid, and what worked or failed before.

**Result:** A smoke turn like "raw and documentary, not glossy, slow-motion close-ups" now produces structured memories such as `raw documentary realism`, `avoid glossy polish`, and `slow-motion movement`.

**Next:** Add contradiction handling and recall context assembly.

## v0.4 - Supersession and recall context

**What changed:** Added mutable-slot supersession and a recall assembler that ranks active memories and formats them into prompt-readable sections.

**Why:** The eval explicitly checks fact evolution. We need new facts/preferences to replace stale ones while preserving history for inspection.

**Result:** In a smoke test, `cinematic visuals` became inactive after the user shifted to `raw documentary realism`; `/recall` returned only the current style while `/users/{user_id}/memories` preserved both rows.

**Next:** Add automated tests and a recall-quality fixture.

## v0.5 - Black-box tests and recall-quality fixture

**What changed:** Added contract tests, malformed input checks, cross-user isolation checks, cold recall behavior, and a video-generation recall-quality fixture.

**Why:** The project needs an iteration loop that resembles the private eval. Black-box HTTP tests catch contract and behavior regressions better than only testing internal functions.

**Result:** `docker compose run --rm test` passes 6 tests, including fixture scenarios for style evolution, fashion prompt DNA, and negative generation feedback.

**Next:** Improve extraction breadth, add README detail, and consider optional LLM extraction or embeddings.
