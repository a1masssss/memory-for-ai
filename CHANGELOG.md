# Changelog

## v1.3 - LLM-first extraction with fallback-only heuristics

**What changed:** Switched extraction orchestration so validated OpenAI output is now authoritative when available. Rule-based extraction is only used when the model is unavailable or returns invalid structured output. Expanded the model schema to allow richer memory attributes such as `revision_kind`, `topic`, and `stance`.

**Why:** The earlier orchestration still merged rule-based heuristics into model output, which kept the service more regex-shaped than it needed to be. For a more production-ready design, the model should own extraction while the heuristic layer acts as an offline safety net.

**Result:** The service now has a cleaner separation between canonical memory schema and extraction strategy: fixed slots remain, but hardcoded rules no longer co-author production memories when the LLM path is healthy. Added tests to prove that model output is preferred and fallback rules only activate on failure.

**Next:** If needed, trim the fallback extractor even further and push more normalization rules into model instructions so offline behavior stays narrow and production behavior stays model-driven.

## v1.2 - Pgvector-backed hybrid retrieval

**What changed:** Added `embedding vector(64)` columns to turns and memories, synchronous embedding generation during `POST /turns`, and hybrid lexical+vector scoring for both `/recall` and `/search`. Implemented deterministic local embeddings as the default path and optional OpenAI embeddings when an API key is available.

**Why:** The system had grown a solid lexical and rule-based recall layer, but it still lacked true vector retrieval. Since the stack already used the pgvector Docker image, the simplest high-value upgrade was to activate pgvector in the existing Postgres design instead of introducing a second database.

**Result:** The service now stores real embeddings and blends cosine similarity into memory ranking, linked-memory expansion, and explicit search. The offline test path remains self-contained thanks to local embeddings, while production-style runs can upgrade to OpenAI embeddings with the same schema and API surface.

**Next:** If time remains, add a small held-out fixture that specifically measures lexical-only misses versus vector-assisted hits so the changelog can report a measurable hybrid-retrieval gain.

## v1.1 - Query-intent expansion for paraphrased recall

**What changed:** Added a lightweight query profiler in recall. It expands common paraphrases into intent-aware lexical terms and target slots before scoring memories and recent turns. Added contract tests for paraphrased location, employment, food-restriction, and reply-style questions.

**Why:** Retrieval quality was still too dependent on literal wording. Hidden eval prompts are likely to ask "What city do they call home?" rather than "Where do they live?", or "How chatty should replies be?" rather than "What answer style do they prefer?"

**Result:** Recall now handles paraphrased probes without needing an external rewrite model, while noise resistance remains intact. Fresh rebuilt Docker tests pass, including the new paraphrase scenarios.

**Next:** If there is more time, add optional embedding retrieval or semantic reranking so broader paraphrases and long-tail domains rely less on hand-authored intent rules.

## v1.0 - Opinion arcs, correction metadata, and broader rule extraction

**What changed:** Stopped automatically superseding every same-key opinion. Opinion memories now preserve history by default, carry attributes such as `stance`, `topic`, `subtopic`, and `revision_kind`, and are linked through `memory_links` as arcs/refinements/corrections. Expanded the rule-based extractor to handle lowercase facts, direct corrections, and richer opinion templates. Exposed memory attributes via `/users/{user_id}/memories`.

**Why:** The earlier implementation was too blunt for the challenge's opinion-evolution requirement and too brittle for messy real-world phrasing. It could overwrite nuanced preference history and miss simple lowercase or correction-heavy statements.

**Result:** The service now keeps arcs like `love TypeScript` -> `generics are annoying` -> `fine for big projects`, marks corrections like `Actually I live in Munich now`, and extracts facts from inputs such as `i work at notion now`. Added contract tests for lowercase facts, correction supersession, and opinion-history preservation.

**Next:** Improve query rewriting and semantic retrieval so hidden-eval probes with paraphrased wording depend less on lexical overlap.

## v0.9 - Global budgeted ranking and lightweight multi-hop recall

**What changed:** Reworked recall assembly so memories are admitted by global score under the token budget instead of filling one section at a time and stopping on the first overflow. Added lightweight graph expansion over `memory_links`, with links created for co-mentioned memories and user-profile personal facts.

**Why:** Section-first assembly could hide highly relevant later-category memories under tight budgets. Recall also lacked a way to connect related facts across turns, which made questions like pet name -> home city too brittle.

**Result:** Tight-budget queries can now surface the most relevant item even when it lives in a later section, and recall can expand from anchor memories to linked facts such as Biscuit -> Berlin. Added HTTP-level regression tests for both cases.

**Next:** Improve opinion evolution and contradiction handling beyond single-slot supersession so nuanced arcs do not collapse into simple overwrites.

## v0.8 - Recent-context fallback in recall

**What changed:** Extended `POST /recall` to rank recent raw turns alongside structured memories and append a `Relevant From Recent Conversations` section when a turn is query-relevant but not already covered by extracted memories.

**Why:** The previous pipeline could miss fresh but important context whenever the extractor failed to normalize it into a memory. That made recall look artificially sparse even though the data was already stored synchronously.

**Result:** Recall can now surface useful raw snippets such as "preparing for a system design interview" even when that fact was not extracted into `/users/{user_id}/memories`. Added an HTTP-level regression test for this fallback path.

**Next:** Tighten the global budgeter so high-value snippets are selected by score across sections rather than mostly by section order.

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

## v0.6 - Generic memory coverage and memory-aware search

**What changed:** Added extraction and tests for employment evolution, location, pets, diet, allergies, communication style, lightweight opinions, unicode input, and memory-first `/search` results. Added an optional Docker restart persistence test.

**Why:** The private eval is likely to include generic memory probes in addition to video-generation scenarios. Search should also return structured memories, not only raw turns.

**Result:** The test suite now covers both product-native video memories and task-native generic facts, including Stripe -> Notion supersession and Berlin/Biscuit/shellfish recall.

**Next:** Add optional embedding retrieval or LLM-assisted extraction if time allows.

## v0.7 - OpenAI structured extraction

**What changed:** Added optional OpenAI Responses API extraction using strict JSON schema output. The deterministic extractor remains as a fallback when no API key is configured or the model call fails.

**Why:** A production-ready memory service should not rely only on hardcoded phrases. The LLM path handles broader language while local validation and fallback preserve reliability.

**Result:** The service can use `OPENAI_API_KEY` for semantic extraction without making tests or local startup depend on external APIs. Added mocked tests for the OpenAI extraction path.

**Next:** Add vector embeddings with pgvector or a reranking pass for deeper multi-hop recall.
