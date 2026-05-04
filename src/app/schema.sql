CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE EXTENSION IF NOT EXISTS vector;

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'memory_kind') THEN
        CREATE TYPE memory_kind AS ENUM (
            'fact',
            'preference',
            'opinion',
            'event',
            'constraint',
            'style_profile',
            'generation_feedback',
            'continuity_anchor'
        );
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS turns (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id text NOT NULL,
    user_id text,
    messages jsonb NOT NULL,
    timestamp timestamptz NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    content_text text NOT NULL,
    search_vector tsvector NOT NULL DEFAULT ''::tsvector,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_turns_session ON turns (session_id);
CREATE INDEX IF NOT EXISTS idx_turns_user ON turns (user_id);
CREATE INDEX IF NOT EXISTS idx_turns_timestamp ON turns (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_turns_search ON turns USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_turns_metadata ON turns USING gin (metadata jsonb_path_ops);

CREATE TABLE IF NOT EXISTS memories (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id text,
    memory_type memory_kind NOT NULL,
    category text NOT NULL,
    key text NOT NULL,
    value text NOT NULL,
    attributes jsonb NOT NULL DEFAULT '{}'::jsonb,
    confidence numeric(3,2) NOT NULL DEFAULT 0.75,
    evidence text NOT NULL,
    source_session text NOT NULL,
    source_turn uuid NOT NULL REFERENCES turns(id) ON DELETE CASCADE,
    active boolean NOT NULL DEFAULT true,
    supersedes uuid REFERENCES memories(id) ON DELETE SET NULL,
    search_vector tsvector NOT NULL DEFAULT ''::tsvector,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories (user_id);
CREATE INDEX IF NOT EXISTS idx_memories_session ON memories (source_session);
CREATE INDEX IF NOT EXISTS idx_memories_active ON memories (active);
CREATE INDEX IF NOT EXISTS idx_memories_category_key ON memories (category, key);
CREATE INDEX IF NOT EXISTS idx_memories_search ON memories USING gin (search_vector);
CREATE INDEX IF NOT EXISTS idx_memories_attributes ON memories USING gin (attributes jsonb_path_ops);

CREATE TABLE IF NOT EXISTS memory_links (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source_memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    target_memory_id uuid NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    relation text NOT NULL,
    weight numeric(4,3) NOT NULL DEFAULT 1.0,
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (source_memory_id, target_memory_id, relation)
);

CREATE INDEX IF NOT EXISTS idx_memory_links_source ON memory_links (source_memory_id);
CREATE INDEX IF NOT EXISTS idx_memory_links_target ON memory_links (target_memory_id);

CREATE OR REPLACE FUNCTION update_turn_search_vector()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector('english', coalesce(NEW.content_text, ''));
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_turns_search_vector ON turns;
CREATE TRIGGER trg_turns_search_vector
BEFORE INSERT OR UPDATE OF content_text ON turns
FOR EACH ROW EXECUTE FUNCTION update_turn_search_vector();

CREATE OR REPLACE FUNCTION update_memory_search_vector()
RETURNS trigger AS $$
BEGIN
    NEW.search_vector := to_tsvector(
        'english',
        coalesce(NEW.memory_type::text, '') || ' ' ||
        coalesce(NEW.category, '') || ' ' ||
        coalesce(NEW.key, '') || ' ' ||
        coalesce(NEW.value, '') || ' ' ||
        coalesce(NEW.evidence, '')
    );
    NEW.updated_at := now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_memories_search_vector ON memories;
CREATE TRIGGER trg_memories_search_vector
BEFORE INSERT OR UPDATE OF memory_type, category, key, value, evidence ON memories
FOR EACH ROW EXECUTE FUNCTION update_memory_search_vector();
