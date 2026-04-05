CREATE TABLE IF NOT EXISTS market_analysis_cache (
    id                       INTEGER PRIMARY KEY AUTOINCREMENT,
    cache_key                TEXT    NOT NULL UNIQUE,
    job_titles               TEXT    NOT NULL,   -- JSON array
    country                  TEXT    NOT NULL,
    analysis_date            TEXT    NOT NULL,   -- ISO date string YYYY-MM-DD
    raw_job_postings         TEXT,               -- JSON
    extracted_requirements   TEXT,               -- JSON
    market_analysis_markdown TEXT,
    total_posts              INTEGER,
    created_at               TEXT    DEFAULT (datetime('now')),
    expires_at               TEXT                -- datetime string
);

CREATE INDEX IF NOT EXISTS idx_cache_key ON market_analysis_cache(cache_key);
CREATE INDEX IF NOT EXISTS idx_expires_at ON market_analysis_cache(expires_at);

CREATE TABLE IF NOT EXISTS sessions (
    session_id      TEXT PRIMARY KEY,
    created_at      TEXT DEFAULT (datetime('now')),
    last_active     TEXT,
    resume_bytes    BLOB,
    resume_filename TEXT
);

CREATE TABLE IF NOT EXISTS conversation_history (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT    NOT NULL REFERENCES sessions(session_id),
    role        TEXT    NOT NULL CHECK(role IN ('human', 'ai', 'tool')),
    content     TEXT    NOT NULL,
    metadata    TEXT,   -- JSON
    created_at  TEXT    DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_conv_session ON conversation_history(session_id, created_at);
