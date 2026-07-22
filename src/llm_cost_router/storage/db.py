import sqlite3
from contextlib import contextmanager
from pathlib import Path

from llm_cost_router import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS request_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt_hash TEXT NOT NULL,
    tier INTEGER,
    model_id TEXT,
    provider TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cost_usd REAL,
    latency_ms REAL,
    quality_score REAL,
    escalated INTEGER NOT NULL DEFAULT 0,
    escalated_model_id TEXT,
    cost_delta REAL,
    error TEXT
);

-- Deliberate, narrow exception to request_log's hash-only privacy policy:
-- only escalated (verified-bad) requests land here, and only because the
-- classifier feedback loop (Step 7) needs the actual prompt text to retrain
-- on. This is a much smaller, more consequential slice of traffic than
-- request_log's full firehose.
CREATE TABLE IF NOT EXISTS classifier_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    prompt TEXT NOT NULL,
    original_tier INTEGER NOT NULL,
    corrected_tier INTEGER NOT NULL,
    quality_score REAL NOT NULL
);
"""


def init_db(path: Path | None = None) -> None:
    path = path or settings.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        conn.executescript(SCHEMA)


@contextmanager
def get_connection(path: Path | None = None):
    path = path or settings.DB_PATH
    conn = sqlite3.connect(path)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()
