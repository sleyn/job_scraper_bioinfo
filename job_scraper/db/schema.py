import sqlite3

SCHEMA = """
CREATE TABLE IF NOT EXISTS job_postings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    url             TEXT NOT NULL UNIQUE,
    raw_id          TEXT,
    source          TEXT NOT NULL,
    company         TEXT NOT NULL,
    title           TEXT NOT NULL,
    location        TEXT,
    description     TEXT,
    posted_date     DATE,
    first_seen_at   TIMESTAMP NOT NULL,
    last_seen_at    TIMESTAMP NOT NULL,
    is_relevant     INTEGER NOT NULL,
    extra_json      TEXT,
    score           REAL
);

CREATE INDEX IF NOT EXISTS idx_job_postings_company ON job_postings(company);
CREATE INDEX IF NOT EXISTS idx_job_postings_source ON job_postings(source);
CREATE INDEX IF NOT EXISTS idx_job_postings_first_seen ON job_postings(first_seen_at);
CREATE INDEX IF NOT EXISTS idx_job_postings_is_relevant ON job_postings(is_relevant);

CREATE TABLE IF NOT EXISTS scrape_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at      TIMESTAMP NOT NULL,
    finished_at     TIMESTAMP,
    source          TEXT NOT NULL,
    fetched_count   INTEGER,
    new_count       INTEGER,
    updated_count   INTEGER,
    error           TEXT
);
"""


def _ensure_score_column(conn: sqlite3.Connection) -> None:
    """Migration for DBs created before the `score` column existed.
    CREATE TABLE IF NOT EXISTS won't add it to an already-existing table."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(job_postings)")}
    if "score" not in columns:
        conn.execute("ALTER TABLE job_postings ADD COLUMN score REAL")


def init_db(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _ensure_score_column(conn)
        conn.commit()
    finally:
        conn.close()


def get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn
