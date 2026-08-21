from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone

from job_scraper.db.schema import get_connection
from job_scraper.models import JobPosting

UPSERT_SQL = """
INSERT INTO job_postings (
    url, raw_id, source, company, title, location, description,
    posted_date, first_seen_at, last_seen_at, is_relevant, extra_json
) VALUES (
    :url, :raw_id, :source, :company, :title, :location, :description,
    :posted_date, :now, :now, :is_relevant, :extra_json
)
ON CONFLICT(url) DO UPDATE SET
    last_seen_at = excluded.last_seen_at,
    title = excluded.title,
    description = excluded.description,
    location = excluded.location,
    is_relevant = excluded.is_relevant
"""


@dataclass
class RunStats:
    fetched_count: int
    new_count: int
    updated_count: int


def upsert_postings(
    db_path: str, postings: list[JobPosting], relevance: dict[str, bool]
) -> RunStats:
    """Upsert postings into the DB. `relevance` maps posting.url -> is_relevant,
    computed by the caller (pipeline.py) via the keyword filter before calling this."""
    now = datetime.now(timezone.utc).isoformat()
    new_count = 0
    updated_count = 0
    conn = get_connection(db_path)
    try:
        for posting in postings:
            existed = conn.execute(
                "SELECT 1 FROM job_postings WHERE url = ?", (posting.url,)
            ).fetchone()
            conn.execute(
                UPSERT_SQL,
                {
                    "url": posting.url,
                    "raw_id": posting.raw_id,
                    "source": posting.source,
                    "company": posting.company,
                    "title": posting.title,
                    "location": posting.location,
                    "description": posting.description,
                    "posted_date": posting.posted_date.isoformat()
                    if posting.posted_date
                    else None,
                    "now": now,
                    "is_relevant": int(relevance.get(posting.url, False)),
                    "extra_json": json.dumps(posting.extra),
                },
            )
            if existed:
                updated_count += 1
            else:
                new_count += 1
        conn.commit()
    finally:
        conn.close()
    return RunStats(
        fetched_count=len(postings), new_count=new_count, updated_count=updated_count
    )


def get_postings_missing_score(db_path: str) -> dict[str, str]:
    """Returns {url: description} for relevant postings that haven't been scored yet.

    Scoring is the expensive second stage: only postings that passed the cheap keyword
    pre-filter (is_relevant) are worth embedding, so the gating lives here rather than
    in the caller."""
    conn = get_connection(db_path)
    try:
        rows = conn.execute(
            "SELECT url, description FROM job_postings "
            "WHERE score IS NULL AND is_relevant = 1"
        ).fetchall()
        return {row["url"]: row["description"] or "" for row in rows}
    finally:
        conn.close()


def update_scores(db_path: str, scores: dict[str, float]) -> int:
    """Writes posting.url -> score. Returns the number of rows updated."""
    conn = get_connection(db_path)
    try:
        cursor = conn.executemany(
            "UPDATE job_postings SET score = :score WHERE url = :url",
            [{"url": url, "score": score} for url, score in scores.items()],
        )
        conn.commit()
        return cursor.rowcount
    finally:
        conn.close()


def record_run(
    db_path: str,
    source: str,
    started_at: datetime,
    finished_at: datetime | None,
    stats: RunStats | None,
    error: str | None,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO scrape_runs (
                started_at, finished_at, source, fetched_count,
                new_count, updated_count, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                started_at.isoformat(),
                finished_at.isoformat() if finished_at else None,
                source,
                stats.fetched_count if stats else None,
                stats.new_count if stats else None,
                stats.updated_count if stats else None,
                error,
            ),
        )
        conn.commit()
    finally:
        conn.close()
