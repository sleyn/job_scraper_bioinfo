from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import upsert_postings
from job_scraper.db.schema import get_connection, init_db
from job_scraper.models import JobPosting


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url, title="Bioinformatics Scientist"):
    return JobPosting(
        source="greenhouse",
        company="Acme",
        title=title,
        location="Remote",
        description="desc",
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
        raw_id="123",
    )


def test_insert_new_posting(db_path):
    posting = _posting("https://example.com/1")
    stats = upsert_postings(db_path, [posting], {posting.url: True})

    assert stats.new_count == 1
    assert stats.updated_count == 0

    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM job_postings WHERE url = ?", (posting.url,)).fetchone()
    conn.close()
    assert row["is_relevant"] == 1
    assert row["first_seen_at"] == row["last_seen_at"]


def test_reinsert_same_url_updates_not_duplicates(db_path):
    posting = _posting("https://example.com/1")
    upsert_postings(db_path, [posting], {posting.url: True})

    updated_posting = _posting("https://example.com/1", title="Senior Bioinformatics Scientist")
    stats = upsert_postings(db_path, [updated_posting], {updated_posting.url: True})

    assert stats.new_count == 0
    assert stats.updated_count == 1

    conn = get_connection(db_path)
    rows = conn.execute("SELECT * FROM job_postings WHERE url = ?", (posting.url,)).fetchall()
    conn.close()
    assert len(rows) == 1
    assert rows[0]["title"] == "Senior Bioinformatics Scientist"
    assert rows[0]["first_seen_at"] != rows[0]["last_seen_at"]


def test_default_relevance_false_when_missing(db_path):
    posting = _posting("https://example.com/2")
    upsert_postings(db_path, [posting], {})

    conn = get_connection(db_path)
    row = conn.execute("SELECT * FROM job_postings WHERE url = ?", (posting.url,)).fetchone()
    conn.close()
    assert row["is_relevant"] == 0
