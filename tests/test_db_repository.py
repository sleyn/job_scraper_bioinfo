from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import (
    get_postings_missing_score,
    update_scores,
    upsert_postings,
)
from job_scraper.db.schema import get_connection, init_db
from job_scraper.models import JobPosting


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url, title="Bioinformatics Scientist", description="desc"):
    return JobPosting(
        source="greenhouse",
        company="Acme",
        title=title,
        location="Remote",
        description=description,
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


def test_newly_inserted_posting_has_no_score(db_path):
    posting = _posting("https://example.com/3")
    upsert_postings(db_path, [posting], {posting.url: True})

    pending = get_postings_missing_score(db_path)
    assert pending == {posting.url: posting.description}


def test_update_scores_clears_pending_and_persists_value(db_path):
    posting = _posting("https://example.com/4")
    upsert_postings(db_path, [posting], {posting.url: True})

    updated = update_scores(db_path, {posting.url: 0.75})
    assert updated == 1
    assert get_postings_missing_score(db_path) == {}

    conn = get_connection(db_path)
    row = conn.execute("SELECT score FROM job_postings WHERE url = ?", (posting.url,)).fetchone()
    conn.close()
    assert row["score"] == pytest.approx(0.75)


def test_rescrape_does_not_clear_existing_score(db_path):
    posting = _posting("https://example.com/5")
    upsert_postings(db_path, [posting], {posting.url: True})
    update_scores(db_path, {posting.url: 0.42})

    rescraped = _posting("https://example.com/5", title="Updated Title")
    upsert_postings(db_path, [rescraped], {rescraped.url: True})

    conn = get_connection(db_path)
    row = conn.execute("SELECT score FROM job_postings WHERE url = ?", (posting.url,)).fetchone()
    conn.close()
    assert row["score"] == pytest.approx(0.42)


def test_pending_score_query_skips_irrelevant_postings(db_path):
    relevant = _posting("https://example.com/6")
    irrelevant = _posting("https://example.com/7", title="Sales Development Rep")
    upsert_postings(
        db_path,
        [relevant, irrelevant],
        {relevant.url: True, irrelevant.url: False},
    )

    pending = get_postings_missing_score(db_path)
    assert pending == {relevant.url: relevant.description}


def test_pending_score_query_skips_postings_with_no_description(db_path):
    """An empty description embeds to an ordinary vector, so scoring one produces a
    confident-looking number that means nothing. Real runs hit this hard: every JobSpy
    LinkedIn posting arrives with no description."""
    postings = [
        _posting("https://example.com/described", description="Bioinformatics pipelines."),
        _posting("https://example.com/empty", description=""),
        _posting("https://example.com/whitespace", description="   \n  "),
        _posting("https://example.com/null", description=None),
    ]
    upsert_postings(db_path, postings, {p.url: True for p in postings})

    pending = get_postings_missing_score(db_path)

    assert set(pending) == {"https://example.com/described"}
