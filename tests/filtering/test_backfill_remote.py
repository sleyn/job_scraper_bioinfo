from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import upsert_postings
from job_scraper.db.schema import get_connection, init_db
from job_scraper.filtering.backfill_remote import backfill_remote
from job_scraper.models import JobPosting


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url, location=None, extra=None):
    return JobPosting(
        source="greenhouse",
        company="Acme",
        title="Bioinformatics Scientist",
        location=location,
        description="desc",
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
        extra=extra or {},
    )


def test_backfill_remote_recomputes_stale_rows(db_path):
    remote = _posting("https://example.com/remote", location="Remote")
    onsite = _posting("https://example.com/onsite", location="Boston, MA")
    jobspy = _posting("https://example.com/jobspy", location="Boston, MA", extra={"is_remote": True})
    upsert_postings(
        db_path, [remote, onsite, jobspy],
        {remote.url: True, onsite.url: True, jobspy.url: True},
    )

    conn = get_connection(db_path)
    conn.execute("UPDATE job_postings SET is_remote = NULL")
    conn.commit()
    conn.close()

    updated = backfill_remote(db_path)

    conn = get_connection(db_path)
    rows = {
        row["url"]: row["is_remote"]
        for row in conn.execute("SELECT url, is_remote FROM job_postings").fetchall()
    }
    conn.close()
    assert updated == 3
    assert rows[remote.url] == 1
    assert rows[onsite.url] == 0
    assert rows[jobspy.url] == 1
