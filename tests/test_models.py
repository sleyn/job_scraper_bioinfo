from datetime import datetime, timezone

from job_scraper.models import JobPosting


def test_job_posting_defaults():
    posting = JobPosting(
        source="greenhouse",
        company="Acme",
        title="Bioinformatics Scientist",
        location="Remote",
        description="...",
        url="https://example.com/jobs/1",
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
    )
    assert posting.raw_id is None
    assert posting.extra == {}
