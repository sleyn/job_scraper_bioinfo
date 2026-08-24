from unittest.mock import patch

from job_scraper.aggregators.backfill_linkedin_descriptions import backfill_linkedin_descriptions
from job_scraper.db.repository import get_postings_missing_score, upsert_postings
from job_scraper.db.schema import init_db
from job_scraper.models import JobPosting


def _linkedin_posting(url):
    from datetime import datetime, timezone

    return JobPosting(
        source="jobspy:linkedin",
        company="Acme",
        title="Bioinformatics Scientist",
        location="Remote",
        description="",
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
        raw_id=None,
    )


@patch("job_scraper.aggregators.backfill_linkedin_descriptions.fetch_linkedin_description")
@patch("job_scraper.aggregators.backfill_linkedin_descriptions.time.sleep")
def test_backfill_fetches_and_persists_missing_descriptions(
    mock_sleep, mock_fetch_description, config_dir, tmp_path
):
    db_path = str(tmp_path / "jobs.db")
    init_db(db_path)
    postings = [
        _linkedin_posting("https://linkedin.com/jobs/view/1"),
        _linkedin_posting("https://linkedin.com/jobs/view/2"),
    ]
    upsert_postings(db_path, postings, {p.url: True for p in postings})
    mock_fetch_description.side_effect = ["Description one.", "Description two."]

    stats = backfill_linkedin_descriptions(config_dir, db_path)

    assert stats.attempted_count == 2
    assert stats.updated_count == 2
    pending = get_postings_missing_score(db_path)
    assert pending == {
        "https://linkedin.com/jobs/view/1": "Description one.",
        "https://linkedin.com/jobs/view/2": "Description two.",
    }
    mock_sleep.assert_called_once()


@patch("job_scraper.aggregators.backfill_linkedin_descriptions.fetch_linkedin_description")
@patch("job_scraper.aggregators.backfill_linkedin_descriptions.time.sleep")
def test_backfill_skips_rows_with_no_description_returned(
    mock_sleep, mock_fetch_description, config_dir, tmp_path
):
    db_path = str(tmp_path / "jobs.db")
    init_db(db_path)
    posting = _linkedin_posting("https://linkedin.com/jobs/view/3")
    upsert_postings(db_path, [posting], {posting.url: True})
    mock_fetch_description.return_value = None

    stats = backfill_linkedin_descriptions(config_dir, db_path)

    assert stats.attempted_count == 1
    assert stats.updated_count == 0
    assert get_postings_missing_score(db_path) == {}


@patch("job_scraper.aggregators.backfill_linkedin_descriptions.fetch_linkedin_description")
def test_backfill_is_noop_when_nothing_missing(mock_fetch_description, config_dir, tmp_path):
    db_path = str(tmp_path / "jobs.db")
    init_db(db_path)

    stats = backfill_linkedin_descriptions(config_dir, db_path)

    assert stats.attempted_count == 0
    assert stats.updated_count == 0
    mock_fetch_description.assert_not_called()
