from datetime import datetime, timezone

from job_scraper.filtering.remote_filter import compute_is_remote
from job_scraper.models import JobPosting


def _posting(location=None, extra=None):
    return JobPosting(
        source="test",
        company="Acme",
        title="Bioinformatics Scientist",
        location=location,
        description="",
        url="https://example.com/1",
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
        extra=extra or {},
    )


def test_jobspy_is_remote_true_wins_over_location_text():
    posting = _posting(location="New York, NY", extra={"is_remote": True})
    assert compute_is_remote(posting) == 1


def test_jobspy_is_remote_false_wins_over_location_text():
    posting = _posting(location="Remote", extra={"is_remote": False})
    assert compute_is_remote(posting) == 0


def test_location_text_matches_remote_word_boundary():
    assert compute_is_remote(_posting(location="Remote - US")) == 1
    assert compute_is_remote(_posting(location="Remote")) == 1


def test_remote_word_boundary_does_not_match_substring():
    assert compute_is_remote(_posting(location="Remotely, TX")) == 0


def test_hybrid_location_counts_as_onsite():
    assert compute_is_remote(_posting(location="Hybrid - Boston, MA")) == 0


def test_hybrid_takes_precedence_over_remote_in_same_location_string():
    assert compute_is_remote(_posting(location="Hybrid/Remote optional")) == 0


def test_plain_location_counts_as_onsite():
    assert compute_is_remote(_posting(location="Boston, MA")) == 0


def test_empty_or_missing_location_is_unknown():
    assert compute_is_remote(_posting(location=None)) is None
    assert compute_is_remote(_posting(location="")) is None
    assert compute_is_remote(_posting(location="   ")) is None
