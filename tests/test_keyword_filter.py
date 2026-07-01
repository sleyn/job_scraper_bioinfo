from datetime import datetime, timezone

import pytest

from job_scraper.config import KeywordConfig
import re

from job_scraper.filtering.keyword_filter import is_relevant
from job_scraper.models import JobPosting


def _posting(title="", description=""):
    return JobPosting(
        source="test",
        company="Acme",
        title=title,
        location=None,
        description=description,
        url="https://example.com/1",
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def cfg():
    return KeywordConfig(
        include_patterns=[re.compile(p, re.IGNORECASE) for p in ["bioinformatics", "NGS"]],
        exclude_patterns=[re.compile(p, re.IGNORECASE) for p in ["data entry"]],
        match_fields=["title", "description"],
    )


def test_matches_include_keyword(cfg):
    posting = _posting(title="Senior Bioinformatics Scientist")
    assert is_relevant(posting, cfg) is True


def test_no_match_returns_false(cfg):
    posting = _posting(title="Marketing Manager", description="No relevant terms here")
    assert is_relevant(posting, cfg) is False


def test_exclude_overrides_include(cfg):
    posting = _posting(title="Bioinformatics Data Entry Clerk")
    assert is_relevant(posting, cfg) is False


def test_match_in_description(cfg):
    posting = _posting(title="Scientist II", description="Experience with NGS pipelines required")
    assert is_relevant(posting, cfg) is True
