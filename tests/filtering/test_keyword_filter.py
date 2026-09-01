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
        off_topic_title_patterns=[re.compile(r"\bsales\b", re.IGNORECASE)],
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


def test_off_topic_title_blocks_boilerplate_description_match(cfg):
    """A Sales posting at a genomics company whose "About Us" boilerplate mentions NGS
    should not count as relevant just because the keyword happens to be in the
    description somewhere."""
    posting = _posting(
        title="Channel Sales Account Executive",
        description="Utilizing your strong technical expertise in NGS and/or single-cell...",
    )
    assert is_relevant(posting, cfg) is False


def test_title_include_match_wins_even_with_off_topic_words_in_description(cfg):
    posting = _posting(
        title="Bioinformatics Scientist",
        description="Supports the Sales and Marketing teams with pipeline QC.",
    )
    assert is_relevant(posting, cfg) is True


def test_non_off_topic_title_still_falls_back_to_description(cfg):
    """A generic title (not off-topic, not itself an include match) still falls back to
    a genuine description-only match — off-topic scoping shouldn't over-exclude."""
    posting = _posting(title="Scientist II", description="Experience with NGS pipelines required")
    assert is_relevant(posting, cfg) is True
