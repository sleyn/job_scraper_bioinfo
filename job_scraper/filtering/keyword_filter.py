from __future__ import annotations

from job_scraper.config import KeywordConfig
from job_scraper.models import JobPosting

_FIELD_MAP = {
    "title": lambda p: p.title or "",
    "description": lambda p: p.description or "",
}


def is_relevant(posting: JobPosting, cfg: KeywordConfig) -> bool:
    text = " ".join(_FIELD_MAP[field](posting) for field in cfg.match_fields)

    if any(pattern.search(text) for pattern in cfg.exclude_patterns):
        return False

    return any(pattern.search(text) for pattern in cfg.include_patterns)
