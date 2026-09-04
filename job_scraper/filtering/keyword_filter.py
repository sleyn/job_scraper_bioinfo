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

    title = _FIELD_MAP["title"](posting) if "title" in cfg.match_fields else ""
    if any(pattern.search(title) for pattern in cfg.include_patterns):
        return True

    # A title matching neither an include keyword nor an off-topic keyword (e.g. a
    # generic "Scientist II") still falls through to a description match below. But a
    # title that clearly names a non-bioinformatics function (Sales, Marketing, ...)
    # skips that fallback, so a company's boilerplate "About Us" text mentioning a
    # keyword (e.g. "...expertise in NGS...") can't make that posting relevant.
    if any(pattern.search(title) for pattern in cfg.off_topic_title_patterns):
        return False

    if "description" not in cfg.match_fields:
        return False

    description = _FIELD_MAP["description"](posting)
    return any(pattern.search(description) for pattern in cfg.include_patterns)
