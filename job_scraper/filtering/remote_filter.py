from __future__ import annotations

import re

from job_scraper.models import JobPosting

_REMOTE_PATTERN = re.compile(r"\bremote\b", re.IGNORECASE)
_HYBRID_PATTERN = re.compile(r"\bhybrid\b", re.IGNORECASE)


def compute_is_remote(posting: JobPosting) -> int | None:
    """Tri-state remote-ness signal for the web triage filter: 1 = remote, 0 = on-site
    (including hybrid), None = no signal either way.

    JobSpy postings carry a real `is_remote` boolean from the aggregator itself (see
    aggregators/jobspy_source.py's `_EXTRA_COLUMNS`), so that value wins whenever present.
    ATS postings (Greenhouse/Lever/Ashby/Workday) have no such field — remote-ness is
    inferred from the free-text `location` string instead. "Hybrid" is checked first and
    forces on-site even when the same string also says "remote" (e.g. "Hybrid/Remote
    optional"), since hybrid still requires in-person time."""
    extra_is_remote = posting.extra.get("is_remote")
    if extra_is_remote is not None:
        return int(bool(extra_is_remote))

    location = posting.location or ""
    if not location.strip():
        return None
    if _HYBRID_PATTERN.search(location):
        return 0
    return int(bool(_REMOTE_PATTERN.search(location)))
