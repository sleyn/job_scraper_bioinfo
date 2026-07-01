from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime


@dataclass
class JobPosting:
    source: str
    company: str
    title: str
    location: str | None
    description: str
    url: str
    posted_date: date | None
    scraped_at: datetime
    raw_id: str | None = None
    extra: dict = field(default_factory=dict)
