from __future__ import annotations

from datetime import date, datetime, timezone

from job_scraper.ats.base import get_session
from job_scraper.models import JobPosting

BOARD_URL = "https://api.lever.co/v0/postings/{company}?mode=json"


def _parse_created_at(value: int | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromtimestamp(value / 1000, tz=timezone.utc).date()
    except (TypeError, ValueError, OSError):
        return None


def fetch_lever(company_slug: str, company_name: str) -> list[JobPosting]:
    session = get_session()
    resp = session.get(BOARD_URL.format(company=company_slug))
    resp.raise_for_status()
    data = resp.json()

    now = datetime.now(timezone.utc)
    postings = []
    for job in data:
        categories = job.get("categories") or {}
        postings.append(
            JobPosting(
                source="lever",
                company=company_name,
                title=job.get("text", ""),
                location=categories.get("location"),
                description=job.get("descriptionPlain", "") or "",
                url=job.get("hostedUrl", ""),
                posted_date=_parse_created_at(job.get("createdAt")),
                scraped_at=now,
                raw_id=str(job.get("id")) if job.get("id") is not None else None,
                extra={"categories": categories},
            )
        )
    return postings
