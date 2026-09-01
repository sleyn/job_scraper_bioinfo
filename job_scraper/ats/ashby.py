from __future__ import annotations

from datetime import date, datetime, timezone

from job_scraper.ats.base import get_session
from job_scraper.models import JobPosting

BOARD_URL = "https://api.ashbyhq.com/posting-api/job-board/{company}?includeCompensation=true"


def _parse_published_at(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fetch_ashby(company_slug: str, company_name: str) -> list[JobPosting]:
    session = get_session()
    resp = session.get(BOARD_URL.format(company=company_slug))
    resp.raise_for_status()
    data = resp.json()

    now = datetime.now(timezone.utc)
    postings = []
    for job in data.get("jobs", []):
        postings.append(
            JobPosting(
                source="ashby",
                company=company_name,
                title=job.get("title", ""),
                location=job.get("location"),
                description=job.get("descriptionPlain", "") or "",
                url=job.get("jobUrl", ""),
                posted_date=_parse_published_at(job.get("publishedAt")),
                scraped_at=now,
                raw_id=str(job.get("id")) if job.get("id") is not None else None,
                extra={
                    "department": job.get("department"),
                    "team": job.get("team"),
                    "employmentType": job.get("employmentType"),
                },
            )
        )
    return postings
