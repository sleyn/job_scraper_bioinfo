from __future__ import annotations

from datetime import datetime, timezone

from bs4 import BeautifulSoup

from job_scraper.ats.base import get_session
from job_scraper.models import JobPosting

BOARD_URL = "https://api.greenhouse.io/v1/boards/{board_token}/jobs"


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _parse_updated_at(value: str | None):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).date()
    except ValueError:
        return None


def fetch_greenhouse(board_token: str, company_name: str) -> list[JobPosting]:
    session = get_session()
    resp = session.get(BOARD_URL.format(board_token=board_token), params={"content": "true"})
    resp.raise_for_status()
    data = resp.json()

    now = datetime.now(timezone.utc)
    postings = []
    for job in data.get("jobs", []):
        postings.append(
            JobPosting(
                source="greenhouse",
                company=company_name,
                title=job.get("title", ""),
                location=(job.get("location") or {}).get("name"),
                description=_strip_html(job.get("content")),
                url=job.get("absolute_url", ""),
                posted_date=_parse_updated_at(job.get("updated_at")),
                scraped_at=now,
                raw_id=str(job.get("id")) if job.get("id") is not None else None,
                extra={
                    "departments": [d.get("name") for d in job.get("departments", [])],
                    "offices": [o.get("name") for o in job.get("offices", [])],
                },
            )
        )
    return postings
