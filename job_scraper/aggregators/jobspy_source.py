from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from jobspy import scrape_jobs

from job_scraper.models import JobPosting

_EXTRA_COLUMNS = [
    "is_remote",
    "job_type",
    "min_amount",
    "max_amount",
    "currency",
    "company_industry",
]


def _str_or_default(value, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value)


def _row_to_posting(row: pd.Series) -> JobPosting:
    posted_date = None
    if pd.notna(row.get("date_posted")):
        posted_date = pd.to_datetime(row["date_posted"]).date()

    location = ", ".join(
        str(part) for part in (row.get("city"), row.get("state")) if pd.notna(part)
    ) or None

    extra = {
        col: row[col]
        for col in _EXTRA_COLUMNS
        if col in row and pd.notna(row[col])
    }

    return JobPosting(
        source=f"jobspy:{_str_or_default(row.get('site'), 'unknown')}",
        company=_str_or_default(row.get("company")),
        title=_str_or_default(row.get("title")),
        location=location,
        description=_str_or_default(row.get("description")),
        url=_str_or_default(row.get("job_url")),
        posted_date=posted_date,
        scraped_at=datetime.now(timezone.utc),
        raw_id=None,
        extra=extra,
    )


def fetch_jobspy(
    site_names: list[str],
    search_terms: list[str],
    locations: list[str],
    results_wanted: int = 30,
    hours_old: int | None = None,
) -> list[JobPosting]:
    postings: list[JobPosting] = []
    for search_term in search_terms:
        for location in locations:
            df = scrape_jobs(
                site_name=site_names,
                search_term=search_term,
                location=location,
                results_wanted=results_wanted,
                hours_old=hours_old,
                # No-op for non-LinkedIn sites; JobSpy only makes the extra
                # per-posting description request when "linkedin" is in site_name.
                linkedin_fetch_description=True,
            )
            if df is None or df.empty:
                continue
            for _, row in df.iterrows():
                if pd.isna(row.get("job_url")) or not row.get("job_url"):
                    continue
                postings.append(_row_to_posting(row))
    return postings
