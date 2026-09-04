from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timezone

from bs4 import BeautifulSoup

from job_scraper.ats.base import get_session
from job_scraper.models import JobPosting

# Workday assigns one slug per tenant and reuses it as the subdomain "company" segment
# for every tenant observed so far (e.g. illumina.wd1.myworkdayjobs.com/wday/cxs/illumina/...),
# so the CXS tenant slug doubles as the URL company slug rather than needing its own field.
PAGE_SIZE = 20
MAX_DETAIL_WORKERS = 5


def cxs_base(tenant: str, site: str, wd_subdomain: str) -> str:
    """Public: reused by scripts/verify_companies.py to probe a company's CXS API."""
    return f"https://{tenant}.{wd_subdomain}.myworkdayjobs.com/wday/cxs/{tenant}/{site}"


def _site_base(tenant: str, site: str, wd_subdomain: str) -> str:
    return f"https://{tenant}.{wd_subdomain}.myworkdayjobs.com/{site}"


def _strip_html(html: str | None) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(separator=" ", strip=True)


def _parse_start_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _list_postings(session, base_url: str) -> list[dict]:
    postings: list[dict] = []
    offset = 0
    while True:
        resp = session.post(
            f"{base_url}/jobs",
            json={"appliedFacets": {}, "limit": PAGE_SIZE, "offset": offset, "searchText": ""},
        )
        resp.raise_for_status()
        data = resp.json()
        batch = data.get("jobPostings", [])
        postings.extend(batch)
        offset += len(batch)
        # Some Workday tenants only report an accurate "total" on the first page (later
        # pages report 0 despite still returning full pages), so a short page -- not the
        # advertised total -- is the reliable end-of-results signal.
        if len(batch) < PAGE_SIZE:
            break
    return postings


def _fetch_detail(session, base_url: str, external_path: str) -> dict:
    resp = session.get(f"{base_url}{external_path}")
    resp.raise_for_status()
    return resp.json()


def fetch_workday(tenant: str, site: str, wd_subdomain: str, company_name: str) -> list[JobPosting]:
    session = get_session()
    base = cxs_base(tenant, site, wd_subdomain)
    site_base = _site_base(tenant, site, wd_subdomain)

    listed = _list_postings(session, base)

    # Two HTTP calls per posting (list + detail), so the detail fetches run in a small
    # thread pool rather than serially.
    with ThreadPoolExecutor(max_workers=MAX_DETAIL_WORKERS) as pool:
        details = list(
            pool.map(
                lambda job: _fetch_detail(session, base, job.get("externalPath", "")),
                listed,
            )
        )

    now = datetime.now(timezone.utc)
    postings = []
    for job, detail in zip(listed, details):
        info = detail.get("jobPostingInfo", {})
        external_path = job.get("externalPath", "")
        postings.append(
            JobPosting(
                source="workday",
                company=company_name,
                title=info.get("title") or job.get("title", ""),
                location=info.get("location") or job.get("locationsText"),
                description=_strip_html(info.get("jobDescription")),
                url=info.get("externalUrl") or f"{site_base}{external_path}",
                posted_date=_parse_start_date(info.get("startDate")),
                scraped_at=now,
                raw_id=info.get("jobReqId"),
                extra={"bulletFields": job.get("bulletFields")},
            )
        )
    return postings
