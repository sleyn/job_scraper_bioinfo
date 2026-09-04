"""Check each Greenhouse company in config/companies.csv resolves to a valid board.

Usage: python scripts/verify_companies.py [--config-dir config]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from job_scraper.ats.ashby import BOARD_URL as ASHBY_BOARD_URL
from job_scraper.ats.base import get_session
from job_scraper.ats.greenhouse import BOARD_URL as GREENHOUSE_BOARD_URL
from job_scraper.ats.lever import BOARD_URL as LEVER_BOARD_URL
from job_scraper.ats.workday import cxs_base as workday_cxs_base
from job_scraper.config import CompanyEntry, load_companies


def _request_greenhouse(session, company: CompanyEntry, timeout: int):
    url = GREENHOUSE_BOARD_URL.format(board_token=company.board_identifier)
    return session.get(url, timeout=timeout)


def _request_lever(session, company: CompanyEntry, timeout: int):
    url = LEVER_BOARD_URL.format(company=company.board_identifier)
    return session.get(url, timeout=timeout)


def _request_ashby(session, company: CompanyEntry, timeout: int):
    url = ASHBY_BOARD_URL.format(company=company.board_identifier)
    return session.get(url, timeout=timeout)


def _request_workday(session, company: CompanyEntry, timeout: int):
    url = f"{workday_cxs_base(company.tenant, company.site, company.wd_subdomain)}/jobs"
    return session.post(
        url,
        json={"appliedFacets": {}, "limit": 5, "offset": 0, "searchText": ""},
        timeout=timeout,
    )


# Each ats_type's request function plus how to pull the job list back out of its response
# JSON, so verify() can stay a single generic loop instead of one branch per source as new
# ats/*.py fetchers are added.
_ATS_SOURCES = {
    "greenhouse": (_request_greenhouse, lambda body: body.get("jobs", [])),
    "lever": (_request_lever, lambda body: body),
    "ashby": (_request_ashby, lambda body: body.get("jobs", [])),
    "workday": (_request_workday, lambda body: body.get("jobPostings", [])),
}


def verify(config_dir: str | Path) -> bool:
    companies = load_companies(config_dir)
    session = get_session()
    all_ok = True

    for company in companies:
        if company.ats_type not in _ATS_SOURCES:
            print(f"SKIP    {company.company_name} (ats_type={company.ats_type}, not yet supported)")
            continue

        request_fn, extract_jobs = _ATS_SOURCES[company.ats_type]
        label = company.board_identifier or company.tenant

        try:
            resp = request_fn(session, company, 10)
        except Exception as exc:
            print(f"ERROR   {company.company_name} ({label}): {exc}")
            all_ok = False
            continue

        if resp.status_code == 200 and extract_jobs(resp.json()):
            count = len(extract_jobs(resp.json()))
            print(f"OK      {company.company_name} ({label}): {count} jobs")
        elif resp.status_code == 200:
            print(f"WARN    {company.company_name} ({label}): 200 but no jobs returned")
            all_ok = False
        else:
            print(f"FAIL    {company.company_name} ({label}): HTTP {resp.status_code}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()
    ok = verify(args.config_dir)
    sys.exit(0 if ok else 1)
