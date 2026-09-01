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
from job_scraper.config import load_companies

# Each ats_type's board URL template plus how to pull the job list back out of its
# response JSON, so verify() can stay a single generic loop instead of one branch per
# source as new ats/*.py fetchers are added.
_BOARD_URLS = {
    "greenhouse": (GREENHOUSE_BOARD_URL, lambda body: body.get("jobs", [])),
    "lever": (LEVER_BOARD_URL, lambda body: body),
    "ashby": (ASHBY_BOARD_URL, lambda body: body.get("jobs", [])),
}


def verify(config_dir: str | Path) -> bool:
    companies = load_companies(config_dir)
    session = get_session()
    all_ok = True

    for company in companies:
        if company.ats_type not in _BOARD_URLS:
            print(f"SKIP    {company.company_name} (ats_type={company.ats_type}, not yet supported)")
            continue

        url_template, extract_jobs = _BOARD_URLS[company.ats_type]
        url = url_template.format(
            board_token=company.board_identifier, company=company.board_identifier
        )
        try:
            resp = session.get(url, timeout=10)
        except Exception as exc:
            print(f"ERROR   {company.company_name} ({company.board_identifier}): {exc}")
            all_ok = False
            continue

        if resp.status_code == 200 and extract_jobs(resp.json()):
            count = len(extract_jobs(resp.json()))
            print(f"OK      {company.company_name} ({company.board_identifier}): {count} jobs")
        elif resp.status_code == 200:
            print(f"WARN    {company.company_name} ({company.board_identifier}): 200 but no jobs returned")
            all_ok = False
        else:
            print(f"FAIL    {company.company_name} ({company.board_identifier}): HTTP {resp.status_code}")
            all_ok = False

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()
    ok = verify(args.config_dir)
    sys.exit(0 if ok else 1)
