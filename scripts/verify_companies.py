"""Check each Greenhouse company in config/companies.csv resolves to a valid board.

Usage: python scripts/verify_companies.py [--config-dir config]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from job_scraper.ats.base import get_session
from job_scraper.ats.greenhouse import BOARD_URL
from job_scraper.config import load_companies


def verify(config_dir: str | Path) -> bool:
    companies = load_companies(config_dir)
    session = get_session()
    all_ok = True

    for company in companies:
        if company.ats_type != "greenhouse":
            print(f"SKIP    {company.company_name} (ats_type={company.ats_type}, not yet supported)")
            continue

        url = BOARD_URL.format(board_token=company.board_identifier)
        try:
            resp = session.get(url, timeout=10)
        except Exception as exc:
            print(f"ERROR   {company.company_name} ({company.board_identifier}): {exc}")
            all_ok = False
            continue

        if resp.status_code == 200 and resp.json().get("jobs"):
            count = len(resp.json()["jobs"])
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
