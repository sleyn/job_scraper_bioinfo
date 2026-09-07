"""Check whether companies discovered via JobSpy (LinkedIn/Indeed) run a public,
directly-scrapable ATS (Greenhouse, Lever, or Ashby), so they can be promoted into
config/companies.csv for richer ATS-direct scraping.

Workday is not probed: its API needs a tenant + CXS site slug that can't be guessed
from a company name (see job_scraper/ats/workday.py), unlike Greenhouse/Lever/Ashby
which key off one guessable slug.

Usage: python scripts/detect_jobspy_ats.py [--db-path data/jobs.db] [--config-dir config]
                                            [--company "Some Company"]
"""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from job_scraper.ats.ashby import BOARD_URL as ASHBY_BOARD_URL
from job_scraper.ats.base import get_session
from job_scraper.ats.greenhouse import BOARD_URL as GREENHOUSE_BOARD_URL
from job_scraper.ats.lever import BOARD_URL as LEVER_BOARD_URL

_SUFFIXES = re.compile(
    r"\b(inc|incorporated|llc|l\.l\.c|corp|corporation|co|ltd|plc|group|holdings)\b\.?",
    re.IGNORECASE,
)


def slug_candidates(company_name: str) -> list[str]:
    """Guess likely board-token slugs for a company name.

    Greenhouse/Lever/Ashby board tokens are conventionally a lowercased,
    punctuation-stripped version of the company name, so a handful of common
    variants covers most real tokens without needing per-company lookup.
    """
    base = company_name.strip()
    base = re.sub(r"[.,]", "", base)
    base = _SUFFIXES.sub("", base).strip()
    base = re.sub(r"&", "and", base)
    base = re.sub(r"[^A-Za-z0-9 -]", "", base)
    words = base.split()

    candidates = set()
    if words:
        candidates.add("".join(words).lower())
        candidates.add("-".join(words).lower())
    raw_words = company_name.strip().split()
    if raw_words:
        candidates.add("".join(raw_words).lower())
    return sorted(c for c in candidates if c)


_ATS_PROBES = {
    "greenhouse": (GREENHOUSE_BOARD_URL, lambda body: body.get("jobs", [])),
    "lever": (LEVER_BOARD_URL, lambda body: body),
    "ashby": (ASHBY_BOARD_URL, lambda body: body.get("jobs", [])),
}


def probe(session, ats_type: str, slug: str, timeout: int = 10) -> int | None:
    """Return the job count if `slug` resolves to a live board on `ats_type`, else None."""
    url_template, extract_jobs = _ATS_PROBES[ats_type]
    try:
        resp = session.get(url_template.format(board_token=slug, company=slug), timeout=timeout)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    try:
        jobs = extract_jobs(resp.json())
    except (ValueError, AttributeError):
        return None
    return len(jobs) if jobs else None


def jobspy_companies(db_path: str | Path) -> list[str]:
    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT DISTINCT company FROM job_postings WHERE source LIKE 'jobspy%' ORDER BY company"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def known_company_names(config_dir: str | Path) -> set[str]:
    path = Path(config_dir) / "companies.csv"
    if not path.exists():
        return set()
    with open(path, newline="", encoding="utf-8") as f:
        return {row["company_name"].strip().lower() for row in csv.DictReader(f)}


def detect(companies: list[str], delay: float = 0.3) -> list[tuple[str, str, str, int]]:
    """Probe each company against every ATS/slug candidate; return (company, ats_type, slug, count) hits."""
    session = get_session()
    hits = []
    for company in companies:
        found = False
        for slug in slug_candidates(company):
            for ats_type in _ATS_PROBES:
                count = probe(session, ats_type, slug)
                time.sleep(delay)
                if count:
                    hits.append((company, ats_type, slug, count))
                    found = True
            if found:
                break
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/jobs.db")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--company", help="Check a single company name instead of scanning the DB")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    args = parser.parse_args()

    if args.company:
        targets = [args.company]
    else:
        known = known_company_names(args.config_dir)
        targets = [c for c in jobspy_companies(args.db_path) if c.strip().lower() not in known]
        print(f"Probing {len(targets)} JobSpy companies not already in companies.csv...\n")

    hits = detect(targets, delay=args.delay)

    if not hits:
        print("No matches found.")
        sys.exit(0)

    print(f"{'COMPANY':<40} {'ATS':<12} {'SLUG':<25} JOBS")
    for company, ats_type, slug, count in hits:
        print(f"{company:<40} {ats_type:<12} {slug:<25} {count}")

    print("\nCandidate companies.csv rows to append (verify manually before adding):")
    for company, ats_type, slug, _ in hits:
        print(f'{company},{ats_type},{slug},,https://boards.{ats_type}.io/{slug},UNVERIFIED - auto-detected')
