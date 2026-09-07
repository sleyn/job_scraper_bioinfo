"""Check whether a list of company names run a public, directly-scrapable ATS
(Greenhouse, Lever, or Ashby), so they can be promoted into config/companies.csv
for richer ATS-direct scraping. Company names can come from JobSpy-discovered
postings already in the DB, or from an arbitrary text file (e.g. names copied
from a biotech company directory).

Workday is not probed: its API needs a tenant + CXS site slug that can't be guessed
from a company name (see job_scraper/ats/workday.py), unlike Greenhouse/Lever/Ashby
which key off one guessable slug.

Usage: python scripts/detect_jobspy_ats.py [--db-path data/jobs.db] [--config-dir config]
                                            [--company "Some Company"]
                                            [--names-file path/to/names.txt]
                                            [--check-relevance] [--workers 16]
"""
from __future__ import annotations

import argparse
import csv
import re
import sqlite3
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from job_scraper.ats.ashby import BOARD_URL as ASHBY_BOARD_URL
from job_scraper.ats.base import get_session
from job_scraper.ats.greenhouse import BOARD_URL as GREENHOUSE_BOARD_URL
from job_scraper.ats.lever import BOARD_URL as LEVER_BOARD_URL
from job_scraper.config import load_keywords
from job_scraper.filtering.keyword_filter import is_relevant
from job_scraper.models import JobPosting

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


def probe(session, ats_type: str, slug: str, timeout: int = 10) -> list | None:
    """Return the raw job list if `slug` resolves to a live board on `ats_type`, else None."""
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
    return jobs or None


_TITLE_KEYS = {
    "greenhouse": lambda job: job.get("title", ""),
    "lever": lambda job: job.get("text", ""),
    "ashby": lambda job: job.get("title", ""),
}


def relevant_titles(ats_type: str, jobs: list, cfg) -> list[str]:
    """Titles among `jobs` that match the bioinformatics keyword filter."""
    now = datetime.now(timezone.utc)
    get_title = _TITLE_KEYS[ats_type]
    matches = []
    for job in jobs:
        title = get_title(job)
        posting = JobPosting(
            source=ats_type,
            company="",
            title=title,
            location=None,
            description="",
            url="",
            posted_date=None,
            scraped_at=now,
        )
        if is_relevant(posting, cfg):
            matches.append(title)
    return matches


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


def _check_one(company: str, delay: float) -> tuple[str, str, str, list] | None:
    session = get_session()
    for slug in slug_candidates(company):
        for ats_type in _ATS_PROBES:
            jobs = probe(session, ats_type, slug)
            time.sleep(delay)
            if jobs:
                return (company, ats_type, slug, jobs)
    return None


def detect(
    companies: list[str], delay: float = 0.3, workers: int = 1
) -> list[tuple[str, str, str, list]]:
    """Probe each company against every ATS/slug candidate; return (company, ats_type, slug, jobs) hits."""
    hits = []
    if workers <= 1:
        for company in companies:
            result = _check_one(company, delay)
            if result:
                hits.append(result)
        return hits

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda c: _check_one(c, delay), companies):
            if result:
                hits.append(result)
    return hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/jobs.db")
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--company", help="Check a single company name instead of scanning the DB")
    parser.add_argument("--names-file", help="Text file of company names, one per line")
    parser.add_argument("--delay", type=float, default=0.3, help="Seconds between requests")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent probe workers")
    parser.add_argument(
        "--check-relevance",
        action="store_true",
        help="Fetch each hit's job list and filter by config/keywords.yaml before reporting",
    )
    args = parser.parse_args()

    if args.company:
        targets = [args.company]
    elif args.names_file:
        known = known_company_names(args.config_dir)
        with open(args.names_file, encoding="utf-8") as f:
            names = [line.strip() for line in f if line.strip()]
        seen = set()
        targets = []
        for name in names:
            key = name.lower()
            if key not in known and key not in seen:
                seen.add(key)
                targets.append(name)
        print(f"Probing {len(targets)} companies from {args.names_file}...\n")
    else:
        known = known_company_names(args.config_dir)
        targets = [c for c in jobspy_companies(args.db_path) if c.strip().lower() not in known]
        print(f"Probing {len(targets)} JobSpy companies not already in companies.csv...\n")

    hits = detect(targets, delay=args.delay, workers=args.workers)

    if not hits:
        print("No matches found.")
        sys.exit(0)

    cfg = load_keywords(args.config_dir) if args.check_relevance else None
    rows = []
    for company, ats_type, slug, jobs in hits:
        if cfg is not None:
            matches = relevant_titles(ats_type, jobs, cfg)
            if not matches:
                continue
            rows.append((company, ats_type, slug, len(jobs), matches))
        else:
            rows.append((company, ats_type, slug, len(jobs), None))

    if not rows:
        print(f"{len(hits)} companies resolved to a live ATS board, but none had relevant titles.")
        sys.exit(0)

    print(f"{'COMPANY':<40} {'ATS':<12} {'SLUG':<25} JOBS  RELEVANT")
    for company, ats_type, slug, count, matches in rows:
        relevant_count = len(matches) if matches is not None else "-"
        print(f"{company:<40} {ats_type:<12} {slug:<25} {count:<5} {relevant_count}")

    careers_url = {
        "greenhouse": "https://job-boards.greenhouse.io/{slug}",
        "lever": "https://jobs.lever.co/{slug}",
        "ashby": "https://jobs.ashbyhq.com/{slug}",
    }
    print("\nCandidate companies.csv rows to append (verify manually before adding):")
    for company, ats_type, slug, count, matches in rows:
        note = f"verified {datetime.now().date()} - {count} jobs"
        if matches is not None:
            note += f" ({len(matches)} relevant by title)"
        url = careers_url[ats_type].format(slug=slug)
        print(f'{company},{ats_type},{slug},,{url},{note}')
