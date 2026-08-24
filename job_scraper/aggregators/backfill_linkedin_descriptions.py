"""One-off backfill for LinkedIn postings stored before description fetching was enabled
(see ticket 01, `.scratch/jobspy-linkedin-descriptions/`). A normal daily scrape only
revisits postings within its `hours_old` window, so these older rows are never re-fetched
on their own — this re-fetches each one's description directly by job id, then runs the
score stage so they stop sitting at `score IS NULL`.

Usage: python -m job_scraper.aggregators.backfill_linkedin_descriptions
       [--config-dir config] [--db-path data/jobs.db]
"""
from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass
from pathlib import Path

from job_scraper.aggregators.jobspy_source import fetch_linkedin_description
from job_scraper.db.repository import get_linkedin_urls_missing_description, update_descriptions
from job_scraper.db.schema import init_db
from job_scraper.pipeline import score_pending_postings

# Matches jobspy's own LinkedIn.delay/band_delay: this hits the same per-job endpoint
# scrape_jobs(linkedin_fetch_description=True) does, just outside its search loop.
_DELAY_SECONDS = 3
_DELAY_BAND_SECONDS = 4


@dataclass
class BackfillStats:
    attempted_count: int
    updated_count: int


def backfill_linkedin_descriptions(config_dir: str | Path, db_path: str | Path) -> BackfillStats:
    config_dir = Path(config_dir)
    db_path = str(db_path)
    init_db(db_path)

    urls = get_linkedin_urls_missing_description(db_path)
    descriptions: dict[str, str] = {}
    for i, url in enumerate(urls):
        if i > 0:
            time.sleep(random.uniform(_DELAY_SECONDS, _DELAY_SECONDS + _DELAY_BAND_SECONDS))
        description = fetch_linkedin_description(url)
        if description:
            descriptions[url] = description
        else:
            print(f"WARN: no description returned for {url}")

    updated_count = update_descriptions(db_path, descriptions) if descriptions else 0
    return BackfillStats(attempted_count=len(urls), updated_count=updated_count)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--db-path", default="data/jobs.db")
    args = parser.parse_args()

    backfill_stats = backfill_linkedin_descriptions(args.config_dir, args.db_path)
    print(
        f"Backfilled descriptions for {backfill_stats.updated_count}/"
        f"{backfill_stats.attempted_count} LinkedIn postings"
    )

    score_stats = score_pending_postings(args.config_dir, args.db_path)
    print(f"Scored {score_stats.scored_count}/{score_stats.pending_count} pending postings")
