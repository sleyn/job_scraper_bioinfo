"""Recompute is_relevant for all stored postings against the current keywords.yaml.

Usage: python -m job_scraper.filtering.backfill [--config-dir config] [--db-path data/jobs.db]
"""
from __future__ import annotations

import argparse
from pathlib import Path

from job_scraper.config import load_keywords
from job_scraper.db.schema import get_connection
from job_scraper.filtering.keyword_filter import is_relevant
from job_scraper.models import JobPosting


def backfill(config_dir: str | Path, db_path: str | Path) -> int:
    keyword_cfg = load_keywords(config_dir)
    conn = get_connection(str(db_path))
    try:
        rows = conn.execute("SELECT id, title, description FROM job_postings").fetchall()
        updated = 0
        for row in rows:
            posting = JobPosting(
                source="", company="", title=row["title"], location=None,
                description=row["description"] or "", url="", posted_date=None,
                scraped_at=None,
            )
            relevant = int(is_relevant(posting, keyword_cfg))
            conn.execute(
                "UPDATE job_postings SET is_relevant = ? WHERE id = ?",
                (relevant, row["id"]),
            )
            updated += 1
        conn.commit()
        return updated
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--db-path", default="data/jobs.db")
    args = parser.parse_args()
    count = backfill(args.config_dir, args.db_path)
    print(f"Recomputed is_relevant for {count} postings")
