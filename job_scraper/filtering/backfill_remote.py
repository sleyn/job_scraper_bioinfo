"""Recompute is_remote for all stored postings, using each row's location and extras.

Usage: python -m job_scraper.filtering.backfill_remote [--db-path data/jobs.db]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from job_scraper.db.schema import get_connection
from job_scraper.filtering.remote_filter import compute_is_remote
from job_scraper.models import JobPosting


def backfill_remote(db_path: str | Path) -> int:
    conn = get_connection(str(db_path))
    try:
        rows = conn.execute("SELECT id, location, extra_json FROM job_postings").fetchall()
        updated = 0
        for row in rows:
            extra = json.loads(row["extra_json"]) if row["extra_json"] else {}
            posting = JobPosting(
                source="", company="", title="", location=row["location"],
                description="", url="", posted_date=None, scraped_at=None,
                extra=extra,
            )
            is_remote = compute_is_remote(posting)
            conn.execute(
                "UPDATE job_postings SET is_remote = ? WHERE id = ?",
                (is_remote, row["id"]),
            )
            updated += 1
        conn.commit()
        return updated
    finally:
        conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db-path", default="data/jobs.db")
    args = parser.parse_args()
    count = backfill_remote(args.db_path)
    print(f"Recomputed is_remote for {count} postings")
