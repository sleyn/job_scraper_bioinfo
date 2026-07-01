from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from job_scraper.aggregators.jobspy_source import fetch_jobspy
from job_scraper.ats.greenhouse import fetch_greenhouse
from job_scraper.config import load_companies, load_keywords, load_settings
from job_scraper.db.repository import RunStats, record_run, upsert_postings
from job_scraper.db.schema import init_db
from job_scraper.filtering.keyword_filter import is_relevant
from job_scraper.models import JobPosting

SUPPORTED_SOURCES = ("greenhouse", "jobspy")


def _fetch_greenhouse_postings(config_dir: Path) -> list[JobPosting]:
    companies = load_companies(config_dir)
    postings: list[JobPosting] = []
    for company in companies:
        if company.ats_type != "greenhouse":
            continue
        try:
            postings.extend(
                fetch_greenhouse(company.board_identifier, company.company_name)
            )
        except Exception as exc:
            print(f"WARN: skipping {company.company_name} ({company.board_identifier}): {exc}")
    return postings


def _fetch_jobspy_postings(config_dir: Path) -> list[JobPosting]:
    settings = load_settings(config_dir)["jobspy"]
    return fetch_jobspy(
        site_names=settings["site_names"],
        search_terms=settings["search_terms"],
        locations=settings["locations"],
        results_wanted=settings.get("results_wanted", 30),
        hours_old=settings.get("hours_old"),
    )


def run_source(source_type: str, config_dir: str | Path, db_path: str | Path) -> RunStats:
    if source_type not in SUPPORTED_SOURCES:
        raise ValueError(f"Unsupported source_type: {source_type!r}")

    config_dir = Path(config_dir)
    db_path = str(db_path)
    init_db(db_path)

    started_at = datetime.now(timezone.utc)
    try:
        if source_type == "greenhouse":
            postings = _fetch_greenhouse_postings(config_dir)
        else:
            postings = _fetch_jobspy_postings(config_dir)

        keyword_cfg = load_keywords(config_dir)
        relevance = {p.url: is_relevant(p, keyword_cfg) for p in postings}

        stats = upsert_postings(db_path, postings, relevance)
        record_run(
            db_path,
            source=source_type,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stats=stats,
            error=None,
        )
        return stats
    except Exception as exc:
        record_run(
            db_path,
            source=source_type,
            started_at=started_at,
            finished_at=datetime.now(timezone.utc),
            stats=None,
            error=str(exc),
        )
        raise
