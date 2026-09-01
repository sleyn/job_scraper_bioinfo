from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from job_scraper.aggregators.jobspy_source import fetch_jobspy
from job_scraper.ats.ashby import fetch_ashby
from job_scraper.ats.greenhouse import fetch_greenhouse
from job_scraper.ats.lever import fetch_lever
from job_scraper.config import load_companies, load_keywords, load_scoring_config, load_settings
from job_scraper.db.repository import (
    RunStats,
    get_postings_missing_score,
    record_run,
    update_scores,
    upsert_postings,
)
from job_scraper.db.schema import init_db
from job_scraper.filtering.keyword_filter import is_relevant
from job_scraper.models import JobPosting
from job_scraper.scoring.embedding_scorer import score_postings as embed_score_postings

SUPPORTED_SOURCES = ("greenhouse", "lever", "ashby", "jobspy")


@dataclass
class ScoreStats:
    pending_count: int
    scored_count: int


def _fetch_ats_postings(config_dir: Path, ats_type: str, fetch_fn) -> list[JobPosting]:
    companies = load_companies(config_dir)
    postings: list[JobPosting] = []
    for company in companies:
        if company.ats_type != ats_type:
            continue
        try:
            postings.extend(fetch_fn(company.board_identifier, company.company_name))
        except Exception as exc:
            print(f"WARN: skipping {company.company_name} ({company.board_identifier}): {exc}")
    return postings


def _fetch_greenhouse_postings(config_dir: Path) -> list[JobPosting]:
    return _fetch_ats_postings(config_dir, "greenhouse", fetch_greenhouse)


def _fetch_lever_postings(config_dir: Path) -> list[JobPosting]:
    return _fetch_ats_postings(config_dir, "lever", fetch_lever)


def _fetch_ashby_postings(config_dir: Path) -> list[JobPosting]:
    return _fetch_ats_postings(config_dir, "ashby", fetch_ashby)


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
        elif source_type == "lever":
            postings = _fetch_lever_postings(config_dir)
        elif source_type == "ashby":
            postings = _fetch_ashby_postings(config_dir)
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


def score_pending_postings(config_dir: str | Path, db_path: str | Path) -> ScoreStats:
    """Scores every relevant posting that doesn't have a score yet. Meant to run after both
    scrape sources have upserted their postings, so the embedding model loads once
    and scores everything new in a single batched call."""
    config_dir = Path(config_dir)
    db_path = str(db_path)
    init_db(db_path)

    pending = get_postings_missing_score(db_path)
    if not pending:
        return ScoreStats(pending_count=0, scored_count=0)

    scoring_cfg = load_scoring_config(config_dir)
    scores = embed_score_postings(pending, scoring_cfg)
    scored_count = update_scores(db_path, scores)

    return ScoreStats(pending_count=len(pending), scored_count=scored_count)
