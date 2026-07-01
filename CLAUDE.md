# Job Scraper

Personal bioinformatics job-search scraper. Aggregates postings from Greenhouse-backed
company career pages and job board aggregators (LinkedIn, Indeed, etc. via the JobSpy
library) into a local SQLite database, filtered by a configurable bioinformatics keyword
list, orchestrated by a daily Airflow DAG running in Docker.

## Architecture

- `job_scraper/` is an Airflow-independent Python package — testable with pytest, runnable
  from a local venv with no Docker/Airflow dependency.
- `dags/bioinformatics_job_scrape_dag.py` only imports from `job_scraper` and calls
  `job_scraper.pipeline.run_source(source_type, config_dir, db_path)` per task. No scraping/
  filtering/DB logic lives in the DAG file itself.
- Storage: SQLite at `data/jobs.db`. `url` is the dedupe key (`job_postings.url UNIQUE`,
  upserted via `ON CONFLICT`). `is_relevant` is computed once at ingestion via keyword
  matching and stored as a queryable column.
- Each scraper module (`job_scraper/ats/greenhouse.py`, `job_scraper/aggregators/jobspy_source.py`)
  is a pure "fetch and return `list[JobPosting]`" function — no DB or filtering logic — so they're
  unit-testable with mocked HTTP/DataFrame responses.

## MVP scope

This is intentionally an MVP: only Greenhouse (ATS-direct) and JobSpy (aggregator boards) are
implemented. Lever, Ashby, Workday, and niche bio job boards (BioSpace, Naturejobs, etc.) are
deferred extension points, not unfinished work — `job_scraper/ats/base.py` and the `JobPosting`
data model are designed so each new source drops in as a new fetch function + one new DAG task,
without touching storage or filtering code. See `TODO.md` for the full deferred-work roadmap.

## Project Structure

One line per file/module — read this before opening files, to orient without re-reading the
whole codebase.

```
job_scraper/
  models.py              JobPosting dataclass — the common shape every source normalizes into
  config.py               loaders for keywords.yaml / companies.csv / settings.yaml; compiles
                           keyword regex with word boundaries (\b(?:pattern)\b)
  pipeline.py              run_source(source_type, config_dir, db_path) — fetch -> filter -> upsert,
                           the single entry point both the DAG and manual scripts call; per-company
                           fetch failures are caught and logged, not fatal to the whole task
  db/
    schema.py               CREATE TABLE statements, init_db(), get_connection() (WAL mode, for
                             safe concurrent writes from parallel Airflow tasks)
    repository.py            upsert_postings() — dedup via url UNIQUE constraint + ON CONFLICT;
                             record_run() — writes scrape_runs (one row per source per run)
  filtering/
    keyword_filter.py        is_relevant(posting, cfg) -> bool
    backfill.py               CLI (python -m job_scraper.filtering.backfill): recompute
                             is_relevant for existing rows after editing keywords.yaml
  ats/
    base.py                   shared requests.Session() w/ User-Agent, reused by future ats/*.py
    greenhouse.py              fetch_greenhouse(board_token, company_name) -> list[JobPosting];
                             strips HTML from `content`, maps `updated_at` -> posted_date
  aggregators/
    jobspy_source.py           fetch_jobspy(...) -> list[JobPosting], wraps the jobspy library;
                             handles NaN fields from the returned DataFrame via pd.isna() checks
config/
  keywords.yaml              relevance filter include/exclude regex patterns
  companies.csv               Greenhouse seed list (company_name, ats_type, board_identifier,
                             tenant, careers_url, notes — some rows flagged UNVERIFIED, see TODO.md)
  settings.yaml                JobSpy search terms/locations/site_names
scripts/
  verify_companies.py          pings each companies.csv row's derived Greenhouse API URL,
                             reports pass/fail; run after editing companies.csv
dags/
  bioinformatics_job_scrape_dag.py   thin Airflow wrapper, @daily schedule, two independent
                             tasks (scrape_greenhouse_task, scrape_jobspy_task) calling
                             pipeline.run_source()
tests/                       mirrors job_scraper/ layout; mocked HTTP for ats/, temp SQLite
                             (tmp_path fixture) for db/
docker-compose.yaml           local Airflow stack: postgres (Airflow's own metadata DB, separate
                             from data/jobs.db), airflow-init, airflow-webserver, airflow-scheduler
Dockerfile                   extends apache/airflow, installs requirements.txt + editable
                             job_scraper package
.env.example                  AIRFLOW_UID, AIRFLOW_FERNET_KEY — copy to .env before docker compose up
data/jobs.db                  the actual SQLite output, gitignored — query directly, see README.md
```

## Config

- `config/keywords.yaml` — bioinformatics relevance keyword include/exclude lists.
- `config/companies.csv` — Greenhouse company seed list (`company_name,ats_type,board_identifier,tenant,careers_url,notes`).
- `config/settings.yaml` — JobSpy search terms/locations/sites.

## Running tests

```
source .venv/bin/activate
python -m pytest tests/
```

See `README.md` for installation and usage instructions.
