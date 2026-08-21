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
- Pipeline is three stages: scrape (per-source fetch) → filter (`is_relevant`, cheap keyword
  pre-filter) → score (`job_scraper/scoring/`, expensive ML fit-prediction, second stage). Scoring
  is intended to run only on `is_relevant = true` postings — see `CONTEXT.md` for the domain model
  (Score, Hand-scored JD, Targeting Screen) and `TODO.md`'s "Scoring" section for what's still
  unwired.

## MVP scope

This is intentionally an MVP: only Greenhouse (ATS-direct) and JobSpy (aggregator boards) are
implemented for sourcing, plus embedding-based ML scoring of relevant postings. Lever, Ashby,
Workday, and niche bio job boards (BioSpace, Naturejobs, etc.) are deferred extension points, not
unfinished work — `job_scraper/ats/base.py` and the `JobPosting` data model are designed so each
new source drops in as a new fetch function + one new DAG task, without touching storage or
filtering code. See `TODO.md` for the full deferred-work roadmap.

Scoring is core to this repo's purpose, not a deferred extra, but the `ML_JD_scoring` branch's
integration work (DAG wiring, `is_relevant` gating, exported model artifacts) is still in
progress — see `TODO.md`.

## Project Structure

One line per file/module — read this before opening files, to orient without re-reading the
whole codebase.

```
job_scraper/
  models.py              JobPosting dataclass — the common shape every source normalizes into
  config.py               loaders for keywords.yaml / companies.csv / settings.yaml / scoring.yaml;
                           compiles keyword regex with word boundaries (\b(?:pattern)\b);
                           load_scoring_config() resolves the three career-history paths from their
                           own env vars (not from scoring.yaml), failing loudly if one is unset
  pipeline.py              run_source(source_type, config_dir, db_path) — fetch -> filter -> upsert,
                           the single entry point both the DAG and manual scripts call; per-company
                           fetch failures are caught and logged, not fatal to the whole task
  db/
    schema.py               CREATE TABLE statements, init_db(), get_connection() (WAL mode, for
                             safe concurrent writes from parallel Airflow tasks)
    repository.py            upsert_postings() — dedup via url UNIQUE constraint + ON CONFLICT;
                             get_postings_missing_score()/update_scores() — the score stage's read/
                             write seam; the read side gates on is_relevant = 1 so only relevant
                             postings are ever embedded;
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
  scoring/
    embedding_scorer.py        build_features() embeds JD text + reference/MEMORY.md sections +
                             resume via a sentence-transformer, then score_postings() applies a
                             joblib-exported regressor+scaler to predict a 0-100 fit score
    train_export.py            CLI: fits the production regressor (ridge/elasticnet/nusvr) on the
                             hand-scored JD set via Optuna (maximizing CV Spearman), exports to
                             config/scoring/{regressor,scaler}.joblib
config/
  keywords.yaml              relevance filter include/exclude regex patterns
  companies.csv               Greenhouse seed list (company_name, ats_type, board_identifier,
                             tenant, careers_url, notes — some rows flagged UNVERIFIED, see TODO.md)
  settings.yaml                JobSpy search terms/locations/site_names
  scoring.yaml                 embedding model name + paths for career-history reference data
                             (env-var driven, see Config below) and exported model artifact paths
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
- `config/scoring.yaml` — embedding model + artifact paths for scoring. Career-history reference
  data (MEMORY.md, resume, hand-scored training CSV) is owned by the separate `AI_Job_Helper`
  project, not this repo — each required file is reached via its own env var
  (`JOB_HELPER_MEMORY_PATH`, `JOB_HELPER_RESUME_PATH`, `JOB_HELPER_JD_SCORES_CSV`), not a shared
  root directory, so this repo never needs to know `AI_Job_Helper`'s internal layout.

## Running tests

```
source .venv/bin/activate
python -m pytest tests/
```

## Agent skills

### Issue tracker

Local markdown: specs and tickets live under `.scratch/<feature-slug>/` (no git remote, so
no GitHub/GitLab issues). See `docs/agents/issue-tracker.md`.

### Domain docs

Single-context: one `CONTEXT.md` and `docs/adr/` at the repo root, both created lazily.
See `docs/agents/domain.md`.

See `README.md` for installation and usage instructions.
