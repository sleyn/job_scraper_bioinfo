# Job Scraper

A personal job-search automation tool for bioinformatics roles. It scrapes Greenhouse-backed
company career pages and aggregator boards (LinkedIn, Indeed, ZipRecruiter, Glassdoor, Google
— via [JobSpy](https://github.com/speedyapply/JobSpy)), filters postings by a configurable
bioinformatics keyword list, and stores everything in a local SQLite database. An Airflow DAG
runs the scrape daily; you query the database yourself whenever you want to see results.

## Architecture

```
sources (Greenhouse API, JobSpy) -> normalize to JobPosting -> keyword filter -> SQLite (data/jobs.db)
```

- `job_scraper/` — the scraper package (storage, filtering, fetch logic). Independent of
  Airflow, runnable from a plain Python venv.
- `dags/` — the Airflow DAG, a thin wrapper that calls into `job_scraper`.
- `config/` — `keywords.yaml` (relevance filter), `companies.csv` (Greenhouse targets),
  `settings.yaml` (JobSpy search params).

This is an **MVP**: only Greenhouse + JobSpy are wired up. Lever, Ashby, Workday, and niche
bio job boards (BioSpace, Naturejobs, etc.) are deferred — the architecture supports adding
them later without rework.

## Installation

**Prerequisites:** Docker Desktop (or Docker Engine + Compose) running; Python 3.11+ and
[`uv`](https://docs.astral.sh/uv/) for local development.

1. `cd Job_Scraper`
2. Copy `.env.example` to `.env` and fill in values:
   - `AIRFLOW_UID` — on macOS/Linux, run `id -u` and use that value.
   - `AIRFLOW_FERNET_KEY` — generate with
     `python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"`.
3. Local dev setup (for running tests/scripts without Docker):
   ```
   uv venv --python 3.12 .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt -e .
   ```
4. Run the test suite: `python -m pytest tests/`
5. Verify the seed company list against the live Greenhouse API:
   `python scripts/verify_companies.py` — fix or drop any rows that fail (see
   `config/companies.csv` notes for known-unverified entries).
6. Start Airflow:
   ```
   docker compose up -d airflow-init   # one-time: initializes metadata DB + admin user
   docker compose up -d                # starts webserver + scheduler
   ```
7. Open `http://localhost:8080` (login `admin` / `admin`, set in `docker-compose.yaml`),
   find `bioinformatics_job_scrape`, unpause it, and trigger a manual run.
8. Ongoing use: the DAG runs daily on its own. Query `data/jobs.db` directly anytime — no
   need to open the Airflow UI day-to-day.

## Querying results

```bash
sqlite3 data/jobs.db "SELECT company, title, url FROM job_postings WHERE is_relevant = 1 AND first_seen_at >= date('now', '-1 day');"
```

Or with Pandas:

```python
import sqlite3, pandas as pd
conn = sqlite3.connect("data/jobs.db")
df = pd.read_sql("SELECT * FROM job_postings WHERE is_relevant = 1", conn)
```

## Adding a Greenhouse company

Add a row to `config/companies.csv` with `ats_type=greenhouse` and your best guess at the
board token (visible in the company's Greenhouse-hosted careers URL, or
`https://api.greenhouse.io/v1/boards/{token}/jobs`). Run `python scripts/verify_companies.py`
to confirm it resolves before the next DAG run.

## Tuning keyword filtering

Edit `config/keywords.yaml` (`include`/`exclude` regex fragments, matched against
title + description). Changes only affect newly-ingested postings by default. To recompute
`is_relevant` for postings already in the database against the current keyword list:

```
python -m job_scraper.filtering.backfill
```

## Known limitations

- **Cross-source duplicates**: the same job can appear as two rows if found via both
  Greenhouse directly and JobSpy/LinkedIn (different URLs). No cross-source dedup in v1.
- **Greenhouse `posted_date`**: Greenhouse's API doesn't expose a clean "posted" date
  distinct from "updated" — `posted_date` is mapped from `updated_at`.
- **MVP scope**: no Lever, Ashby, Workday, or niche board (BioSpace, Naturejobs) scrapers yet.
- **JobSpy site reliability**: Glassdoor and ZipRecruiter frequently return 403/400 errors due
  to anti-bot measures on their end; this is a known JobSpy/upstream limitation, not a bug here.
  LinkedIn and Indeed are the most reliable sites in `config/settings.yaml`.
- **Keyword filter is substring/regex matching, not NLP**: postings at genomics-focused
  companies (e.g. 10x Genomics) can match on boilerplate "About Us" text even for unrelated
  roles (e.g. sales, finance) since that text mentions the company's core technology. Tune
  `config/keywords.yaml` `exclude` list or review matches manually if this is noisy.
