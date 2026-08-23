# TODO

Work intentionally deferred from the Greenhouse + JobSpy + SQLite + Airflow MVP, plus issues
found (and documented, not fixed) during live verification. See `CLAUDE.md` for architecture
and `README.md` "Known limitations" for user-facing notes on some of these.

## Scoring

`job_scraper/scoring/` (embedding-based fit scoring) is core scope now, not a deferred extra —
see `CONTEXT.md` for the Score/Hand-scored JD/Targeting Screen domain model — but the
`ML_JD_scoring` branch's integration is unfinished:

- [x] **Export production model artifacts.** `config/scoring/{regressor,scaler}.joblib` are
      exported by `python -m job_scraper.scoring.train_export`. Re-run it after adding
      hand-scored JDs; the artifacts are gitignored, so each checkout must train its own.
- [x] **Pass the career-history env vars into the Docker/Airflow scoring container.**
      `docker-compose.yaml` bind-mounts MEMORY.md and the resume read-only at
      `/opt/airflow/career/` and overrides the two vars to those paths. `JOB_HELPER_JD_SCORES_CSV`
      is intentionally not passed in — only `train_export` needs it, and that runs on the host.
- [x] **Test `train_export.py`.** `tests/scoring/test_train_export.py` covers the loader, the
      tuning helpers and the export round-trip; the degenerate-trial guard in `_spearman_scorer`
      was fixed in the process. See `.scratch/train-export-tests/`.
- [ ] **Run the scoring stage under Airflow once.** Everything below the DAG is exercised —
      the stage has been run end to end on a copy of the real `data/jobs.db` from a local venv —
      but `score_postings_task` has never executed inside the container, so the HF cache mount
      and the career-history bind mounts are verified by `docker compose config` only. Boot the
      stack and trigger one run before trusting the daily schedule.
- [ ] **Migrate the live `data/jobs.db`.** It predates the `score` column and has 627 postings,
      338 of them relevant and unscored. `_ensure_score_column()` adds the column on the next
      `init_db()`, verified on a copy — but until something runs, the live DB is pre-scoring.

- [ ] **JobSpy LinkedIn returns no descriptions.** All 373 LinkedIn rows in `data/jobs.db`
      have an empty `description`; Indeed and Greenhouse rows are fine. JobSpy needs
      `linkedin_fetch_description=True` (an extra request per posting, so it is slow and
      rate-limit-prone — check whether the run still fits the daily schedule). Until then
      those postings are never scored: `get_postings_missing_score()` skips empty
      descriptions rather than scoring the embedding of an empty string, so they sit at
      `score IS NULL`. That is 166 of 338 otherwise-relevant postings unassessed.

## New ATS sources

- [ ] **Lever scraper** (`job_scraper/ats/lever.py`). `GET https://api.lever.co/v0/postings/{company}?mode=json`.
      Returns `text` (title), `descriptionPlain`, `categories`, `location`, `hostedUrl`. Follow
      the `greenhouse.py` pattern: pure function `fetch_lever(company_slug, company_name) -> list[JobPosting]`,
      no DB/filtering logic. Add a `scrape_lever_task()` to the DAG and a `lever` row type to
      `config/companies.csv` (already has a spare `ats_type` column for this).
- [ ] **Ashby scraper** (`job_scraper/ats/ashby.py`). `GET https://api.ashbyhq.com/posting-api/job-board/{company}?includeCompensation=true`.
      Returns `title`, `location`, `descriptionPlain`, `publishedAt`, `jobUrl`. Same pattern as above.
- [ ] **Workday scraper** (`job_scraper/ats/workday.py`). More involved than the others:
      `POST https://{company}.wdN.myworkdayjobs.com/wday/cxs/{tenant}/{site}/jobs` with JSON body
      `{appliedFacets, limit, offset, searchText}`, offset-paginated (`wdN` subdomain varies per
      tenant — wd1/wd3/wd5 — verify per company). The list endpoint only gives `title`, `location`,
      `externalPath`; need a second `GET /wday/cxs/{tenant}/{site}/job/{externalPath}` per posting
      for the full description. Consider a small thread pool for the per-job fetches given the
      2-call-per-posting pattern.

## Niche bio job boards

- [ ] **Structure investigation spike** (no code yet) — for each target site (BioSpace,
      Naturejobs, etc.), check the Network tab for an underlying JSON/RSS/XML feed before
      resorting to HTML scraping; many job-board platforms (e.g. Madgex) expose one. This
      determines whether `requests`+`BeautifulSoup` is sufficient or Playwright is needed.
- [ ] **Implement `job_scraper/boards/` package** once structure is known — `base.py` for shared
      session/rate-limiting helpers (mirroring `ats/base.py`), one module per site, normalizing
      into the same `JobPosting` shape. Add one DAG task per board for failure isolation.

## Data quality / seed list

- [ ] **Fix unverified companies in `config/companies.csv`** — Benchling, Insitro, and Vir
      Biotechnology are flagged `UNVERIFIED` (guessed Greenhouse board tokens return 404).
      Check their actual careers page to find the real ATS (may not be Greenhouse at all) and
      correct the row, or drop it. Re-run `python scripts/verify_companies.py` after.
- [ ] **Expand the seed list** as you find more target companies — add rows to `companies.csv`
      with `ats_type=greenhouse` (or `lever`/`ashby`/`workday` once those scrapers exist).

## Filtering quality

- [ ] **Reduce boilerplate false positives in keyword filtering.** Substring matching against
      full job descriptions catches a company's "About Us" blurb, not just role-specific text —
      e.g. a Sales or Finance posting at 10x Genomics matches because the company description
      mentions "genomics"/"NGS". A word-boundary bug (bare acronyms like `NGS`/`STAR` matching
      inside unrelated words like "savings"/"Started") was already fixed in `job_scraper/config.py`
      during MVP verification, but boilerplate matching remains noisy. Options to try: weight
      title matches above description matches, strip a known "About [Company]" preamble before
      matching, or only fall back to description matching when the title doesn't clearly
      indicate role type.
- [ ] **Tune `config/keywords.yaml` based on real results** — after a few days of real DAG runs,
      review `SELECT * FROM job_postings WHERE is_relevant=1` for false positives/negatives and
      adjust `include`/`exclude`. Run `python -m job_scraper.filtering.backfill` after changes to
      reclassify existing rows.

## Cross-source dedup

- [ ] **The same job can appear as two rows** if found via both Greenhouse directly and
      JobSpy/LinkedIn (different URLs, same underlying posting). Current dedup is purely on
      `url`. Fixing this needs fuzzy matching on (company, title, location) — not worth building
      until it's actually a noticeable annoyance when reviewing results.

## JobSpy reliability

- [ ] **ZipRecruiter and Glassdoor consistently fail** (403 / 400 "location not parsed") as of
      this MVP's verification — an upstream anti-bot/parsing issue in the `python-jobspy`
      library, not this codebase. Periodically check for JobSpy releases that fix this, or try
      adjusting the `locations` format in `config/settings.yaml` for Glassdoor specifically.
      LinkedIn and Indeed are reliable today.

## Operational hardening (lower priority — only if the daily-cron MVP proves insufficient)

- [ ] **Per-company dynamic task mapping** in the DAG — `scrape_greenhouse_task()` currently
      loops all Greenhouse companies inside one Airflow task. Fine at today's scale (~7
      companies); revisit with `.expand()` per company if the list grows large enough that
      per-company retry/failure isolation becomes valuable.
- [ ] **CI** to run `pytest tests/` automatically on changes.
- [ ] **Periodic `scrape_runs` review** — `SELECT * FROM scrape_runs WHERE error IS NOT NULL`
      to catch silent failures (e.g. a company's board token going stale) before they go
      unnoticed for weeks.
