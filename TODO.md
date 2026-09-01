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
- [x] **Run the scoring stage under Airflow once.** Done 2026-08-23, run `manual_verify_1`:
      all three tasks succeeded, both mounts resolved inside the container, and scoring wrote
      210 scores to the live DB with no Hub access (offline mode held). Building the image
      needed a Dockerfile fix — see the CPU-only torch note there.
- [ ] **Scoring takes ~14 min in the container vs ~110s locally.** Same 210 postings. The
      container is CPU-only while a local run uses Apple MPS. Fine for a daily schedule now,
      but it scales with the backlog, and the first full run over an empty DB would be far
      longer. Revisit if the DAG starts overrunning its schedule.
- [x] **Migrate the live `data/jobs.db`.** Done by run `manual_verify_1`: the column was added
      in place, all rows preserved (751 postings, 396 relevant, 210 scored).

- [x] **JobSpy LinkedIn returns no descriptions.** Fixed: `fetch_jobspy` now passes
      `linkedin_fetch_description=True` to `scrape_jobs`. The scrape -> filter -> score chain
      for a LinkedIn posting is covered by a repeatable test
      (`tests/test_pipeline_jobspy_linkedin.py`: mocks `scrape_jobs` and
      `embed_score_postings`, runs the real `run_source`/`score_pending_postings` path). Also
      verified manually against a live LinkedIn scrape (not repeatable from the repo, numbers
      recorded here for reference) — 5 postings, all with non-empty descriptions, all flowed
      through `is_relevant` and were scored. Timing: ~1.1s per LinkedIn posting for the extra
      description request (~35s for 30 results, one search term); with the production
      `settings.yaml` (2 search terms x 1 location) that's ~70s added to the LinkedIn leg of a
      daily run — well within the daily schedule, and no rate-limiting was hit across ~35
      requests in that manual run. The 166 already-stored LinkedIn rows with empty
      descriptions are not touched by this fix (a normal scrape only revisits postings within
      `hours_old`) — backfilled separately, see below.

- [x] **Backfill descriptions for existing NULL-description LinkedIn rows.** Added
      `python -m job_scraper.backfill_linkedin_descriptions`: re-fetches each
      relevant, empty-description LinkedIn row's description directly by job id (via
      `jobspy.linkedin.LinkedIn._get_job_details`, the same per-job request
      `linkedin_fetch_description=True` makes internally — jobspy has no public single-job
      API), updates it in place, then runs the score stage. Run live against `data/jobs.db`:
      of 186 relevant, unscored LinkedIn rows, 88 got a real description and a score; the
      other 98 came back empty because the posting itself has expired on LinkedIn
      (`GET .../jobs/view/{id}` 200s to `.../jobs/<slug>-jobs?trk=expired_jd_redirect`, not an
      error or rate limit — confirmed by hand for several ids). Those 98 can never be
      backfilled — the description no longer exists anywhere to fetch — so they stay at
      `score IS NULL` by the same rule that skips empty-description rows generally (see
      `get_postings_missing_score`'s docstring): no description means no judgment is possible,
      and that's the honest state, not a bug. `SELECT COUNT(*) FROM job_postings WHERE
      is_relevant = 1 AND score IS NULL AND source LIKE 'jobspy:linkedin%'` is 98, all
      permanently-expired postings — not near zero in absolute count, but zero recoverable
      ones remain.

- [ ] **Scoring fails opaquely when the model artifacts are absent.** `config/scoring/*.joblib`
      is gitignored, so a fresh clone has no model and `score_postings()` dies on a bare
      `joblib.load` `FileNotFoundError` pointing at a path, with nothing saying "run
      train_export first". Every other missing input in this pipeline fails loudly and by
      name (`_required_env_path`, the two loader raises); this one should match.
- [ ] **Nothing checks the artifacts against the configured embedding revision.**
      `scoring.yaml` pins `embedding_model_revision` to the revision the current regressor was
      fitted on, but that pin is a declaration only — re-pin it without re-running
      `train_export` and scoring proceeds against a feature space the model never saw, with no
      error. Storing the revision alongside the artifacts at export and comparing on load
      would make the mismatch visible instead of silent.
- [ ] **The revision pin does not cover the custom architecture code.** `revision=` reaches
      `nomic-ai/nomic-embed-text-v1.5` only; the `trust_remote_code=True` modelling code comes
      from `nomic-ai/nomic-bert-2048`, which is pinned by nothing but `HF_HUB_OFFLINE=1` and
      whatever happens to sit in the cache. A machine with a different cache can produce
      different features from the same config.

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

- [x] **Fix unverified companies in `config/companies.csv`.** Benchling and Insitro are on
      Ashby, which this repo doesn't support yet (see "New ATS sources" below) — dropped. Vir
      Biotechnology's real Greenhouse token is `virbiotechnologyinc`, not the guessed `vir` —
      corrected. Verified with `python scripts/verify_companies.py` (16 jobs) and a manual
      `run_source('greenhouse', ...)` run that persisted 16 real Vir postings to the DB.
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

## Container image

- [ ] **The Airflow image installs the whole `pyproject.toml`.** `pip install -e /opt/airflow`
      pulls `marimo`, `catboost` and `ipython` into the worker — they exist for the
      `thinking_space/` notebook and are never imported by `job_scraper`. `torch` and
      `sentence-transformers` are genuinely needed, so the image stays large regardless
      (3.43GB today), but it is carrying a notebook stack on top of that. Split the deps into
      optional groups (`[project.optional-dependencies]`) and have the Dockerfile install only
      the scoring set.
- [ ] **Two dependency manifests.** The Dockerfile installs `requirements.txt` and then
      `pyproject.toml`; the former lists six packages that the latter also declares. Nothing
      keeps them in step, and it is not obvious which one a new dependency belongs in. Fold
      `requirements.txt` into `pyproject.toml` and drop it.

## Scoring notebook (`thinking_space/score_jd/score.py`)

- [ ] **The notebook and the package disagree about how to find `AI_Job_Helper`.** The
      notebook reads a single `AI_JOB_HELPER_ROOT` and derives `job_descriptions/`,
      `reference/MEMORY.md` and the resume from it. The package deliberately does the
      opposite — four independent per-file vars, so this repo assumes nothing about that
      project's layout (ADR-0001). Anyone running both has to set two different, overlapping
      sets of variables, and the notebook still breaks if `AI_Job_Helper` is rearranged.
- [ ] **`read_jd_from_file()`'s existence check never runs.** `if Path.exists:` tests a bound
      method, which is always truthy, so a missing `jd.md` raises from `open()` rather than
      returning the intended empty string. `train_export.py` hit exactly this and now skips
      missing rows by name; the notebook still has the original bug.
