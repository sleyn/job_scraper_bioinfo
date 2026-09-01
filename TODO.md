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

- [x] **Scoring fails opaquely when the model artifacts are absent.** `_require_artifacts()`
      in `job_scraper/scoring/embedding_scorer.py` now raises a named `FileNotFoundError`
      pointing at `python -m job_scraper.scoring.train_export` instead of a bare
      `joblib.load` traceback.
- [x] **Nothing checks the artifacts against the configured embedding revision.**
      `train_export.py` now records a fingerprint (`save_fingerprint(model_fingerprint(...))`)
      alongside the exported artifacts; `score_postings()` calls `_check_fingerprint()` before
      scoring and raises `ArtifactMismatchError` on any mismatch, naming the differing fields.
- [x] **The revision pin does not cover the custom architecture code.** The fingerprint
      includes `modeling_code_revision`, extracted from the locally-resolved
      `trust_remote_code` module path (`_extract_modeling_revision`), not just the
      `nomic-embed-text-v1.5` weights revision — so a `nomic-bert-2048` cache change is
      caught too. Covered by `tests/scoring/test_embedding_scorer.py`.

## New ATS sources

- [x] **Lever scraper** (`job_scraper/ats/lever.py`). `fetch_lever(company_slug, company_name) ->
      list[JobPosting]`, pure function following the `greenhouse.py` pattern. Wired into
      `job_scraper/pipeline.py` (`_fetch_ats_postings` shared by all three ATS sources) and a
      `scrape_lever_task()` in the DAG; `lever` is a supported `ats_type` in
      `config/companies.csv` (Deep Genomics, verified 2026-09-01, 3 jobs). Unit tests in
      `tests/ats/test_lever.py`. Manually verified: `run_source('lever', ...)` landed 3 real
      postings in `data/jobs.db`. See `.scratch/todo-backlog-2026-09/issues/04-lever-ashby-scrapers.md`.
- [x] **Ashby scraper** (`job_scraper/ats/ashby.py`). Same pattern as Lever above. `scripts/verify_companies.py`
      generalized to dispatch by `ats_type` for both new sources. `config/companies.csv` gained
      Benchling and Insitro (both ashby, verified 2026-09-01, 49 + 16 jobs — the same two
      companies dropped from the seed list earlier for lacking Ashby support, see "Data
      quality / seed list" below). Unit tests in `tests/ats/test_ashby.py`. Manually verified:
      `run_source('ashby', ...)` landed 65 real postings in `data/jobs.db` (11 `is_relevant=1`).
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
      with `ats_type=greenhouse`, `lever`, or `ashby` (or `workday` once that scraper exists).

## Filtering quality

- [x] **Reduce boilerplate false positives in keyword filtering.** Substring matching against
      full job descriptions catches a company's "About Us" blurb, not just role-specific text —
      e.g. a Sales or Finance posting at 10x Genomics matches because the company description
      mentions "genomics"/"NGS". A word-boundary bug (bare acronyms like `NGS`/`STAR` matching
      inside unrelated words like "savings"/"Started") was already fixed in `job_scraper/config.py`
      during MVP verification, but boilerplate matching remained noisy. Fixed by scoping: an
      `include` match in the title returns relevant immediately; otherwise a new
      `off_topic_titles` list in `config/keywords.yaml` (Sales, Marketing, Business Development,
      Investor Relations, Communications Manager, Recruiter, Talent Acquisition, Human Resources,
      Mechanical Engineer) skips the description fallback, so boilerplate text can't make a
      clearly non-bioinformatics posting relevant — confirmed against the live DB: 10x Genomics'
      "Channel Sales Account Executive", "District Sales Manager", "Regional Marketing Manager",
      and Generate Biomedicines' "Head of Investor Relations" / "...Communications Manager" all
      dropped out of `is_relevant=1`. See `job_scraper/filtering/keyword_filter.py` and
      `tests/filtering/test_keyword_filter.py`.
- [x] **Tune `config/keywords.yaml` based on real results.** Reviewed `SELECT * FROM
      job_postings WHERE is_relevant=1`/`=0` against the live DB (751 rows, 7 real DAG runs).
      Found two include-pattern false negatives, not just false positives: `bioinformatics`
      didn't match "Bioinformatician"/"Bioinformaticist" titles (63 rows misclassified), and
      `computational biolog(?:y|ist)` didn't match the plural "computational biologists" —
      both broadened (`bioinformatic(?:s|ian|ist)`, `computational biolog(?:y|ists?)`). Ran
      `python -m job_scraper.filtering.backfill` against `data/jobs.db` afterward — 751 rows
      recomputed, `is_relevant=1` went from 396 to 457 (net gain from the bioinformatician fix
      outweighing the boilerplate removals).

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
