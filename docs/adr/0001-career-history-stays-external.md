# Career history reference data stays in a separate repo, reached via per-file env vars

`job_scraper`'s scoring stage needs the user's career history (`reference/MEMORY.md`), resume,
and hand-scored training CSV to build fit features and train the regressor. We decided this data
stays owned by the separate `AI_Job_Helper` project rather than being copied or committed into
this repo, and is reached via four independent env vars (`JOB_HELPER_MEMORY_PATH`,
`JOB_HELPER_RESUME_PATH`, `JOB_HELPER_JD_SCORES_CSV`, `JOB_HELPER_JD_DIR`) rather than one shared
root directory plus derived relative paths.

**Why**: `job_scraper` is meant to stay a general-purpose scraper/scoring tool with no personal
data committed to it; `AI_Job_Helper` stays the sole owner of that data and its internal layout.
Per-file env vars (instead of one root + relative subpaths) mean this repo never needs to assume
anything about `AI_Job_Helper`'s directory structure — each path is independently overridable.

**Consequences**: Docker/Airflow deployment must mount or otherwise expose these files
into the container individually (see the comment in `config/scoring.yaml`) — there is no single
volume mount that covers all of them. Local (non-Docker) runs need all three env vars set before
scoring will work; `score_pending_postings()` will fail fast if they're missing.

As implemented in `docker-compose.yaml`, only two of the three reach the container: MEMORY.md and
the resume are bind-mounted read-only onto fixed paths under `/opt/airflow/career/`, with the env
vars overridden to point there, so the host layout stays in `.env`. `JOB_HELPER_JD_SCORES_CSV` is
deliberately not passed in — only `train_export.py` reads it, and training runs on the host.

**Amendment, 2026-08-23**: `_load_training_data()` originally read only the scores CSV var and
took the JD directory to be that file's parent, walking `<parent>/<name>/jd.md`. That is the same
layout assumption this ADR exists to prevent, arrived at by inference rather than configuration —
it silently required `AI_Job_Helper` to keep the CSV inside the JD directory forever. The
directory now has its own var, `JOB_HELPER_JD_DIR`, and a test asserts the two paths can point
anywhere independently. Both are training-only: neither reaches the Airflow container.
