# Split scoring dependencies into their own extra; serve the triage web UI from a separate lightweight Docker image

We're adding a web UI (`job_scraper/web/`) for reviewing postings and setting **Application
Status**, running as its own always-on container alongside the existing Airflow stack, reading
and writing the same `data/jobs.db`. It has no need for the scoring stage's dependencies
(`torch`, `sentence-transformers`, `transformers`, `optuna`, `scikit-learn`, `scipy`) — it never
imports `job_scraper/scoring/`. Those packages were previously unconditional entries in
`pyproject.toml`'s core `dependencies`, so a plain `pip install -e .` (or a hypothetical `.[web]`
extra layered on top) would still pull in the multi-GB ML stack.

We moved those scoring-only packages into a new `scoring` extra (mirroring the existing
`notebook` extra) and built a separate `Dockerfile.web` (lean `python:3.12-slim` base, installs
`.[web]`) for a new `web` service in `docker-compose.yaml`, sharing the same `./data` bind mount
as the Airflow services. The Airflow `Dockerfile` now installs `.[scoring]` explicitly instead of
relying on those packages being unconditional.

**Why**: the whole point of a separate image for the triage UI is a small, fast-starting
container; without this split it would carry the same ML dependency weight as the Airflow image
for a service that never uses it.
