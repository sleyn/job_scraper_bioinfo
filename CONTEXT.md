# Job Scraper

Aggregates bioinformatics job postings into a local SQLite database, and — as of the
`ML_JD_scoring` work — predicts a per-posting fit score against the user's career history,
so postings can be triaged without hand-scoring every one.

## Language

**Posting**:
A single job listing fetched from a source (Greenhouse or a JobSpy-backed aggregator) and
upserted into `job_postings`, deduped on `url`.
_Avoid_: Job, listing, ad.

**Source**:
One of the two pipelines a posting can come from: Greenhouse (ATS-direct) or JobSpy
(aggregator boards like LinkedIn/Indeed). Each source is a pure fetch function with no DB
or filtering logic.

**Relevance / `is_relevant`**:
A boolean computed once at ingestion via keyword regex matching (`filtering/keyword_filter.py`)
against the posting's title/description. A cheap, static pre-filter — not a judgment of fit,
just "mentions bioinformatics-adjacent terms."
_Avoid_: Match, filtered, qualifying.

**Score**:
A predicted 0–100 fit estimate for a single posting, stored on `job_postings.score`. Produced
by a regressor trained on hand-scored examples (see **Hand-scored JD**) using JD-embedding
features. Scoring runs only on postings where `is_relevant = true` — it is the second,
expensive stage after the cheap keyword pre-filter, not an independent full-stream pass.
_Avoid_: Rating, relevance score, match score.

**Hand-scored JD**:
A job description the user manually scored 0–100 by reading it against
`reference/MEMORY.md` and judging fit across four weighted buckets: core technical skills
(40 pts), domain/industry fit (20 pts), tooling/stack overlap (20 pts), seniority/scope
alignment (20 pts). There is no formula behind the human score — it's a judgment call per
JD. This is the training signal for the **Score** regressor; the regressor approximates
this judgment so it doesn't have to be made by hand for every new posting.

**Career history reference data**:
The user's `reference/MEMORY.md` (career history) and resume, both owned by a separate
sibling project (`AI_Job_Helper`), not this repo. This repo treats them as external inputs
reached via environment variables — this repo stays a general-purpose scraper/scoring tool
with no personal data committed to it, and `AI_Job_Helper` stays the sole owner of that data.
_Avoid_: Memory, profile, resume data (as if it lived in this repo).

**Targeting Screen**:
A downstream triage view, owned by `AI_Job_Helper` (not this repo), that uses **Score**
to flag postings worth deeper attention. Score < 75 is a flag, not a hard block — empirically,
scores ≥75 saw a 20% response rate (7/35) vs. 11% (8/73) below 75, against a 14% base rate
(Apr–Jul 2026, n=149). A second, independent flag is role family (Data Scientist/ML/AI,
Data Engineer, generic Software Engineer are low-yield families) — role family classification
is not something this repo computes.
