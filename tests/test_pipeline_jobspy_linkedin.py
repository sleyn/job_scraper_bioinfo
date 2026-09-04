from unittest.mock import patch

import pandas as pd
import pytest

from job_scraper.db.schema import init_db
from job_scraper.pipeline import run_source, score_pending_postings

KEYWORDS_YAML = """
include:
  - bioinformatics
exclude: []
"""

SETTINGS_YAML = """
jobspy:
  site_names: ["linkedin"]
  search_terms:
    - "bioinformatics scientist"
  locations:
    - "United States"
  results_wanted: 5
  hours_old: 72
"""

LINKEDIN_FIXTURE = pd.DataFrame(
    [
        {
            "site": "linkedin",
            "company": "Acme Inc",
            "title": "Bioinformatics Scientist",
            "city": "Boston",
            "state": "MA",
            "description": "We need someone skilled in NGS pipelines.",
            "job_url": "https://linkedin.com/jobs/view/12345",
            "date_posted": "2026-06-20",
        }
    ]
)


@pytest.fixture
def full_config_dir(config_dir):
    (config_dir / "keywords.yaml").write_text(KEYWORDS_YAML)
    (config_dir / "settings.yaml").write_text(SETTINGS_YAML)
    return config_dir


@patch("job_scraper.pipeline.embed_score_postings")
@patch("job_scraper.aggregators.jobspy_source.scrape_jobs")
def test_linkedin_scrape_flows_through_to_a_score(
    mock_scrape_jobs, mock_embed_score_postings, full_config_dir, tmp_path
):
    mock_scrape_jobs.return_value = LINKEDIN_FIXTURE
    mock_embed_score_postings.return_value = {
        "https://linkedin.com/jobs/view/12345": 0.8,
    }

    db_path = str(tmp_path / "jobs.db")
    init_db(db_path)

    run_source("jobspy", full_config_dir, db_path)
    assert mock_scrape_jobs.call_args.kwargs["linkedin_fetch_description"] is True

    score_stats = score_pending_postings(full_config_dir, db_path)

    assert score_stats.pending_count == 1
    assert score_stats.scored_count == 1
    scored_url, description = next(iter(mock_embed_score_postings.call_args[0][0].items()))
    assert scored_url == "https://linkedin.com/jobs/view/12345"
    assert description == "We need someone skilled in NGS pipelines."
