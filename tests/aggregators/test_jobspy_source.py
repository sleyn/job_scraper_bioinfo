from unittest.mock import patch

import pandas as pd

from job_scraper.aggregators.jobspy_source import fetch_jobspy

FIXTURE = pd.DataFrame(
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


@patch("job_scraper.aggregators.jobspy_source.scrape_jobs")
def test_fetch_jobspy_requests_linkedin_descriptions(mock_scrape_jobs):
    mock_scrape_jobs.return_value = FIXTURE

    postings = fetch_jobspy(
        site_names=["linkedin"],
        search_terms=["bioinformatics scientist"],
        locations=["United States"],
    )

    mock_scrape_jobs.assert_called_once()
    assert mock_scrape_jobs.call_args.kwargs["linkedin_fetch_description"] is True
    assert len(postings) == 1
    assert postings[0].description == "We need someone skilled in NGS pipelines."
