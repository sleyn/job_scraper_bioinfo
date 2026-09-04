from unittest.mock import patch

import pandas as pd

from job_scraper.aggregators.jobspy_source import fetch_jobspy, fetch_linkedin_description

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


@patch("job_scraper.aggregators.jobspy_source.LinkedIn._get_job_details")
def test_fetch_linkedin_description_extracts_job_id_from_url(mock_get_job_details):
    mock_get_job_details.return_value = {"description": "Backfilled description."}

    description = fetch_linkedin_description("https://www.linkedin.com/jobs/view/98765")

    mock_get_job_details.assert_called_once_with("98765")
    assert description == "Backfilled description."


@patch("job_scraper.aggregators.jobspy_source.LinkedIn._get_job_details")
def test_fetch_linkedin_description_returns_none_when_page_unavailable(mock_get_job_details):
    mock_get_job_details.return_value = {}

    description = fetch_linkedin_description("https://www.linkedin.com/jobs/view/98765")

    assert description is None
