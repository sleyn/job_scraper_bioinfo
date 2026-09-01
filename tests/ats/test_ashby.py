from unittest.mock import MagicMock, patch

from job_scraper.ats.ashby import fetch_ashby

FIXTURE = {
    "jobs": [
        {
            "id": "job-abc-123",
            "title": "Bioinformatics Scientist",
            "location": "Remote - US",
            "descriptionPlain": "We need someone skilled in NGS pipelines.",
            "publishedAt": "2026-06-20T10:00:00.000Z",
            "jobUrl": "https://jobs.ashbyhq.com/acme/job-abc-123",
            "department": "R&D",
            "team": "Bioinformatics",
            "employmentType": "FullTime",
        }
    ]
}


@patch("job_scraper.ats.ashby.get_session")
def test_fetch_ashby_parses_jobs(mock_get_session):
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = FIXTURE
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response
    mock_get_session.return_value = mock_session

    postings = fetch_ashby("acme", "Acme Inc")

    assert len(postings) == 1
    posting = postings[0]
    assert posting.title == "Bioinformatics Scientist"
    assert posting.company == "Acme Inc"
    assert posting.url == "https://jobs.ashbyhq.com/acme/job-abc-123"
    assert "NGS" in posting.description
    assert posting.location == "Remote - US"
    assert posting.raw_id == "job-abc-123"
    assert posting.posted_date is not None
    assert posting.extra["team"] == "Bioinformatics"
