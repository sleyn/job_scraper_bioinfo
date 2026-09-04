from unittest.mock import MagicMock, patch

from job_scraper.ats.lever import fetch_lever

FIXTURE = [
    {
        "id": "abc123",
        "text": "Bioinformatics Scientist",
        "createdAt": 1750000000000,
        "hostedUrl": "https://jobs.lever.co/acme/abc123",
        "descriptionPlain": "We need someone skilled in NGS pipelines.",
        "categories": {
            "location": "Remote - US",
            "team": "R&D",
            "commitment": "Full-time",
        },
    }
]


@patch("job_scraper.ats.lever.get_session")
def test_fetch_lever_parses_jobs(mock_get_session):
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = FIXTURE
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response
    mock_get_session.return_value = mock_session

    postings = fetch_lever("acme", "Acme Inc")

    assert len(postings) == 1
    posting = postings[0]
    assert posting.title == "Bioinformatics Scientist"
    assert posting.company == "Acme Inc"
    assert posting.url == "https://jobs.lever.co/acme/abc123"
    assert "NGS" in posting.description
    assert posting.location == "Remote - US"
    assert posting.raw_id == "abc123"
    assert posting.posted_date is not None
    assert posting.extra["categories"]["team"] == "R&D"
