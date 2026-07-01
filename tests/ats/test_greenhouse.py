from unittest.mock import MagicMock, patch

from job_scraper.ats.greenhouse import fetch_greenhouse

FIXTURE = {
    "jobs": [
        {
            "id": 12345,
            "title": "Bioinformatics Scientist",
            "updated_at": "2026-06-20T10:00:00Z",
            "absolute_url": "https://boards.greenhouse.io/acme/jobs/12345",
            "content": "<p>We need someone skilled in <b>NGS</b> pipelines.</p>",
            "location": {"name": "Remote - US"},
            "departments": [{"name": "R&D"}],
            "offices": [{"name": "Remote"}],
        }
    ]
}


@patch("job_scraper.ats.greenhouse.get_session")
def test_fetch_greenhouse_parses_jobs(mock_get_session):
    mock_session = MagicMock()
    mock_response = MagicMock()
    mock_response.json.return_value = FIXTURE
    mock_response.raise_for_status.return_value = None
    mock_session.get.return_value = mock_response
    mock_get_session.return_value = mock_session

    postings = fetch_greenhouse("acme", "Acme Inc")

    assert len(postings) == 1
    posting = postings[0]
    assert posting.title == "Bioinformatics Scientist"
    assert posting.company == "Acme Inc"
    assert posting.url == "https://boards.greenhouse.io/acme/jobs/12345"
    assert "NGS" in posting.description
    assert "<b>" not in posting.description
    assert posting.location == "Remote - US"
    assert posting.raw_id == "12345"
    assert posting.posted_date is not None
    assert posting.extra["departments"] == ["R&D"]
