from unittest.mock import MagicMock, patch

from job_scraper.ats.workday import fetch_workday

# PAGE_SIZE is patched to 2 for these tests, so a 2-item page is "full" (keep paginating)
# and a shorter page signals the end.
LIST_PAGE_1 = {
    "total": 3,
    "jobPostings": [
        {
            "title": "Bioinformatics Scientist",
            "externalPath": "/job/US---Remote/Bioinformatics-Scientist_R1",
            "locationsText": "US - Remote",
            "bulletFields": ["R1"],
        },
        {
            "title": "Data Engineer",
            "externalPath": "/job/US---Remote/Data-Engineer_R2",
            "locationsText": "US - Remote",
            "bulletFields": ["R2"],
        },
    ],
}
# Reports total=0 on the second page -- observed on real Workday tenants, where only the
# first page's "total" is accurate -- to confirm fetch_workday doesn't rely on it.
LIST_PAGE_2 = {
    "total": 0,
    "jobPostings": [
        {
            "title": "Research Scientist",
            "externalPath": "/job/US---Remote/Research-Scientist_R3",
            "locationsText": "US - Remote",
            "bulletFields": ["R3"],
        }
    ],
}

DETAIL_R1 = {
    "jobPostingInfo": {
        "title": "Bioinformatics Scientist",
        "jobDescription": "<p>We need someone skilled in NGS pipelines.</p>",
        "location": "US - Remote",
        "startDate": "2026-06-20",
        "jobReqId": "R1",
        "externalUrl": "https://acme.wd1.myworkdayjobs.com/acme-careers/job/US---Remote/Bioinformatics-Scientist_R1",
    }
}
DETAIL_R2 = {
    "jobPostingInfo": {
        "title": "Data Engineer",
        "jobDescription": "<p>Build pipelines.</p>",
        "location": "US - Remote",
        "startDate": "2026-06-21",
        "jobReqId": "R2",
        "externalUrl": "https://acme.wd1.myworkdayjobs.com/acme-careers/job/US---Remote/Data-Engineer_R2",
    }
}
DETAIL_R3 = {
    "jobPostingInfo": {
        "title": "Research Scientist",
        "jobDescription": "<p>Analyze genomic data.</p>",
        "location": "US - Remote",
        "startDate": "2026-06-22",
        "jobReqId": "R3",
        "externalUrl": "https://acme.wd1.myworkdayjobs.com/acme-careers/job/US---Remote/Research-Scientist_R3",
    }
}
DETAILS_BY_REQ = {"R1": DETAIL_R1, "R2": DETAIL_R2, "R3": DETAIL_R3}


@patch("job_scraper.ats.workday.PAGE_SIZE", 2)
@patch("job_scraper.ats.workday.get_session")
def test_fetch_workday_paginates_and_fetches_details(mock_get_session):
    mock_session = MagicMock()

    def post_side_effect(url, json, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = LIST_PAGE_1 if json["offset"] == 0 else LIST_PAGE_2
        return resp

    def get_side_effect(url, **kwargs):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        req_id = url.rsplit("_", 1)[-1]
        resp.json.return_value = DETAILS_BY_REQ[req_id]
        return resp

    mock_session.post.side_effect = post_side_effect
    mock_session.get.side_effect = get_side_effect
    mock_get_session.return_value = mock_session

    postings = fetch_workday("acme", "acme-careers", "wd1", "Acme Inc")

    assert len(postings) == 3
    assert mock_session.post.call_count == 2  # full page, then a short page ends pagination
    assert mock_session.get.call_count == 3  # one detail fetch per posting

    by_req_id = {p.raw_id: p for p in postings}
    scientist = by_req_id["R1"]
    assert scientist.title == "Bioinformatics Scientist"
    assert scientist.company == "Acme Inc"
    assert "NGS" in scientist.description
    assert scientist.location == "US - Remote"
    assert scientist.url == (
        "https://acme.wd1.myworkdayjobs.com/acme-careers/job/US---Remote/Bioinformatics-Scientist_R1"
    )
    assert scientist.posted_date is not None
    assert by_req_id["R3"].title == "Research Scientist"


@patch("job_scraper.ats.workday.get_session")
def test_fetch_workday_stops_on_short_first_page(mock_get_session):
    mock_session = MagicMock()

    resp = MagicMock()
    resp.raise_for_status.return_value = None
    resp.json.return_value = {"total": 0, "jobPostings": []}
    mock_session.post.return_value = resp
    mock_get_session.return_value = mock_session

    postings = fetch_workday("acme", "acme-careers", "wd1", "Acme Inc")

    assert postings == []
    mock_session.post.assert_called_once()
    mock_session.get.assert_not_called()
