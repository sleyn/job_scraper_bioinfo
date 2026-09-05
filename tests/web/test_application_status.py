from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import upsert_postings
from job_scraper.db.schema import get_connection, init_db
from job_scraper.models import JobPosting
from job_scraper.web import create_app


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url, title, company):
    return JobPosting(
        source="greenhouse",
        company=company,
        title=title,
        location="Remote",
        description="Bioinformatics pipelines.",
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path)
    app.testing = True
    return app.test_client()


def _status(db_path, url):
    conn = get_connection(db_path)
    try:
        row = conn.execute(
            "SELECT application_status, application_status_updated_at "
            "FROM job_postings WHERE url = ?",
            (url,),
        ).fetchone()
        return row
    finally:
        conn.close()


def test_marking_applied_persists_and_reflects_in_detail_panel(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    response = client.post("/postings/1/status", data={"status": "applied"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    row = _status(db_path, posting.url)
    assert row["application_status"] == "applied"
    assert row["application_status_updated_at"] is not None
    assert 'status-btn active' in body


def test_marking_skip_removes_card_and_returns_default_state_when_no_next(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    response = client.post("/postings/1/status", data={"status": "skip"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    row = _status(db_path, posting.url)
    assert row["application_status"] == "skip"
    assert 'id="card-1" hx-swap-oob="delete"' in body
    assert "Select a posting" in body


def test_marking_skip_advances_to_next_posting_detail(db_path, client):
    high = _posting("https://example.com/high", "Senior Bioinformatics Scientist", "Acme")
    low = _posting("https://example.com/low", "Bioinformatics Scientist", "Beta")
    upsert_postings(db_path, [high, low], {high.url: True, low.url: True})

    response = client.post("/postings/1/status?next_id=2", data={"status": "skip"})
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert _status(db_path, high.url)["application_status"] == "skip"
    assert "Bioinformatics Scientist" in body
    assert "Beta" in body
    assert 'id="card-1" hx-swap-oob="delete"' in body


def test_skipped_postings_excluded_from_list_but_not_deleted(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})
    client.post("/postings/1/status", data={"status": "skip"})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert "Bioinformatics Scientist" not in body
    row = _status(db_path, posting.url)
    assert row["application_status"] == "skip"


def test_applied_postings_remain_in_list_with_badge(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})
    client.post("/postings/1/status", data={"status": "applied"})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert "Bioinformatics Scientist" in body
    assert "Applied" in body


def test_status_can_be_switched_back_with_no_special_undo_flow(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    client.post("/postings/1/status", data={"status": "skip"})
    client.post("/postings/1/status", data={"status": "new"})

    assert _status(db_path, posting.url)["application_status"] == "new"


def test_invalid_status_returns_400(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    response = client.post("/postings/1/status", data={"status": "bogus"})

    assert response.status_code == 400


def test_status_route_404_for_unknown_id(client):
    response = client.post("/postings/999/status", data={"status": "applied"})

    assert response.status_code == 404
