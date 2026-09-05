from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import upsert_postings
from job_scraper.db.schema import init_db
from job_scraper.models import JobPosting
from job_scraper.web import create_app


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url, title, company, description="Bioinformatics pipelines."):
    return JobPosting(
        source="greenhouse",
        company=company,
        title=title,
        location="Remote",
        description=description,
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path)
    app.testing = True
    return app.test_client()


def test_detail_route_renders_description_and_link(db_path, client):
    posting = _posting(
        "https://example.com/job", "Bioinformatics Scientist", "Acme",
        description="Build variant calling pipelines.",
    )
    upsert_postings(db_path, [posting], {posting.url: True})
    row_id = 1  # first row in a fresh DB

    response = client.get(f"/postings/{row_id}/detail")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Build variant calling pipelines." in body
    assert f'href="{posting.url}"' in body
    assert 'target="_blank"' in body


def test_detail_route_404_for_unknown_id(client):
    response = client.get("/postings/999/detail")

    assert response.status_code == 404


def test_list_page_cards_link_to_detail_route_via_htmx(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert 'hx-get="/postings/1/detail"' in body
    assert 'hx-target="#detail-panel"' in body


def test_list_page_default_state_has_no_detail_selected(db_path, client):
    posting = _posting("https://example.com/job", "Bioinformatics Scientist", "Acme")
    upsert_postings(db_path, [posting], {posting.url: True})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert "Select a posting" in body
    assert "Build variant calling pipelines." not in body
