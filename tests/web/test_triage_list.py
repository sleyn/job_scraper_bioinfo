from datetime import datetime, timezone

import pytest

from job_scraper.db.repository import update_scores, upsert_postings
from job_scraper.db.schema import init_db
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


def test_empty_db_renders_without_error(client):
    response = client.get("/")

    assert response.status_code == 200
    assert b"No relevant postings yet." in response.data


def test_lists_only_relevant_postings_ordered_by_score_desc(db_path, client):
    high = _posting("https://example.com/high", "Senior Bioinformatics Scientist", "Acme")
    low = _posting("https://example.com/low", "Bioinformatics Scientist", "Beta")
    irrelevant = _posting("https://example.com/irrelevant", "Sales Development Rep", "Gamma")
    upsert_postings(
        db_path,
        [high, low, irrelevant],
        {high.url: True, low.url: True, irrelevant.url: False},
    )
    update_scores(db_path, {high.url: 90.0, low.url: 40.0})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Sales Development Rep" not in body
    assert body.index("Senior Bioinformatics Scientist") < body.index("Bioinformatics Scientist")
    assert "Acme" in body
    assert "Beta" in body


def test_unscored_relevant_postings_render_last(db_path, client):
    scored = _posting("https://example.com/scored", "Bioinformatics Scientist", "Acme")
    unscored = _posting("https://example.com/unscored", "Computational Biologist", "Beta")
    upsert_postings(db_path, [scored, unscored], {scored.url: True, unscored.url: True})
    update_scores(db_path, {scored.url: 55.0})

    response = client.get("/")
    body = response.get_data(as_text=True)

    assert body.index("Bioinformatics Scientist") < body.index("Computational Biologist")
