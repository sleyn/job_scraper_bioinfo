from datetime import date, datetime, timezone

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


def _posting(url, title, company, source="greenhouse", posted_date=None, location="Remote"):
    return JobPosting(
        source=source,
        company=company,
        title=title,
        location=location,
        description="Bioinformatics pipelines.",
        url=url,
        posted_date=posted_date,
        scraped_at=datetime.now(timezone.utc),
    )


@pytest.fixture
def client(db_path):
    app = create_app(db_path=db_path)
    app.testing = True
    return app.test_client()


@pytest.fixture
def seeded(db_path):
    acme = _posting("https://example.com/acme", "Bioinformatics Scientist", "Acme")
    beta = _posting(
        "https://example.com/beta", "Computational Biologist", "Beta", source="jobspy:linkedin"
    )
    irrelevant = _posting("https://example.com/irrelevant", "Sales Rep", "Gamma")
    upsert_postings(
        db_path,
        [acme, beta, irrelevant],
        {acme.url: True, beta.url: True, irrelevant.url: False},
    )
    update_scores(db_path, {acme.url: 90.0, beta.url: 40.0})
    return {"acme": acme, "beta": beta, "irrelevant": irrelevant}


def test_company_filter(client, seeded):
    response = client.get("/?company=Acme")
    body = response.get_data(as_text=True)

    assert "Bioinformatics Scientist" in body
    assert "Computational Biologist" not in body


def test_source_filter(client, seeded):
    response = client.get("/?source=jobspy%3Alinkedin")
    body = response.get_data(as_text=True)

    assert "Computational Biologist" in body
    assert "Bioinformatics Scientist" not in body


def test_score_range_filter(client, seeded):
    response = client.get("/?min_score=50")
    body = response.get_data(as_text=True)

    assert "Bioinformatics Scientist" in body
    assert "Computational Biologist" not in body


def test_title_keyword_filter(client, seeded):
    response = client.get("/?q=Computational")
    body = response.get_data(as_text=True)

    assert "Computational Biologist" in body
    assert "Bioinformatics Scientist" not in body


def test_show_all_widens_beyond_is_relevant(client, seeded):
    response = client.get("/?show_all=1")
    body = response.get_data(as_text=True)

    assert "Sales Rep" in body


def test_status_filter_all_includes_skipped(client, db_path, seeded):
    client.post("/postings/1/status", data={"status": "skip"})

    default_response = client.get("/")
    all_response = client.get("/?status=all")

    assert "Bioinformatics Scientist" not in default_response.get_data(as_text=True)
    assert "Bioinformatics Scientist" in all_response.get_data(as_text=True)


def test_status_filter_narrows_to_one_status(client, seeded):
    client.post("/postings/1/status", data={"status": "applied"})

    response = client.get("/?status=applied")
    body = response.get_data(as_text=True)

    assert "Bioinformatics Scientist" in body
    assert "Computational Biologist" not in body


def test_invalid_status_filter_returns_400(client, seeded):
    response = client.get("/?status=bogus")

    assert response.status_code == 400


def test_remote_filter(client, db_path):
    remote = _posting(
        "https://example.com/remote", "Remote Bioinformatics Scientist", "Acme",
        location="Remote",
    )
    onsite = _posting(
        "https://example.com/onsite", "Onsite Bioinformatics Scientist", "Acme",
        location="Boston, MA",
    )
    upsert_postings(db_path, [remote, onsite], {remote.url: True, onsite.url: True})

    remote_response = client.get("/?remote=remote")
    onsite_response = client.get("/?remote=onsite")

    remote_body = remote_response.get_data(as_text=True)
    onsite_body = onsite_response.get_data(as_text=True)
    assert "Remote Bioinformatics Scientist" in remote_body
    assert "Onsite Bioinformatics Scientist" not in remote_body
    assert "Onsite Bioinformatics Scientist" in onsite_body
    assert "Remote Bioinformatics Scientist" not in onsite_body


def test_invalid_remote_filter_returns_400(client, seeded):
    response = client.get("/?remote=bogus")

    assert response.status_code == 400


def test_empty_state_when_filters_match_nothing(client, seeded):
    response = client.get("/?q=nonexistentkeyword")
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "No postings match these filters." in body
    assert "No relevant postings yet." not in body


def test_empty_state_default_message_when_no_filters(client, db_path):
    response = client.get("/")
    body = response.get_data(as_text=True)

    assert "No relevant postings yet." in body
    assert "No postings match these filters." not in body


def test_skip_auto_advance_stays_within_active_filter(client, db_path):
    acme1 = _posting("https://example.com/acme1", "Bioinformatics Scientist I", "Acme")
    acme2 = _posting("https://example.com/acme2", "Bioinformatics Scientist II", "Acme")
    beta = _posting("https://example.com/beta", "Bioinformatics Scientist III", "Beta")
    upsert_postings(
        db_path, [acme1, acme2, beta], {acme1.url: True, acme2.url: True, beta.url: True}
    )
    update_scores(db_path, {acme1.url: 90.0, acme2.url: 80.0, beta.url: 70.0})

    response = client.post(
        "/postings/1/status?next_id=2&company=Acme", data={"status": "skip"}
    )
    body = response.get_data(as_text=True)

    assert response.status_code == 200
    assert "Bioinformatics Scientist II" in body
    assert 'hx-post="/postings/2/status?company=Acme"' in body


def test_filter_values_prefilled_from_query_string(client, seeded):
    response = client.get("/?company=Acme&q=Bioinformatics&min_score=10")
    body = response.get_data(as_text=True)

    assert 'value="10' in body or "10.0" in body
    assert 'value="Bioinformatics"' in body
    assert 'value="Acme" selected' in body
