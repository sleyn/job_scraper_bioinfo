from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from job_scraper.db.repository import upsert_postings
from job_scraper.db.schema import get_connection, init_db
from job_scraper.models import JobPosting
from job_scraper.pipeline import score_pending_postings

SCORING_YAML = """
embedding_model: fake-model
memory_path: /nonexistent/MEMORY.md
resume_path: /nonexistent/resume.md
reference_cache_path: data/scoring_reference_cache.npz
regressor_path: config/scoring/regressor.joblib
scaler_path: config/scoring/scaler.joblib
"""


@pytest.fixture
def config_dir(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "scoring.yaml").write_text(SCORING_YAML)
    return config_dir


@pytest.fixture
def db_path(tmp_path):
    path = str(tmp_path / "jobs.db")
    init_db(path)
    return path


def _posting(url):
    return JobPosting(
        source="greenhouse",
        company="Acme",
        title="Bioinformatics Scientist",
        location="Remote",
        description=f"desc for {url}",
        url=url,
        posted_date=None,
        scraped_at=datetime.now(timezone.utc),
        raw_id="123",
    )


@patch("job_scraper.pipeline.embed_score_postings")
def test_no_pending_postings_skips_scoring(mock_embed_score_postings, config_dir, db_path):
    stats = score_pending_postings(config_dir, db_path)

    assert stats.pending_count == 0
    assert stats.scored_count == 0
    mock_embed_score_postings.assert_not_called()


@patch("job_scraper.pipeline.embed_score_postings")
def test_scores_pending_postings_and_persists(mock_embed_score_postings, config_dir, db_path):
    postings = [_posting("https://example.com/1"), _posting("https://example.com/2")]
    irrelevant = _posting("https://example.com/3")
    upsert_postings(
        db_path,
        [*postings, irrelevant],
        {**{p.url: True for p in postings}, irrelevant.url: False},
    )

    mock_embed_score_postings.return_value = {
        "https://example.com/1": 0.9,
        "https://example.com/2": 0.3,
    }

    stats = score_pending_postings(config_dir, db_path)

    assert stats.pending_count == 2
    assert stats.scored_count == 2
    mock_embed_score_postings.assert_called_once()
    called_pending = mock_embed_score_postings.call_args[0][0]
    assert called_pending == {
        "https://example.com/1": "desc for https://example.com/1",
        "https://example.com/2": "desc for https://example.com/2",
    }

    conn = get_connection(db_path)
    rows = {
        row["url"]: row["score"]
        for row in conn.execute("SELECT url, score FROM job_postings").fetchall()
    }
    conn.close()
    assert rows == {
        "https://example.com/1": 0.9,
        "https://example.com/2": 0.3,
        "https://example.com/3": None,
    }
