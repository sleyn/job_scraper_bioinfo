from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from job_scraper.config import ScoringConfig
from job_scraper.scoring import embedding_scorer


@pytest.fixture
def cfg(tmp_path):
    memory_path = tmp_path / "MEMORY.md"
    memory_path.write_text(
        "## Section One\n"
        "Enough padding text here to clear the forty character minimum easily.\n"
        "## Section Two\n"
        "More than forty characters of padding text goes right here too, yes.\n"
    )
    resume_path = tmp_path / "resume.md"
    resume_path.write_text("Generic resume text.")

    return ScoringConfig(
        embedding_model="fake-model",
        memory_path=memory_path,
        resume_path=resume_path,
        reference_cache_path=tmp_path / "cache.npz",
        regressor_path=tmp_path / "regressor.joblib",
        scaler_path=tmp_path / "scaler.joblib",
        jd_scores_csv=None,
    )


@patch("job_scraper.scoring.embedding_scorer._load_embedding_model")
@patch("job_scraper.scoring.embedding_scorer._encode")
def test_build_features_shape_and_fit_columns(mock_encode, mock_load_model, cfg):
    mock_load_model.return_value = object()

    job_history_emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 MEMORY.md sections
    resume_emb = np.array([[0.6, 0.8]])  # _encode(...)[0] -> [0.6, 0.8]
    jd_emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2 JDs, one matching each section

    mock_encode.side_effect = [job_history_emb, resume_emb, jd_emb]

    X = embedding_scorer.build_features(["jd one", "jd two"], cfg)

    assert X.shape == (2, 2 + 3)
    # jd1 == section1 exactly -> fit_max=1.0; with only 2 sections, top3-mean == mean of both
    np.testing.assert_allclose(X[0], [1.0, 0.0, 1.0, 0.5, 0.6])
    np.testing.assert_allclose(X[1], [0.0, 1.0, 1.0, 0.5, 0.8])


@patch("job_scraper.scoring.embedding_scorer.joblib")
@patch("job_scraper.scoring.embedding_scorer._load_embedding_model")
@patch("job_scraper.scoring.embedding_scorer._encode")
def test_score_postings_maps_url_to_prediction(mock_encode, mock_load_model, mock_joblib, cfg):
    mock_load_model.return_value = object()
    mock_encode.side_effect = [
        np.array([[1.0, 0.0], [0.0, 1.0]]),
        np.array([[0.6, 0.8]]),
        np.array([[1.0, 0.0], [0.0, 1.0]]),
    ]

    scaler = MagicMock()
    scaler.transform.side_effect = lambda X: X
    regressor = MagicMock()
    regressor.predict.return_value = np.array([0.11, 0.42])
    mock_joblib.load.side_effect = [scaler, regressor]

    result = embedding_scorer.score_postings(
        {"url1": "jd one", "url2": "jd two"}, cfg
    )

    assert result == pytest.approx({"url1": 0.11, "url2": 0.42})


def test_score_postings_empty_input_short_circuits(cfg):
    assert embedding_scorer.score_postings({}, cfg) == {}


@patch("job_scraper.scoring.embedding_scorer._encode")
def test_reference_embeddings_cached_on_second_call(mock_encode, cfg):
    job_history_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    resume_emb = np.array([[0.6, 0.8]])
    mock_encode.side_effect = [job_history_emb, resume_emb]

    first = embedding_scorer._load_reference_embeddings(object(), cfg)
    assert cfg.reference_cache_path.exists()
    assert mock_encode.call_count == 2

    # Second call must come off the cache: _encode is exhausted, so any re-embed
    # raises StopIteration rather than silently recomputing.
    second = embedding_scorer._load_reference_embeddings(object(), cfg)
    assert mock_encode.call_count == 2

    np.testing.assert_allclose(second[0], first[0])
    np.testing.assert_allclose(second[1], first[1])


@patch("job_scraper.scoring.embedding_scorer._encode")
def test_reference_cache_invalidated_when_memory_changes(mock_encode, cfg):
    stale = np.array([[1.0, 0.0], [0.0, 1.0]])
    fresh = np.array([[0.0, 1.0], [1.0, 0.0], [0.5, 0.5]])
    resume_emb = np.array([[0.6, 0.8]])
    mock_encode.side_effect = [stale, resume_emb, fresh, resume_emb]

    embedding_scorer._load_reference_embeddings(object(), cfg)

    # A third section changes the cache key, so the stale vectors must not be reused.
    cfg.memory_path.write_text(
        cfg.memory_path.read_text()
        + "## Section Three\n"
        + "Yet another forty-plus characters of padding text lives here now.\n"
    )
    job_history_emb, _ = embedding_scorer._load_reference_embeddings(object(), cfg)

    assert mock_encode.call_count == 4
    np.testing.assert_allclose(job_history_emb, fresh)


@patch("job_scraper.scoring.embedding_scorer._encode")
def test_reference_cache_invalidated_when_resume_changes(mock_encode, cfg):
    job_history_emb = np.array([[1.0, 0.0], [0.0, 1.0]])
    mock_encode.side_effect = [
        job_history_emb, np.array([[0.6, 0.8]]),
        job_history_emb, np.array([[0.0, 1.0]]),
    ]

    embedding_scorer._load_reference_embeddings(object(), cfg)
    cfg.resume_path.write_text("A rewritten resume.")
    _, resume_emb = embedding_scorer._load_reference_embeddings(object(), cfg)

    assert mock_encode.call_count == 4
    np.testing.assert_allclose(resume_emb, [0.0, 1.0])


def test_split_sections_drops_short_chunks_and_keeps_headers():
    sections = embedding_scorer._split_sections(
        "Preamble text that is comfortably longer than forty characters here.\n"
        "## Kept\nThis section body is also well over the forty character floor.\n"
        "## Tiny\nToo short.\n"
        "### Nested\nAn H3 heading starts its own chunk and this body clears forty.\n"
    )

    assert len(sections) == 3
    assert sections[0].startswith("Preamble")
    assert sections[1].startswith("## Kept")
    assert sections[2].startswith("### Nested")
    assert not any(s.startswith("## Tiny") for s in sections)
