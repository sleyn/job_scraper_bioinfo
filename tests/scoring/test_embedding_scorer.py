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
