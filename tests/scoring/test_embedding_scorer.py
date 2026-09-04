import json
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
        fingerprint_path=tmp_path / "fingerprint.json",
        jd_scores_csv=None,
    )


def _touch_artifacts(cfg):
    cfg.regressor_path.write_bytes(b"")
    cfg.scaler_path.write_bytes(b"")


@patch("job_scraper.scoring.embedding_scorer.load_embedding_model")
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


@patch("job_scraper.scoring.embedding_scorer._check_fingerprint")
@patch("job_scraper.scoring.embedding_scorer.joblib")
@patch("job_scraper.scoring.embedding_scorer.load_embedding_model")
@patch("job_scraper.scoring.embedding_scorer._encode")
def test_score_postings_maps_url_to_prediction(
    mock_encode, mock_load_model, mock_joblib, mock_check_fingerprint, cfg
):
    _touch_artifacts(cfg)
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
    mock_check_fingerprint.assert_called_once()


@patch("job_scraper.scoring.embedding_scorer.joblib")
@patch("job_scraper.scoring.embedding_scorer.load_embedding_model")
@patch("job_scraper.scoring.embedding_scorer._encode")
def test_score_postings_scores_normally_with_a_matching_fingerprint(
    mock_encode, mock_load_model, mock_joblib, cfg
):
    """The real _check_fingerprint path, not mocked away: a fingerprint recorded
    against this exact model must let scoring proceed and produce predictions."""
    _touch_artifacts(cfg)
    model = _fake_model()
    mock_load_model.return_value = model
    embedding_scorer.save_fingerprint(embedding_scorer.model_fingerprint(model, cfg), cfg)

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

    result = embedding_scorer.score_postings({"url1": "jd one", "url2": "jd two"}, cfg)

    assert result == pytest.approx({"url1": 0.11, "url2": 0.42})


def test_score_postings_empty_input_short_circuits(cfg):
    # No artifacts on disk and no fingerprint - proves the empty-input short circuit
    # runs before any artifact/fingerprint check, not that the checks pass.
    assert embedding_scorer.score_postings({}, cfg) == {}


def test_score_postings_raises_when_regressor_missing(cfg):
    cfg.scaler_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=str(cfg.regressor_path)):
        embedding_scorer.score_postings({"url1": "jd one"}, cfg)


def test_score_postings_raises_when_scaler_missing(cfg):
    cfg.regressor_path.write_bytes(b"")

    with pytest.raises(FileNotFoundError, match=str(cfg.scaler_path)):
        embedding_scorer.score_postings({"url1": "jd one"}, cfg)


def test_score_postings_raises_when_both_artifacts_missing(cfg):
    with pytest.raises(FileNotFoundError, match="train_export"):
        embedding_scorer.score_postings({"url1": "jd one"}, cfg)


# --- fingerprinting -----------------------------------------------------------------


def test_extract_modeling_revision_finds_the_hash_segment():
    module_path = "transformers_modules.nomic-ai.nomic-bert-2048.abc123ef.modeling_hf_nomic_bert"

    assert embedding_scorer._extract_modeling_revision(module_path) == "abc123ef"


def test_extract_modeling_revision_returns_none_when_no_hash_segment():
    assert embedding_scorer._extract_modeling_revision("some.plain.module") is None


def _fake_model(commit_hash="weights123", modeling_module="pkg.deadbeef12.modeling"):
    auto_model_config = MagicMock()
    auto_model_config._commit_hash = commit_hash
    auto_model_cls = type("FakeAutoModel", (), {})
    auto_model_cls.__module__ = modeling_module
    auto_model = auto_model_cls()
    auto_model.config = auto_model_config
    return {0: MagicMock(auto_model=auto_model)}


def test_model_fingerprint_reports_configured_and_resolved_identifiers(cfg):
    cfg.embedding_model_revision = "pinned-rev"
    model = _fake_model(commit_hash="weights123", modeling_module="pkg.deadbeef12.modeling")

    fingerprint = embedding_scorer.model_fingerprint(model, cfg)

    assert fingerprint == {
        "embedding_model": "fake-model",
        "embedding_model_revision": "pinned-rev",
        "weights_commit_hash": "weights123",
        "modeling_code_revision": "deadbeef12",
    }


def test_save_fingerprint_writes_json(cfg):
    fingerprint = {"embedding_model": "fake-model"}

    embedding_scorer.save_fingerprint(fingerprint, cfg)

    assert json.loads(cfg.fingerprint_path.read_text()) == fingerprint


def test_check_fingerprint_passes_when_matching(cfg):
    model = _fake_model()
    embedding_scorer.save_fingerprint(embedding_scorer.model_fingerprint(model, cfg), cfg)

    embedding_scorer._check_fingerprint(model, cfg)  # must not raise


def test_check_fingerprint_raises_when_missing(cfg):
    model = _fake_model()

    with pytest.raises(embedding_scorer.ArtifactMismatchError, match="No model fingerprint"):
        embedding_scorer._check_fingerprint(model, cfg)


def test_check_fingerprint_raises_on_revision_mismatch(cfg):
    exported_model = _fake_model(modeling_module="pkg.deadbeef01.modeling")
    embedding_scorer.save_fingerprint(
        embedding_scorer.model_fingerprint(exported_model, cfg), cfg
    )

    current_model = _fake_model(modeling_module="pkg.deadbeef02.modeling")

    with pytest.raises(embedding_scorer.ArtifactMismatchError, match="modeling_code_revision"):
        embedding_scorer._check_fingerprint(current_model, cfg)


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
