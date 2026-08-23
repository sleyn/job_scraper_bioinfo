import pytest

SCORING_YAML = """
embedding_model: fake-model
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


@pytest.fixture(autouse=True)
def career_history_env(monkeypatch, tmp_path):
    """The career-history files live outside this repo and are resolved from env vars,
    so every test gets machine-independent stand-ins rather than the developer's own."""
    monkeypatch.setenv("JOB_HELPER_MEMORY_PATH", str(tmp_path / "helper" / "MEMORY.md"))
    monkeypatch.setenv("JOB_HELPER_RESUME_PATH", str(tmp_path / "helper" / "resume.md"))
    monkeypatch.setenv("JOB_HELPER_JD_SCORES_CSV", str(tmp_path / "helper" / "scores.csv"))
    monkeypatch.setenv("JOB_HELPER_JD_DIR", str(tmp_path / "helper"))
