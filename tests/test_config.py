import pytest

from job_scraper.config import load_keywords, load_scoring_config


def test_career_history_paths_come_from_env_vars(config_dir, tmp_path):
    cfg = load_scoring_config(config_dir)

    assert cfg.memory_path == tmp_path / "helper" / "MEMORY.md"
    assert cfg.resume_path == tmp_path / "helper" / "resume.md"
    assert cfg.jd_scores_csv == tmp_path / "helper" / "scores.csv"


def test_repo_relative_paths_still_come_from_yaml(config_dir, tmp_path):
    cfg = load_scoring_config(config_dir)

    assert cfg.embedding_model == "fake-model"
    assert cfg.reference_cache_path == tmp_path / "data" / "scoring_reference_cache.npz"
    assert cfg.regressor_path == tmp_path / "config" / "scoring" / "regressor.joblib"
    assert cfg.scaler_path == tmp_path / "config" / "scoring" / "scaler.joblib"


def test_jd_scores_csv_is_optional(config_dir, monkeypatch):
    monkeypatch.delenv("JOB_HELPER_JD_SCORES_CSV")

    assert load_scoring_config(config_dir).jd_scores_csv is None


@pytest.mark.parametrize("var", ["JOB_HELPER_MEMORY_PATH", "JOB_HELPER_RESUME_PATH"])
def test_unset_required_env_var_raises_clear_error(config_dir, monkeypatch, var):
    monkeypatch.delenv(var)

    with pytest.raises(KeyError, match=var):
        load_scoring_config(config_dir)


@pytest.mark.parametrize("var", ["JOB_HELPER_MEMORY_PATH", "JOB_HELPER_RESUME_PATH"])
def test_empty_required_env_var_raises_clear_error(config_dir, monkeypatch, var):
    # .env.example ships these blank, so a half-filled .env is the common first-run state.
    monkeypatch.setenv(var, "")

    with pytest.raises(KeyError, match=var):
        load_scoring_config(config_dir)


def test_empty_jd_scores_csv_stays_none(config_dir, monkeypatch):
    monkeypatch.setenv("JOB_HELPER_JD_SCORES_CSV", "")

    assert load_scoring_config(config_dir).jd_scores_csv is None


def test_env_var_path_expands_tilde(config_dir, monkeypatch):
    monkeypatch.setenv("HOME", "/home/tester")
    monkeypatch.setenv("JOB_HELPER_RESUME_PATH", "~/resume.md")

    assert load_scoring_config(config_dir).resume_path.as_posix() == "/home/tester/resume.md"


def test_off_topic_titles_are_compiled_with_word_boundaries(config_dir):
    (config_dir / "keywords.yaml").write_text(
        "include: [bioinformatics]\nexclude: []\noff_topic_titles: [sales]\n"
    )

    cfg = load_keywords(config_dir)

    assert len(cfg.off_topic_title_patterns) == 1
    assert cfg.off_topic_title_patterns[0].search("Sales Manager")
    assert not cfg.off_topic_title_patterns[0].search("Salesforce Engineer")


def test_off_topic_titles_default_to_empty(config_dir):
    (config_dir / "keywords.yaml").write_text("include: [bioinformatics]\nexclude: []\n")

    assert load_keywords(config_dir).off_topic_title_patterns == []
