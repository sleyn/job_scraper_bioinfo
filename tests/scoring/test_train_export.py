from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pytest
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from job_scraper.config import load_scoring_config
from job_scraper.scoring import train_export

JD_TEMPLATE = """---
title: {title}
company: {company}
---

Body of {title}. This is the text the regressor is meant to be fitted on.
"""


def _write_jd(corpus_dir, name, title="Some Role", company="Acme"):
    jd_dir = corpus_dir / name
    jd_dir.mkdir(parents=True)
    (jd_dir / "jd.md").write_text(JD_TEMPLATE.format(title=title, company=company))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """A hand-scored corpus shaped like the real one, including the rows that have
    broken real runs: a score of N/A, and a name with no directory behind it."""
    corpus_dir = tmp_path / "job_descriptions"
    corpus_dir.mkdir()

    _write_jd(corpus_dir, "2026-01-01_First", title="First Role")
    _write_jd(corpus_dir, "2026-01-02_Unscored", title="Unscored Role")
    _write_jd(corpus_dir, "2026-01-04_Last", title="Last Role")

    (corpus_dir / "scores.csv").write_text(
        "Name,Score\n"
        "2026-01-01_First,80\n"
        "2026-01-02_Unscored,N/A\n"  # scored N/A: dropped even though its jd.md exists
        "2026-01-03_Typo,55\n"  # no such directory: skipped, and named in the warning
        "2026-01-04_Last,40\n"
    )
    monkeypatch.setenv("JOB_HELPER_JD_SCORES_CSV", str(corpus_dir / "scores.csv"))
    return corpus_dir


@pytest.fixture
def cfg(corpus, config_dir):
    return load_scoring_config(config_dir)


def test_strips_frontmatter_and_keeps_body(cfg):
    descriptions, _ = train_export._load_training_data(cfg)

    assert not any(d.startswith("---") for d in descriptions)
    assert "title:" not in descriptions[0]
    assert descriptions[0].startswith("Body of First Role.")


def test_descriptions_stay_aligned_with_their_own_scores(cfg):
    descriptions, scores = train_export._load_training_data(cfg)

    # Two rows between First and Last drop out; Last must keep its own 40, not slide
    # onto the skipped row's 55.
    assert len(descriptions) == len(scores) == 2
    assert "First Role" in descriptions[0]
    assert "Last Role" in descriptions[1]
    np.testing.assert_array_equal(scores, [80, 40])


def test_na_scored_row_is_dropped(cfg):
    descriptions, _ = train_export._load_training_data(cfg)

    assert not any("Unscored Role" in d for d in descriptions)


def test_row_with_no_jd_is_skipped_and_named(cfg, capsys):
    train_export._load_training_data(cfg)

    printed = capsys.readouterr().out
    assert "Skipped 1 scored row(s)" in printed
    assert "2026-01-03_Typo" in printed
    # An N/A row is not a skip — it never had a score to lose.
    assert "2026-01-02_Unscored" not in printed


def test_nothing_printed_when_every_row_resolves(cfg, corpus, capsys):
    (corpus / "scores.csv").write_text("Name,Score\n2026-01-01_First,80\n")

    train_export._load_training_data(cfg)

    assert "Skipped" not in capsys.readouterr().out


def test_raises_when_jd_scores_csv_unset(cfg, monkeypatch, config_dir):
    monkeypatch.delenv("JOB_HELPER_JD_SCORES_CSV")
    cfg_without_csv = load_scoring_config(config_dir)

    with pytest.raises(ValueError, match="JOB_HELPER_JD_SCORES_CSV"):
        train_export._load_training_data(cfg_without_csv)


def test_raises_when_no_row_resolves_to_a_jd(cfg, corpus):
    (corpus / "scores.csv").write_text("Name,Score\n2026-01-03_Typo,55\n")

    # Returning empty arrays here would fail later inside StandardScaler, far from
    # the cause; the loader is the only place that still knows the corpus is bad.
    with pytest.raises(ValueError, match="No usable training JDs"):
        train_export._load_training_data(cfg)


# --- _spearman_scorer -------------------------------------------------------------


def _score_without_warnings(y_true, y_pred):
    """Returns (score, warning category names) so tests can assert on both."""
    import warnings

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = train_export._spearman_scorer(y_true, y_pred)
    return score, [w.category.__name__ for w in caught]


def test_constant_predictions_score_zero():
    score, warned = _score_without_warnings(np.arange(1.0, 6.0), np.full(5, 7.0))

    assert score == 0.0
    assert warned == []


def test_collapsed_predictions_score_zero_without_warning():
    # The shape a real degenerate NuSVR trial produces: every prediction identical, but
    # np.std() returns ~1e-14 rather than 0 because the mean of large-ish equal values
    # carries float error. A std-based guard misses this and lets spearmanr warn.
    y_pred = np.full(12, 63.77719053400675)
    assert np.std(y_pred) != 0, "fixture no longer reproduces the float-error case"
    assert (y_pred == y_pred[0]).all()

    score, warned = _score_without_warnings(np.arange(1.0, 13.0), y_pred)

    assert score == 0.0
    assert warned == []


def test_constant_targets_score_zero():
    score, warned = _score_without_warnings(np.full(5, 3.0), np.arange(1.0, 6.0))

    assert score == 0.0
    assert warned == []


def test_nan_predictions_score_zero():
    y_pred = np.array([1.0, 2.0, np.nan, 4.0, 5.0])

    score, warned = _score_without_warnings(np.arange(1.0, 6.0), y_pred)

    assert score == 0.0
    assert warned == []


def test_normal_predictions_score_scipy_rho():
    from scipy.stats import spearmanr

    y_true = np.arange(1.0, 6.0)
    y_pred = np.array([2.0, 1.0, 4.0, 3.0, 5.0])

    score, warned = _score_without_warnings(y_true, y_pred)

    assert score == pytest.approx(spearmanr(y_true, y_pred).statistic)
    assert warned == []


# --- _build_estimator -------------------------------------------------------------


def _build(regressor, params):
    return train_export._build_estimator(regressor, optuna.trial.FixedTrial(params))


def test_builds_ridge_with_tuned_alpha():
    estimator = _build("ridge", {"alpha": 0.25})

    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 0.25


def test_builds_elasticnet_with_tuned_params():
    estimator = _build(
        "elasticnet", {"alpha": 0.5, "l1_ratio": 0.3, "max_iter": 5000}
    )

    assert isinstance(estimator, ElasticNet)
    assert (estimator.alpha, estimator.l1_ratio, estimator.max_iter) == (0.5, 0.3, 5000)


def test_builds_nusvr_with_rbf_gamma():
    estimator = _build(
        "nusvr", {"C": 10.0, "nu": 0.4, "kernel": "rbf", "gamma": 0.01}
    )

    assert isinstance(estimator, NuSVR)
    assert (estimator.C, estimator.nu, estimator.kernel) == (10.0, 0.4, "rbf")
    assert estimator.gamma == 0.01


def test_linear_nusvr_does_not_suggest_gamma():
    trial = optuna.trial.FixedTrial({"C": 10.0, "nu": 0.4, "kernel": "linear"})

    estimator = train_export._build_estimator("nusvr", trial)

    assert estimator.kernel == "linear"
    # gamma is meaningless for a linear kernel, so it must not enter the search space:
    # a FixedTrial without it would raise if the code asked.
    assert "gamma" not in trial.params
    assert estimator.gamma == "scale"


def test_unknown_regressor_raises():
    with pytest.raises(ValueError, match="Unknown regressor"):
        _build("randomforest", {})


# --- train_export -----------------------------------------------------------------

N_FEATURES = 5


@pytest.fixture
def big_corpus(tmp_path, monkeypatch):
    """A corpus large enough for the 5-fold CV inside _tune()."""
    corpus_dir = tmp_path / "job_descriptions"
    corpus_dir.mkdir()

    rows = ["Name,Score"]
    for i in range(12):
        name = f"2026-02-{i + 1:02d}_Company"
        _write_jd(corpus_dir, name, title=f"Role {i}")
        rows.append(f"{name},{30 + i * 5}")

    (corpus_dir / "scores.csv").write_text("\n".join(rows) + "\n")
    monkeypatch.setenv("JOB_HELPER_JD_SCORES_CSV", str(corpus_dir / "scores.csv"))
    return corpus_dir


@pytest.fixture
def fake_features():
    """Stands in for build_features() so no sentence-transformer is ever loaded."""
    with patch("job_scraper.scoring.train_export.build_features") as mock:
        mock.side_effect = lambda descriptions, cfg: np.linspace(
            0.0, 1.0, len(descriptions) * N_FEATURES
        ).reshape(len(descriptions), N_FEATURES)
        yield mock


@pytest.mark.parametrize("regressor", ["ridge", "elasticnet", "nusvr"])
def test_exported_artifacts_reload_and_predict(
    regressor, fake_features, big_corpus, config_dir
):
    train_export.train_export(config_dir, regressor, n_trials=1)
    cfg = load_scoring_config(config_dir)

    # The production load path: score_postings() does exactly this, and nothing
    # validates the result, so a mismatch here surfaces as wrong scores, not an error.
    scaler = joblib.load(cfg.scaler_path)
    model = joblib.load(cfg.regressor_path)
    predictions = model.predict(scaler.transform(np.zeros((3, N_FEATURES))))

    assert predictions.shape == (3,)
    assert fake_features.called, "a real embedding model was loaded"


def test_creates_missing_artifact_directories(fake_features, big_corpus, config_dir):
    cfg = load_scoring_config(config_dir)
    assert not cfg.regressor_path.parent.exists()

    train_export.train_export(config_dir, "ridge", n_trials=1)

    assert cfg.regressor_path.is_file()
    assert cfg.scaler_path.is_file()


def test_returns_the_studys_best_value(fake_features, big_corpus, config_dir):
    study = SimpleNamespace(best_params={"alpha": 0.5}, best_value=0.4242)

    with patch("job_scraper.scoring.train_export._tune", return_value=study):
        assert train_export.train_export(config_dir, "ridge", n_trials=1) == 0.4242


def test_writes_only_inside_the_temp_config_dir(fake_features, big_corpus, config_dir):
    train_export.train_export(config_dir, "ridge", n_trials=1)
    cfg = load_scoring_config(config_dir)

    # The suite must never overwrite the repo's own exported production artifacts.
    assert config_dir.parent in cfg.regressor_path.parents
    assert config_dir.parent in cfg.scaler_path.parents


def test_regressor_is_fitted_on_scaled_features(fake_features, big_corpus, config_dir):
    """The exported pair must agree on whether features are scaled.

    score_postings() always predicts through scaler.transform(). If the regressor was
    fitted on raw features instead, every prediction is quietly wrong by the scaler's
    affine factor — and because scaling preserves ordering, the postings still rank
    plausibly. Only the magnitudes are wrong, which is why this needs pinning.
    """
    study = SimpleNamespace(best_params={"alpha": 0.1}, best_value=0.5)
    with patch("job_scraper.scoring.train_export._tune", return_value=study):
        train_export.train_export(config_dir, "ridge", n_trials=1)

    cfg = load_scoring_config(config_dir)
    scaler = joblib.load(cfg.scaler_path)
    model = joblib.load(cfg.regressor_path)

    descriptions, y = train_export._load_training_data(cfg)
    X = fake_features.side_effect(descriptions, cfg)
    expected = Ridge(alpha=0.1).fit(StandardScaler().fit_transform(X), y)

    probe = np.linspace(-1.0, 2.0, 3 * N_FEATURES).reshape(3, N_FEATURES)
    np.testing.assert_allclose(
        model.predict(scaler.transform(probe)),
        expected.predict(scaler.transform(probe)),
    )
