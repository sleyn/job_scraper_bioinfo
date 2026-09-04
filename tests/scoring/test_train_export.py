import json
import warnings
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import optuna
import pytest
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from job_scraper.config import load_scoring_config
from job_scraper.scoring import train_export

# The production artifacts, which no test may touch.
REPO_SCORING_DIR = Path(__file__).resolve().parents[2] / "config" / "scoring"

N_FEATURES = 5

JD_TEMPLATE = """---
title: {title}
company: {company}
---

Body of {title}. This is the text the regressor is meant to be fitted on.
"""


def _write_hand_scored_jds(rows, jd_dir):
    """Builds a hand-scored JD set at `jd_dir`: one directory per row holding a jd.md,
    plus the scores.csv beside them. A row given a title of None gets no directory,
    standing in for a CSV name with nothing behind it.
    """
    jd_dir.mkdir(parents=True, exist_ok=True)
    for name, _score, title in rows:
        if title is None:
            continue
        (jd_dir / name).mkdir(exist_ok=True)
        (jd_dir / name / "jd.md").write_text(
            JD_TEMPLATE.format(title=title, company="Acme")
        )

    csv_lines = ["Name,Score"] + [f"{name},{score}" for name, score, _ in rows]
    (jd_dir / "scores.csv").write_text("\n".join(csv_lines) + "\n")
    return jd_dir


@pytest.fixture
def jd_dir(tmp_path):
    """The directory conftest's career_history_env already points
    JOB_HELPER_JD_SCORES_CSV at, so no test re-sets that variable."""
    return tmp_path / "helper"


@pytest.fixture
def hand_scored_jds(jd_dir):
    """A hand-scored JD set shaped like the real one, including the rows that have
    broken real runs: a score of N/A, and a name with no directory behind it."""
    return _write_hand_scored_jds(
        [
            ("2026-01-01_First", "80", "First Role"),
            ("2026-01-02_Unscored", "N/A", "Unscored Role"),
            ("2026-01-03_Typo", "55", None),
            ("2026-01-04_Last", "40", "Last Role"),
        ],
        jd_dir,
    )


@pytest.fixture
def cfg(hand_scored_jds, config_dir):
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
    # An N/A row is not a skip - it never had a score to lose.
    assert "2026-01-02_Unscored" not in printed


def test_nothing_printed_when_every_row_resolves(cfg, jd_dir, capsys):
    _write_hand_scored_jds([("2026-01-01_First", "80", "First Role")], jd_dir)

    train_export._load_training_data(cfg)

    assert "Skipped" not in capsys.readouterr().out


def test_raises_when_jd_scores_csv_unset(monkeypatch, config_dir):
    monkeypatch.delenv("JOB_HELPER_JD_SCORES_CSV")

    with pytest.raises(ValueError, match="JOB_HELPER_JD_SCORES_CSV"):
        train_export._load_training_data(load_scoring_config(config_dir))


def test_raises_when_jd_dir_unset(hand_scored_jds, monkeypatch, config_dir):
    monkeypatch.delenv("JOB_HELPER_JD_DIR")

    with pytest.raises(ValueError, match="JOB_HELPER_JD_DIR"):
        train_export._load_training_data(load_scoring_config(config_dir))


def test_scores_csv_and_jd_dir_need_not_sit_together(tmp_path, monkeypatch, config_dir):
    """ADR-0001: this repo resolves each career-history path from its own env var and
    assumes nothing about how AI_Job_Helper arranges them. Deriving the JD directory
    from the CSV's parent would silently reintroduce that assumption."""
    jds = _write_hand_scored_jds(
        [("2026-01-01_First", "80", "First Role")], tmp_path / "somewhere" / "jds"
    )
    elsewhere = tmp_path / "unrelated" / "scores.csv"
    elsewhere.parent.mkdir(parents=True)
    elsewhere.write_text((jds / "scores.csv").read_text())
    monkeypatch.setenv("JOB_HELPER_JD_SCORES_CSV", str(elsewhere))
    monkeypatch.setenv("JOB_HELPER_JD_DIR", str(jds))

    descriptions, scores = train_export._load_training_data(
        load_scoring_config(config_dir)
    )

    assert "First Role" in descriptions[0]
    np.testing.assert_array_equal(scores, [80])


def test_raises_when_no_row_resolves_to_a_jd(cfg, jd_dir):
    _write_hand_scored_jds([("2026-01-03_Typo", "55", None)], jd_dir)

    # Returning empty arrays here would fail later inside StandardScaler, far from
    # the cause; the loader is the only place that still knows the set is unusable.
    with pytest.raises(ValueError, match="No usable training JDs"):
        train_export._load_training_data(cfg)


# --- _spearman_scorer -------------------------------------------------------------


def _score_and_warnings(y_true, y_pred):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        score = train_export._spearman_scorer(y_true, y_pred)
    return score, [w.category.__name__ for w in caught]


def test_constant_predictions_score_zero():
    score, warned = _score_and_warnings(np.arange(1.0, 6.0), np.full(5, 7.0))

    assert score == 0.0
    assert warned == []


def test_collapsed_predictions_score_zero_without_warning():
    # The shape a real degenerate NuSVR trial produces: every prediction identical, but
    # np.std() returns ~1e-14 rather than 0 because the mean of large-ish equal values
    # carries float error. A std-based guard misses this and lets spearmanr warn.
    y_pred = np.full(12, 63.77719053400675)
    assert np.std(y_pred) != 0, "fixture no longer reproduces the float-error case"
    assert (y_pred == y_pred[0]).all()

    score, warned = _score_and_warnings(np.arange(1.0, 13.0), y_pred)

    assert score == 0.0
    assert warned == []


def test_constant_targets_score_zero():
    score, warned = _score_and_warnings(np.full(5, 3.0), np.arange(1.0, 6.0))

    assert score == 0.0
    assert warned == []


def test_nan_predictions_score_zero():
    score, warned = _score_and_warnings(
        np.arange(1.0, 6.0), np.array([1.0, 2.0, np.nan, 4.0, 5.0])
    )

    assert score == 0.0
    assert warned == []


def test_empty_input_scores_zero():
    # The only case the first-element comparison cannot handle: it would IndexError.
    score, warned = _score_and_warnings(np.array([]), np.array([]))

    assert score == 0.0
    assert warned == []


def test_normal_predictions_score_scipy_rho():
    y_true = np.arange(1.0, 6.0)
    y_pred = np.array([2.0, 1.0, 4.0, 3.0, 5.0])

    score, warned = _score_and_warnings(y_true, y_pred)

    assert score == pytest.approx(spearmanr(y_true, y_pred).statistic)
    assert warned == []


def test_real_tuning_run_emits_no_warnings():
    """The end the guard exists for: a live search that produces degenerate trials.

    The unit tests above pin hand-built arrays; only this one proves the collapsed
    predictions a real NuSVR search generates never reach spearmanr.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(40, 10))
    y = rng.uniform(20, 95, size=40)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        train_export._tune("nusvr", X, y, n_trials=25)

    assert [w.category.__name__ for w in caught] == []


# --- _build_estimator -------------------------------------------------------------


def _build(regressor, params):
    return train_export._build_estimator(regressor, optuna.trial.FixedTrial(params))


def test_builds_ridge_with_tuned_alpha():
    estimator = _build("ridge", {"alpha": 0.25})

    assert isinstance(estimator, Ridge)
    assert estimator.alpha == 0.25


def test_builds_elasticnet_with_tuned_params():
    estimator = _build("elasticnet", {"alpha": 0.5, "l1_ratio": 0.3, "max_iter": 5000})

    assert isinstance(estimator, ElasticNet)
    assert (estimator.alpha, estimator.l1_ratio, estimator.max_iter) == (0.5, 0.3, 5000)


def test_builds_nusvr_with_rbf_gamma():
    estimator = _build("nusvr", {"C": 10.0, "nu": 0.4, "kernel": "rbf", "gamma": 0.01})

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


def _fake_feature_matrix(descriptions, _cfg=None):
    return np.linspace(0.0, 1.0, len(descriptions) * N_FEATURES).reshape(
        len(descriptions), N_FEATURES
    )


_FAKE_FINGERPRINT = {
    "embedding_model": "fake-model",
    "embedding_model_revision": None,
    "weights_commit_hash": "fakeweights",
    "modeling_code_revision": "fakemodelingcode",
}


@pytest.fixture
def exportable(jd_dir, config_dir):
    """Everything train_export() needs to run without a sentence-transformer: a JD set
    large enough for the 5-fold CV inside _tune(), and build_features()/the embedding
    model load stubbed out."""
    _write_hand_scored_jds(
        [
            (f"2026-02-{i + 1:02d}_Company", str(30 + i * 5), f"Role {i}")
            for i in range(12)
        ],
        jd_dir,
    )
    with (
        patch(
            "job_scraper.scoring.train_export.build_features",
            side_effect=_fake_feature_matrix,
        ) as build_features_stub,
        patch(
            "job_scraper.scoring.train_export.load_embedding_model",
            return_value=object(),
        ) as load_model_stub,
        patch(
            "job_scraper.scoring.train_export.model_fingerprint",
            return_value=_FAKE_FINGERPRINT,
        ) as fingerprint_stub,
    ):
        yield SimpleNamespace(
            config_dir=config_dir,
            build_features=build_features_stub,
            load_model=load_model_stub,
            model_fingerprint=fingerprint_stub,
        )


@pytest.mark.parametrize("regressor", ["ridge", "elasticnet", "nusvr"])
def test_exported_artifacts_reload_and_predict(regressor, exportable):
    train_export.train_export(exportable.config_dir, regressor, n_trials=1)
    cfg = load_scoring_config(exportable.config_dir)

    # The production load path: score_postings() does exactly this, and nothing
    # validates the result, so a mismatch here surfaces as wrong scores, not an error.
    scaler = joblib.load(cfg.scaler_path)
    model = joblib.load(cfg.regressor_path)
    predictions = model.predict(scaler.transform(np.zeros((3, N_FEATURES))))

    assert predictions.shape == (3,)
    assert exportable.build_features.called, "a real embedding model was loaded"


def test_creates_missing_artifact_directories(exportable):
    cfg = load_scoring_config(exportable.config_dir)
    assert not cfg.regressor_path.parent.exists()

    train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    assert cfg.regressor_path.is_file()
    assert cfg.scaler_path.is_file()


def test_writes_fingerprint_alongside_artifacts(exportable):
    cfg = load_scoring_config(exportable.config_dir)

    train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    assert json.loads(cfg.fingerprint_path.read_text()) == _FAKE_FINGERPRINT
    exportable.model_fingerprint.assert_called_once()


def test_fingerprint_reuses_the_model_build_features_already_loaded(exportable):
    """The embedding model load train_export needs for the fingerprint must be a cache
    hit off the one build_features() already triggered, not a second real load."""
    train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    exportable.load_model.assert_called_once()


def test_returns_the_studys_best_value(exportable):
    study = SimpleNamespace(best_params={"alpha": 0.5}, best_value=0.4242)

    with patch("job_scraper.scoring.train_export._tune", return_value=study):
        result = train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    assert result == 0.4242


def test_leaves_the_repos_own_artifacts_untouched(exportable):
    before = {p: p.stat().st_mtime_ns for p in REPO_SCORING_DIR.glob("*.joblib")}

    train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    cfg = load_scoring_config(exportable.config_dir)
    assert cfg.regressor_path.parent != REPO_SCORING_DIR
    assert cfg.fingerprint_path.parent != REPO_SCORING_DIR
    assert {p: p.stat().st_mtime_ns for p in REPO_SCORING_DIR.glob("*.joblib")} == before
    assert not (REPO_SCORING_DIR / "fingerprint.json").exists()


def test_regressor_is_fitted_on_scaled_features(exportable):
    """The exported pair must agree on whether features are scaled.

    score_postings() always predicts through scaler.transform(). If the regressor was
    fitted on raw features instead, every prediction is quietly wrong by the scaler's
    affine factor - and because scaling preserves ordering, the postings still rank
    plausibly. Only the magnitudes are wrong, which is why this needs pinning.
    """
    study = SimpleNamespace(best_params={"alpha": 0.1}, best_value=0.5)
    with patch("job_scraper.scoring.train_export._tune", return_value=study):
        train_export.train_export(exportable.config_dir, "ridge", n_trials=1)

    cfg = load_scoring_config(exportable.config_dir)
    scaler = joblib.load(cfg.scaler_path)
    model = joblib.load(cfg.regressor_path)

    descriptions, y = train_export._load_training_data(cfg)
    X = _fake_feature_matrix(descriptions)
    expected = Ridge(alpha=0.1).fit(StandardScaler().fit_transform(X), y)

    probe = np.linspace(-1.0, 2.0, 3 * N_FEATURES).reshape(3, N_FEATURES)
    np.testing.assert_allclose(
        model.predict(scaler.transform(probe)),
        expected.predict(scaler.transform(probe)),
    )
