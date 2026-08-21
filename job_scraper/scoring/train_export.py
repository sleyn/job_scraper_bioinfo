"""Fit and export the production JD-scoring regressor + scaler.

Trains on the hand-scored JD set pointed at by the JOB_HELPER_JD_SCORES_CSV env var,
using the same embedding_scorer.build_features() used at
inference time, so training and production features never drift apart. Hyperparameters
are picked via a short Optuna search (mirrors thinking_space/score_jd/score.py's
study_nusvr_emb / study_elastic_net_emb cells), then the final model is refit on the
full training set and joblib-dumped to the paths in scoring.yaml.

Usage: python -m job_scraper.scoring.train_export [--config-dir config] [--regressor nusvr] [--trials 30]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import make_scorer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import NuSVR

from job_scraper.config import ScoringConfig, load_scoring_config
from job_scraper.scoring.embedding_scorer import build_features

_HEADER_RE = re.compile(r"^---\n(\w+: [^\n]+\n)+---\n+", flags=re.MULTILINE)


def _load_training_data(cfg: ScoringConfig) -> tuple[list[str], np.ndarray]:
    if cfg.jd_scores_csv is None:
        raise ValueError(
            "JOB_HELPER_JD_SCORES_CSV is not set; it must point at the hand-scored JD CSV"
        )

    jd_dir = cfg.jd_scores_csv.parent
    table = pd.read_csv(cfg.jd_scores_csv).dropna(subset=["Score"])

    descriptions = [
        _HEADER_RE.sub("", (jd_dir / name / "jd.md").read_text())
        for name in table["Name"]
    ]
    return descriptions, table["Score"].to_numpy()


def _spearman_scorer(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if np.std(y_pred) == 0 or np.std(y_true) == 0:
        return 0.0
    correlation, _ = spearmanr(y_true, y_pred)
    return 0.0 if np.isnan(correlation) else correlation


_SPEARMAN_SCORE = make_scorer(_spearman_scorer, greater_is_better=True)


def _build_estimator(regressor: str, trial: optuna.trial.BaseTrial):
    if regressor == "ridge":
        return Ridge(alpha=trial.suggest_float("alpha", 1e-4, 1e2, log=True))
    if regressor == "elasticnet":
        return ElasticNet(
            alpha=trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            l1_ratio=trial.suggest_float("l1_ratio", 0.01, 0.99),
            max_iter=trial.suggest_int("max_iter", 1000, 10000, step=1000),
        )
    if regressor == "nusvr":
        params = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "nu": trial.suggest_float("nu", 0.01, 0.8),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }
        if params["kernel"] == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1e1, log=True)
        return NuSVR(**params)
    raise ValueError(f"Unknown regressor: {regressor!r}")


def _tune(regressor: str, X_scaled: np.ndarray, y: np.ndarray, n_trials: int) -> optuna.Study:
    def objective(trial: optuna.Trial) -> float:
        estimator = _build_estimator(regressor, trial)
        scores = cross_val_score(
            estimator, X_scaled, y,
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring=_SPEARMAN_SCORE,
        )
        return float(np.mean(scores))

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    return study


def train_export(config_dir: str | Path, regressor: str, n_trials: int) -> float:
    cfg = load_scoring_config(config_dir)
    descriptions, y = _load_training_data(cfg)

    print(f"Building features for {len(descriptions)} training JDs...")
    X = build_features(descriptions, cfg)

    scaler = StandardScaler().fit(X)
    X_scaled = scaler.transform(X)

    study = _tune(regressor, X_scaled, y, n_trials)
    print(f"Best {regressor} params: {study.best_params} (CV Spearman: {study.best_value:.3f})")

    final_model = _build_estimator(regressor, optuna.trial.FixedTrial(study.best_params))
    final_model.fit(X_scaled, y)

    cfg.regressor_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.scaler_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_model, cfg.regressor_path)
    joblib.dump(scaler, cfg.scaler_path)

    print(f"Exported {regressor} to {cfg.regressor_path} / {cfg.scaler_path}")
    return study.best_value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-dir", default="config")
    parser.add_argument("--regressor", default="nusvr", choices=["ridge", "elasticnet", "nusvr"])
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    train_export(args.config_dir, args.regressor, args.trials)
