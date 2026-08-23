from __future__ import annotations

import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


def _expand_path(raw: str) -> Path:
    """Expand ~ and $ENV_VARS in a path string."""
    return Path(os.path.expanduser(os.path.expandvars(raw)))


def _required_env_path(var: str, points_at: str) -> Path:
    """Read a required career-history path from its own env var, failing loudly here
    rather than as a FileNotFoundError deep inside embedding_scorer.py. An empty value
    counts as unset, since .env.example ships these vars blank for the user to fill in."""
    raw = os.environ.get(var, "").strip()
    if not raw:
        raise KeyError(
            f"{var} is not set. It must point at {points_at} "
            f"(machine-specific, see .env.example)."
        )
    return _expand_path(raw)


@dataclass
class CompanyEntry:
    company_name: str
    ats_type: str
    board_identifier: str
    tenant: str | None
    careers_url: str
    notes: str


@dataclass
class KeywordConfig:
    include_patterns: list[re.Pattern]
    exclude_patterns: list[re.Pattern]
    match_fields: list[str]


@dataclass
class ScoringConfig:
    embedding_model: str
    memory_path: Path
    resume_path: Path
    reference_cache_path: Path
    regressor_path: Path
    scaler_path: Path
    jd_scores_csv: Path | None
    jd_dir: Path | None = None
    embedding_model_revision: str | None = None
    embedding_batch_size: int = 8


def load_companies(config_dir: Path | str) -> list[CompanyEntry]:
    path = Path(config_dir) / "companies.csv"
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [
            CompanyEntry(
                company_name=row["company_name"],
                ats_type=row["ats_type"],
                board_identifier=row["board_identifier"],
                tenant=row["tenant"] or None,
                careers_url=row["careers_url"],
                notes=row["notes"],
            )
            for row in reader
        ]


def load_keywords(config_dir: Path | str) -> KeywordConfig:
    path = Path(config_dir) / "keywords.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    flags = 0 if raw.get("case_sensitive", False) else re.IGNORECASE
    # Require word boundaries around each pattern so short acronyms (e.g. "NGS", "STAR")
    # don't match as a substring inside unrelated words (e.g. "savings", "Started").
    include_patterns = [re.compile(r"\b(?:" + p + r")\b", flags) for p in raw.get("include", [])]
    exclude_patterns = [re.compile(r"\b(?:" + p + r")\b", flags) for p in raw.get("exclude", [])]
    match_fields = raw.get("match_fields", ["title", "description"])

    return KeywordConfig(
        include_patterns=include_patterns,
        exclude_patterns=exclude_patterns,
        match_fields=match_fields,
    )


def load_settings(config_dir: Path | str) -> dict:
    path = Path(config_dir) / "settings.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def load_scoring_config(config_dir: Path | str) -> ScoringConfig:
    config_dir = Path(config_dir)
    path = config_dir / "scoring.yaml"
    with open(path) as f:
        raw = yaml.safe_load(f)

    # YAML paths are relative to the repo root (config_dir's parent). The three
    # career-history files live outside this repo (in AI_Job_Helper), so each is
    # resolved from its own env var — this repo never assumes that project's layout.
    repo_root = config_dir.parent
    jd_scores_csv = os.environ.get("JOB_HELPER_JD_SCORES_CSV", "").strip()
    jd_dir = os.environ.get("JOB_HELPER_JD_DIR", "").strip()

    return ScoringConfig(
        embedding_model=raw["embedding_model"],
        memory_path=_required_env_path(
            "JOB_HELPER_MEMORY_PATH", "your career-history MEMORY.md"
        ),
        resume_path=_required_env_path("JOB_HELPER_RESUME_PATH", "your resume"),
        reference_cache_path=repo_root / raw["reference_cache_path"],
        regressor_path=repo_root / raw["regressor_path"],
        scaler_path=repo_root / raw["scaler_path"],
        jd_scores_csv=_expand_path(jd_scores_csv) if jd_scores_csv else None,
        # The directory holding one <name>/jd.md per scored row. Its own var rather
        # than the CSV's parent: the two need not sit together, and deriving one from
        # the other is exactly the layout assumption ADR-0001 rules out.
        jd_dir=_expand_path(jd_dir) if jd_dir else None,
        # Optional: absent means "whatever revision resolves", which is what the
        # exported regressor silently depends on if nobody pins it.
        embedding_model_revision=raw.get("embedding_model_revision"),
        embedding_batch_size=raw.get("embedding_batch_size", 8),
    )
