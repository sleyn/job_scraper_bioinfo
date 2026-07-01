from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import yaml


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
