from __future__ import annotations

import hashlib
import re

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from job_scraper.config import ScoringConfig

_model_cache: dict[tuple[str, str | None], SentenceTransformer] = {}


def _load_embedding_model(model_name: str, revision: str | None = None) -> SentenceTransformer:
    """Loads the embedding model, pinned to `revision` when scoring.yaml declares one.

    The revision matters more than it looks: the exported regressor and scaler are fitted
    on this model's output, so a new upstream revision changes the feature space out from
    under them with nothing in the pipeline to notice. Pinning makes that a deliberate
    change. Cache key includes the revision so two pins never collide in one process.
    """
    key = (model_name, revision)
    if key not in _model_cache:
        _model_cache[key] = SentenceTransformer(
            model_name, revision=revision, trust_remote_code=True
        )
    return _model_cache[key]


def _encode(model: SentenceTransformer, texts: list[str]) -> np.ndarray:
    # Uniform "search_document:" prefix so JD, resume, and MEMORY.md section
    # embeddings all live in one consistent (symmetric-similarity) space.
    return np.asarray(
        model.encode(["search_document: " + t for t in texts], normalize_embeddings=True)
    )


def _split_sections(md_text: str) -> list[str]:
    # Split MEMORY.md on markdown H2/H3 headers, keeping each header with its body.
    # Chunking avoids diluting a multi-thousand-token doc into one washed-out vector.
    parts = re.split(r"^(?=#{2,3}\s)", md_text, flags=re.MULTILINE)
    return [p.strip() for p in parts if len(p.strip()) > 40]


def _load_reference_embeddings(
    model: SentenceTransformer, cfg: ScoringConfig
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (job_history_emb, resume_emb) for MEMORY.md sections and the resume.

    Cached to cfg.reference_cache_path, keyed by a hash of both source files, so the
    cache is invalidated automatically whenever MEMORY.md or the resume changes.
    """
    resume_text = cfg.resume_path.read_text()
    mem_sections = _split_sections(cfg.memory_path.read_text())
    cache_key = hashlib.sha256(
        (resume_text + "\x00" + "\x01".join(mem_sections)).encode("utf-8")
    ).hexdigest()

    if cfg.reference_cache_path.exists():
        cached = np.load(cfg.reference_cache_path, allow_pickle=True)
        if str(cached["key"]) == cache_key:
            return cached["job_history_emb"], cached["resume_emb"]

    job_history_emb = _encode(model, mem_sections)
    resume_emb = _encode(model, [resume_text])[0]

    cfg.reference_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.reference_cache_path,
        job_history_emb=job_history_emb,
        resume_emb=resume_emb,
        key=cache_key,
    )
    return job_history_emb, resume_emb


def build_features(descriptions: list[str], cfg: ScoringConfig) -> np.ndarray:
    """Embeds `descriptions` and appends experience-fit features against MEMORY.md
    and the resume. Column layout: [768 embedding dims, fit_max, fit_top3, resume_cos] —
    must match the layout train_export.py fits the regressor/scaler on."""
    model = _load_embedding_model(cfg.embedding_model, cfg.embedding_model_revision)
    job_history_emb, resume_emb = _load_reference_embeddings(model, cfg)

    jd_emb = _encode(model, descriptions)

    # All embeddings are L2-normalized, so dot product == cosine similarity.
    job_history_sims = jd_emb @ job_history_emb.T  # (n_jd, n_sections)
    fit_max = job_history_sims.max(axis=1)
    fit_top3 = np.sort(job_history_sims, axis=1)[:, -3:].mean(axis=1)
    resume_cos = jd_emb @ resume_emb

    return np.hstack([jd_emb, fit_max[:, None], fit_top3[:, None], resume_cos[:, None]])


def score_postings(url_to_description: dict[str, str], cfg: ScoringConfig) -> dict[str, float]:
    if not url_to_description:
        return {}

    urls = list(url_to_description.keys())
    descriptions = [url_to_description[u] for u in urls]

    X = build_features(descriptions, cfg)
    scaler = joblib.load(cfg.scaler_path)
    regressor = joblib.load(cfg.regressor_path)
    predictions = regressor.predict(scaler.transform(X))

    return dict(zip(urls, (float(p) for p in predictions)))
