from __future__ import annotations

import hashlib
import json
import re

import joblib
import numpy as np
from sentence_transformers import SentenceTransformer

from job_scraper.config import ScoringConfig

_model_cache: dict[tuple[str, str | None], SentenceTransformer] = {}

# The locally-resolved commit hash `transformers` bakes into the dynamic module path
# it generates for `trust_remote_code` modelling code, e.g.
# "transformers_modules.nomic-ai.nomic-bert-2048.<hash>.modeling_hf_nomic_bert".
_MODELING_REVISION_RE = re.compile(r"^[0-9a-f]{8,64}$")


class ArtifactMismatchError(RuntimeError):
    """Raised when the exported regressor/scaler don't match the currently configured
    embedding setup — e.g. scoring.yaml's revision was edited without re-running
    train_export, or the trust_remote_code modelling module resolved to a different
    local cache entry than the one the artifacts were fitted against."""


def load_embedding_model(model_name: str, revision: str | None = None) -> SentenceTransformer:
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


def _encode(
    model: SentenceTransformer, texts: list[str], batch_size: int = 8
) -> np.ndarray:
    """Embeds `texts`, capping how many are held on the accelerator at once.

    Batch size is a memory knob, not a modelling one: attention cost grows with the
    square of sequence length, and this model takes 8192 tokens, so a batch of long
    postings is what exhausts a GPU rather than the number of postings overall. It does
    not affect the vectors — batching only groups the same forward passes — so it is
    safe to change without retraining. sentence-transformers' own default of 32 OOMs on
    Apple MPS against real postings, which run to ~5k tokens.
    """
    # Uniform "search_document:" prefix so JD, resume, and MEMORY.md section
    # embeddings all live in one consistent (symmetric-similarity) space.
    return np.asarray(
        model.encode(
            ["search_document: " + t for t in texts],
            normalize_embeddings=True,
            batch_size=batch_size,
        )
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

    job_history_emb = _encode(model, mem_sections, cfg.embedding_batch_size)
    resume_emb = _encode(model, [resume_text], cfg.embedding_batch_size)[0]

    cfg.reference_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cfg.reference_cache_path,
        job_history_emb=job_history_emb,
        resume_emb=resume_emb,
        key=cache_key,
    )
    return job_history_emb, resume_emb


def _extract_modeling_revision(module_path: str) -> str | None:
    """Pulls the locally-resolved commit hash for `trust_remote_code` modelling code out
    of its dynamically generated module path. SentenceTransformer's `revision=` pin only
    reaches the model weights' own repo; the custom architecture code
    (`nomic-ai/nomic-bert-2048` for this model) comes from a separate repo pinned by
    nothing but `HF_HUB_OFFLINE=1` and cache contents — this hash is the only record of
    which version actually loaded."""
    for part in module_path.split("."):
        if _MODELING_REVISION_RE.fullmatch(part):
            return part
    return None


def model_fingerprint(model: SentenceTransformer, cfg: ScoringConfig) -> dict[str, str | None]:
    """Identifies the exact embedding setup `model` resolved to: the configured model
    name/revision plus the locally-resolved weights and modelling-code commit hashes.
    Compared against what train_export.py recorded, to catch a feature-space change the
    exported regressor/scaler never saw."""
    auto_model = model[0].auto_model
    weights_config = getattr(auto_model, "config", None)
    return {
        "embedding_model": cfg.embedding_model,
        "embedding_model_revision": cfg.embedding_model_revision,
        "weights_commit_hash": getattr(weights_config, "_commit_hash", None),
        "modeling_code_revision": _extract_modeling_revision(type(auto_model).__module__),
    }


def save_fingerprint(fingerprint: dict[str, str | None], cfg: ScoringConfig) -> None:
    cfg.fingerprint_path.parent.mkdir(parents=True, exist_ok=True)
    cfg.fingerprint_path.write_text(json.dumps(fingerprint, indent=2, sort_keys=True))


def _require_artifacts(cfg: ScoringConfig) -> None:
    missing = [p for p in (cfg.regressor_path, cfg.scaler_path) if not p.is_file()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise FileNotFoundError(
            f"Scoring model artifact(s) not found: {names}. Run "
            "`python -m job_scraper.scoring.train_export` to produce them."
        )


def _check_fingerprint(model: SentenceTransformer, cfg: ScoringConfig) -> None:
    if not cfg.fingerprint_path.is_file():
        raise ArtifactMismatchError(
            f"No model fingerprint recorded at {cfg.fingerprint_path}. The exported "
            "regressor/scaler predate fingerprint tracking. Re-run "
            "`python -m job_scraper.scoring.train_export` to regenerate them together "
            "with a fingerprint."
        )
    recorded = json.loads(cfg.fingerprint_path.read_text())
    current = model_fingerprint(model, cfg)
    mismatched = {k: (recorded.get(k), v) for k, v in current.items() if recorded.get(k) != v}
    if mismatched:
        detail = "; ".join(f"{k}: exported={old!r}, now={new!r}" for k, (old, new) in mismatched.items())
        raise ArtifactMismatchError(
            "Scoring model artifacts were exported against a different embedding setup "
            f"than is currently configured ({detail}). Re-run "
            "`python -m job_scraper.scoring.train_export` to re-export against the "
            "current setup, or revert the config/cache change that caused this."
        )


def build_features(descriptions: list[str], cfg: ScoringConfig) -> np.ndarray:
    """Embeds `descriptions` and appends experience-fit features against MEMORY.md
    and the resume. Column layout: [768 embedding dims, fit_max, fit_top3, resume_cos] —
    must match the layout train_export.py fits the regressor/scaler on."""
    model = load_embedding_model(cfg.embedding_model, cfg.embedding_model_revision)
    job_history_emb, resume_emb = _load_reference_embeddings(model, cfg)

    jd_emb = _encode(model, descriptions, cfg.embedding_batch_size)

    # All embeddings are L2-normalized, so dot product == cosine similarity.
    job_history_sims = jd_emb @ job_history_emb.T  # (n_jd, n_sections)
    fit_max = job_history_sims.max(axis=1)
    fit_top3 = np.sort(job_history_sims, axis=1)[:, -3:].mean(axis=1)
    resume_cos = jd_emb @ resume_emb

    return np.hstack([jd_emb, fit_max[:, None], fit_top3[:, None], resume_cos[:, None]])


def score_postings(url_to_description: dict[str, str], cfg: ScoringConfig) -> dict[str, float]:
    if not url_to_description:
        return {}

    _require_artifacts(cfg)

    urls = list(url_to_description.keys())
    descriptions = [url_to_description[u] for u in urls]

    model = load_embedding_model(cfg.embedding_model, cfg.embedding_model_revision)
    _check_fingerprint(model, cfg)

    X = build_features(descriptions, cfg)
    scaler = joblib.load(cfg.scaler_path)
    regressor = joblib.load(cfg.regressor_path)
    predictions = regressor.predict(scaler.transform(X))

    return dict(zip(urls, (float(p) for p in predictions)))
