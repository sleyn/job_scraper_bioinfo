import marimo

__generated_with = "0.23.9"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Score downloaded job descriptions

    I have a set of job descriptions that were scored against my job history using Claude. The idea is to use this set as a training data for a fast NLP model that could score downloaded job descriptions on the fly.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _():
    import os
    import pandas as pd
    from pathlib import Path
    import re

    # AI_JOB_HELPER_ROOT points at the sibling AI_Job_Helper project (resume + job-history
    # notes); set it in .env and export before running (see README.md "JD scoring").
    AI_JOB_HELPER_ROOT = Path(os.path.expanduser(os.environ["AI_JOB_HELPER_ROOT"]))
    JD_PATH = AI_JOB_HELPER_ROOT / 'job_descriptions'
    JD_SCORES_TABLE = Path('job_socres.csv')
    return AI_JOB_HELPER_ROOT, JD_PATH, JD_SCORES_TABLE, Path, pd, re


@app.cell
def _(Path, pd):
    def read_jd_from_file(jd_path: Path) -> str:
        jd_content = ""

        if Path.exists:
            with open(jd_path, 'r') as jd_fh:
                jd_content = jd_fh.read()

        return jd_content

    def collect_jd_from_path(jd_dir_path: Path, score_file: Path) -> pd.DataFrame:
        # Read score table
        score_tbl = pd.read_csv(jd_dir_path / score_file).dropna(subset=['Score'])

        # Add JD
        score_tbl['JD'] = score_tbl['Name'].apply(lambda jd_folder: read_jd_from_file(jd_dir_path / jd_folder / "jd.md"))

        return score_tbl

    return (collect_jd_from_path,)


@app.cell
def _(JD_PATH, JD_SCORES_TABLE, collect_jd_from_path, re):
    jd_header_re = re.compile(r'^---\n(\w+: [^\n]+\n)+---\n+', flags = re.MULTILINE)

    jd_table_raw = collect_jd_from_path(JD_PATH, JD_SCORES_TABLE). \
        drop(columns=["Name"]). \
        assign(JD = lambda jd: jd['JD'].str.replace(jd_header_re, "", regex=True))
    return (jd_table_raw,)


@app.cell
def _(jd_table_raw):
    jd_table_raw
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preprocessing

    The main idea for preprocessing is to keep only **Responsibilities** and **Job Requirements** paragraphs. To do this I will use two techniques:
    1) Use TextTiling algorithm for splitting JD into paragraphs.
    2) Use `AhmedBou/JoBert` BERT model from `Hugging Face` to classify each paragraph into categories:
       - 'About the Company'
       - 'Job Description'
       - 'Job Requirements'
       - 'Responsibilities'
       - 'Benefits'
       - 'Other'
    3) Keep only 'Job Requirements' and 'Responsibilities' and merge all paragraphs with these tags.

    To stanrdize the tokenization:
    1) English stop words would be removed
    2) the PorterStemmer will be applied
    """)
    return


@app.cell
def _(jd_table_raw):
    import torch
    import string
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    from nltk.tokenize import TextTilingTokenizer, word_tokenize
    from nltk.stem import PorterStemmer
    from nltk import download as nltk_downloader
    from sklearn.base import TransformerMixin

    nltk_downloader('stopwords')
    nltk_downloader('punkt_tab')

    from nltk.corpus import stopwords

    rr_classifier_tokenizer = AutoTokenizer.from_pretrained("AhmedBou/JoBert")
    rr_classifier_model = AutoModelForSequenceClassification.from_pretrained("AhmedBou/JoBert")

    class TonkenizeJD():
        def __init__(self):
            self.tt = TextTilingTokenizer(w=20, k=10)
            self.ps = PorterStemmer()
            self.stopwords = set(stopwords.words('english'))

        @staticmethod
        def classify_paragraph(jd_text_paragraph):
            LABEL_NAMES = ['About the Company', 'Job Description', 'Job Requirements', 'Responsibilities', 'Benefits', 'Other']

            inference_inputs = rr_classifier_tokenizer(
                jd_text_paragraph,
                return_tensors='pt',
                truncation=True,
                max_length=rr_classifier_tokenizer.model_max_length
            )
            inference_inputs = {key: val for key, val in inference_inputs.items()}
            inference_outputs = rr_classifier_model(**inference_inputs)
            inference_logits = inference_outputs.logits
            inference_prediction = torch.argmax(inference_logits).item()
            inference_label_name = LABEL_NAMES[inference_prediction]
            return inference_label_name

        def extract_responsibilities_requirements(self, jdtext):
            """
            Extract Responsibilities and Requirements paragraphs

            Input:
                - JD text

            Output:
                - Text with only text classified as Responsibilities and Requirements
            """
            jd_paragraphs = self.tt.tokenize(jdtext)

            rr_paragraph_indexes = [
                i for i, label in enumerate(list(map(self.classify_paragraph, jd_paragraphs))) 
                if label in ["Responsibilities", "Job Requirements"]
            ]

            return "\n\n".join([jd_paragraphs[i] for i in rr_paragraph_indexes])

        def tokenize(self, jdtext):
            # Extract only Responsibilities and Job Requirements
            jdtext_rr = self.extract_responsibilities_requirements(jdtext)

            # Tokenize
            tokens = word_tokenize(jdtext_rr.lower())

            # Remove stop words
            filtered_tokens = [
                word 
                for word in tokens 
                if word not in self.stopwords
            ]

            # Remove punctuation
            filtered_tokens = [
                word
                for word in filtered_tokens
                if word not in string.punctuation
            ]

            return filtered_tokens

    # Precompute tokenization
    jd_table_raw["tokens"] = jd_table_raw["JD"].apply(lambda text: TonkenizeJD().tokenize(text))

    # Remove items that failed tokenization
    # The tokenization failing because some JD do not have a pargraph structure or Responsibilities and Requirements rubrics
    jd_table = jd_table_raw[jd_table_raw["tokens"].str.len() > 0]
    return (jd_table,)


@app.cell
def _(jd_table):
    jd_table
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Create a preprocessing pipeline for the Cross Validation. Use `CountVectorizer` for one hot encoding and `TfidfTransformer` to scale token impact.
    """)
    return


@app.cell
def _():
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge, ElasticNet
    from sklearn.svm import NuSVR
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.metrics import make_scorer
    from catboost import CatBoostRegressor
    import numpy as np
    import optuna

    from scipy.stats import spearmanr

    return (
        CatBoostRegressor,
        CountVectorizer,
        ElasticNet,
        KFold,
        NuSVR,
        Pipeline,
        Ridge,
        TfidfTransformer,
        cross_val_score,
        make_scorer,
        np,
        optuna,
        spearmanr,
    )


@app.cell
def _(CountVectorizer, Pipeline, TfidfTransformer, make_scorer, np, spearmanr):
    def spearman_scorer(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        if np.std(y_pred) == 0 or np.std(y_true) == 0:
            return 0.0
        correlation, p_value = spearmanr(y_true, y_pred)
        return 0.0 if np.isnan(correlation) else correlation

    spearman_score = make_scorer(spearman_scorer, greater_is_better=True)

    def preprocess_tokens(trial):
        # Count vectorizer
        params_count_vectorizer = {
            "min_df": trial.suggest_int("min_df", 1, 5),
            "max_df": trial.suggest_float("max_df", 0.8, 1.0),
            "max_features": trial.suggest_int('max_features', 100, 3000, step=100),
            "ngram_range": (1, trial.suggest_int('max_ngram', 1, 2))
        }

        vectorizer = CountVectorizer(
            tokenizer=lambda tokens: tokens,
            preprocessor=lambda x: x,
            token_pattern=None,
            binary=True,
            **params_count_vectorizer
        )

        # Feature transformation
        params_tfidf = {
            "use_idf": trial.suggest_categorical("use_idf", [True, False]),
            "sublinear_tf": trial.suggest_categorical("sublinear_tf", [True, False])
        }

        tfidf_transformer = TfidfTransformer(
            norm="l2",
            **params_tfidf
        )

        return Pipeline([
            ("vectorize", vectorizer),
            ("tfidf", tfidf_transformer)
        ])

    return preprocess_tokens, spearman_score, spearman_scorer


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Regression models

    For regression fit use **Ridge regression**, **Elastic Net**, **Nu Support Vector Regression** and **CatBoost**.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ridge Reression
    """)
    return


@app.cell
def _(
    KFold,
    Pipeline,
    Ridge,
    cross_val_score,
    jd_table,
    np,
    optuna,
    preprocess_tokens,
    spearman_score,
):
    def ridge_objective(trial):
        preprocessor = preprocess_tokens(trial)

        # Ridge regression
        params_ridge = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)
        }

        ridge_reqressor = Ridge(
            **params_ridge
        )

        # Full pipeline
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", ridge_reqressor)
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=jd_table["tokens"],
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring = spearman_score
        )

        return np.mean(cv_results)

    study_ridge = optuna.create_study(direction="maximize")
    study_ridge.optimize(ridge_objective, n_trials=50)
    return (study_ridge,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elastic Net
    """)
    return


@app.cell
def _(
    ElasticNet,
    KFold,
    Pipeline,
    cross_val_score,
    jd_table,
    np,
    optuna,
    preprocess_tokens,
    spearman_score,
):
    def elastic_net_objective(trial):
        preprocessor = preprocess_tokens(trial)

        # Ridge regression
        params_en = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000, step=1000)
        }

        en_reqressor = ElasticNet(
            **params_en
        )

        # Full pipeline
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", en_reqressor)
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=jd_table["tokens"],
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring = spearman_score
        )

        return np.mean(cv_results)

    study_elastic_net = optuna.create_study(direction="maximize")
    study_elastic_net.optimize(elastic_net_objective, n_trials=50)
    return (study_elastic_net,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Support Vector Regression
    """)
    return


@app.cell
def _(
    KFold,
    NuSVR,
    Pipeline,
    cross_val_score,
    jd_table,
    np,
    optuna,
    preprocess_tokens,
    spearman_score,
):
    def nusvr_objective(trial):
        preprocessor = preprocess_tokens(trial)

        # Ridge regression
        params_nusvr = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "nu": trial.suggest_float("nu", 0.01, 0.8),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"])
        }

        if params_nusvr["kernel"] == "rbf":
            params_nusvr["gamma"] = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

        nu_svr = NuSVR(
            **params_nusvr
        )

        # Full pipeline
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", nu_svr)
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=jd_table["tokens"],
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring = spearman_score
        )

        return np.mean(cv_results)

    study_nusvr = optuna.create_study(direction="maximize")
    study_nusvr.optimize(nusvr_objective, n_trials=50)
    return (study_nusvr,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### CatBoost
    """)
    return


@app.cell
def _(
    CatBoostRegressor,
    KFold,
    Pipeline,
    cross_val_score,
    jd_table,
    np,
    optuna,
    preprocess_tokens,
    spearman_score,
):
    def catboost_objective(trial):
        preprocessor = preprocess_tokens(trial)

        # Ridge regression
        params_cat_boost = {
            "iterations": trial.suggest_int("iterations", 100, 400, step=50),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 2, 5),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
            "random_strength": trial.suggest_float("random_strength", 1e-3, 10.0, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "loss_function": "RMSE",
            "verbose": 0,
            "random_seed": 42
        }

        cat_boost = CatBoostRegressor(
            **params_cat_boost
        )

        # Full pipeline
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", cat_boost)
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=jd_table["tokens"],
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring = spearman_score
        )

        return np.mean(cv_results)

    study_cat_boost = optuna.create_study(direction="maximize")
    study_cat_boost.optimize(catboost_objective, n_trials=25)
    return (study_cat_boost,)


@app.cell
def _(pd, study_cat_boost, study_elastic_net, study_nusvr, study_ridge):
    pd.DataFrame(
        {
            "nuSVR": [study_nusvr.best_value],
            "Ridge": [study_ridge.best_value],
            "ElasticNet": [study_elastic_net.best_value],
            "CatBoost": [study_cat_boost.best_value]
        },
        index=["Spearman"]
    ).T
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Embedding-based model (nomic-embed-text-v1.5)

    Alternative to the BoW / TF-IDF pipeline above. Instead of sparse token counts, embed the
    **whole** job description with `nomic-ai/nomic-embed-text-v1.5` (768-dim, 8192-token context),
    so no paragraph extraction / stemming is needed. We also add a **resume-fit** feature:
    the cosine similarity between each JD embedding and the embedding of a single *canonical*
    (non-company-tailored) resume — this directly encodes "fit against my history", which is what
    the target score measures. Using the generic resume (not the per-company tailored ones) avoids
    leaking the target.

    The same `spearman_score` / 5-fold `KFold(shuffle, random_state=0)` harness is reused, so these
    numbers line up directly with the BoW studies above.
    """)
    return


@app.cell
def _(AI_JOB_HELPER_ROOT, Path, jd_table, np, re):
    from sentence_transformers import SentenceTransformer
    import hashlib

    # Experience anchors (non-company-tailored -> no target leakage).
    # MEMORY.md is the ground-truth job-history knowledge base AND the source the
    # JD scores were labeled against; the generic resume is kept for comparison.
    _MEMORY_PATH = AI_JOB_HELPER_ROOT / "reference" / "MEMORY.md"
    _RESUME_PATH = (
        AI_JOB_HELPER_ROOT / "resumes" / "Semion_Leyn_Resume_Generic_2026-07-23.md"
    )
    _EMB_CACHE = Path(__file__).parent / "jd_embeddings_nomic.npz"

    embedding_model = SentenceTransformer(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )

    def split_sections(md_text):
        # Split MEMORY.md on markdown H2/H3 headers, keeping each header with its body.
        # Chunking avoids diluting a 7k-token doc into one washed-out centroid vector.
        parts = re.split(r"^(?=#{2,3}\s)", md_text, flags=re.MULTILINE)
        return [p.strip() for p in parts if len(p.strip()) > 40]

    jd_texts = jd_table["JD"].tolist()
    resume_text = _RESUME_PATH.read_text()
    mem_sections = split_sections(_MEMORY_PATH.read_text())

    # Cache key: hash of every input, so any edit invalidates the cache
    cache_key = hashlib.sha256(
        (
            "\x00".join(jd_texts)
            + "\x01" + resume_text
            + "\x02" + "\x03".join(mem_sections)
        ).encode("utf-8")
    ).hexdigest()

    # Uniform "search_document:" prefix on JD, resume, and sections so all cosines
    # live in one consistent (symmetric-similarity) space.
    def encode_text_to_embedding(model, texts):
        return np.asarray(
            model.encode(
                ["search_document: " + t for t in texts],
                normalize_embeddings=True,
                show_progress_bar=True,
            )
        )

    def compute_search_bundle():
        return (
            encode_text_to_embedding(embedding_model, jd_texts),
            encode_text_to_embedding(embedding_model, [resume_text])[0],
            encode_text_to_embedding(embedding_model, mem_sections),
        )

    if _EMB_CACHE.exists():
        cached = np.load(_EMB_CACHE, allow_pickle=True)
        if str(cached["key"]) == cache_key:
            jd_emb, resume_emb, job_history_emb = (
                cached["jd_emb"], cached["resume_emb"], cached["job_history_emb"]
            )
        else:
            jd_emb, resume_emb, job_history_emb = compute_search_bundle()
            np.savez(
                _EMB_CACHE, jd_emb=jd_emb, resume_emb=resume_emb,
                job_history_emb=job_history_emb, key=cache_key,
            )
    else:
        jd_emb, resume_emb, job_history_emb = compute_search_bundle()
        np.savez(
            _EMB_CACHE, jd_emb=jd_emb, resume_emb=resume_emb,
            job_history_emb=job_history_emb, key=cache_key,
        )
    return jd_emb, job_history_emb, resume_emb


@app.cell
def _(jd_emb, job_history_emb, np, resume_emb):
    # All embeddings L2-normalized, so dot product == cosine similarity.
    job_history_sims = jd_emb @ job_history_emb.T          # (n_jd, n_sections)
    fit_max = job_history_sims.max(axis=1)         # best-matching experience section
    fit_top3 = np.sort(job_history_sims, axis=1)[:, -3:].mean(axis=1)  # mean of top-3 sections
    resume_cos = jd_emb @ resume_emb        # single-vector generic-resume cosine

    X_emb = np.hstack([
        jd_emb,
        fit_max[:, None],
        fit_top3[:, None],
        resume_cos[:, None],
    ])
    X_emb.shape
    return X_emb, fit_max, fit_top3, resume_cos


@app.cell
def _():
    from sklearn.preprocessing import StandardScaler

    return (StandardScaler,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Ridge (embeddings)
    """)
    return


@app.cell
def _(
    KFold,
    Pipeline,
    Ridge,
    StandardScaler,
    X_emb,
    cross_val_score,
    jd_table,
    np,
    optuna,
    spearman_score,
):
    def ridge_emb_objective(trial):
        params_ridge = {"alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True)}

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", Ridge(**params_ridge)),
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=X_emb,
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring=spearman_score,
        )

        return np.mean(cv_results)

    study_ridge_emb = optuna.create_study(direction="maximize")
    study_ridge_emb.optimize(ridge_emb_objective, n_trials=50)
    return (study_ridge_emb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Elastic Net (embeddings)
    """)
    return


@app.cell
def _(
    ElasticNet,
    KFold,
    Pipeline,
    StandardScaler,
    X_emb,
    cross_val_score,
    jd_table,
    np,
    optuna,
    spearman_score,
):
    def elasticnet_emb_objective(trial):
        params_en = {
            "alpha": trial.suggest_float("alpha", 1e-4, 1e2, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.01, 0.99),
            "max_iter": trial.suggest_int("max_iter", 1000, 10000, step=1000),
        }

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", ElasticNet(**params_en)),
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=X_emb,
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring=spearman_score,
        )

        return np.mean(cv_results)

    study_elastic_net_emb = optuna.create_study(direction="maximize")
    study_elastic_net_emb.optimize(elasticnet_emb_objective, n_trials=50)
    return (study_elastic_net_emb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Support Vector Regression (embeddings)
    """)
    return


@app.cell
def _(
    KFold,
    NuSVR,
    Pipeline,
    StandardScaler,
    X_emb,
    cross_val_score,
    jd_table,
    np,
    optuna,
    spearman_score,
):
    def nusvr_emb_objective(trial):
        params_nusvr = {
            "C": trial.suggest_float("C", 1e-2, 1e3, log=True),
            "nu": trial.suggest_float("nu", 0.01, 0.8),
            "kernel": trial.suggest_categorical("kernel", ["linear", "rbf"]),
        }

        if params_nusvr["kernel"] == "rbf":
            params_nusvr["gamma"] = trial.suggest_float("gamma", 1e-4, 1e1, log=True)

        pipe = Pipeline([
            ("scale", StandardScaler()),
            ("model", NuSVR(**params_nusvr)),
        ])

        cv_results = cross_val_score(
            estimator=pipe,
            X=X_emb,
            y=jd_table["Score"],
            cv=KFold(n_splits=5, shuffle=True, random_state=0),
            scoring=spearman_score,
        )

        return np.mean(cv_results)

    study_nusvr_emb = optuna.create_study(direction="maximize")
    study_nusvr_emb.optimize(nusvr_emb_objective, n_trials=50)
    return (study_nusvr_emb,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Zero-shot baselines

    Spearman of each experience-fit column **alone** (no training) — reference floors, and a check on
    which anchor carries the most signal: MEMORY.md section max-cosine, MEMORY.md top-3 mean, and the
    single-vector generic-resume cosine.
    """)
    return


@app.cell
def _(fit_max, fit_top3, jd_table, pd, resume_cos, spearman_scorer):
    zero_shot_table = pd.Series(
        {
            "MEMORY.md max-cosine": spearman_scorer(jd_table["Score"], fit_max),
            "MEMORY.md top3-mean": spearman_scorer(jd_table["Score"], fit_top3),
            "generic resume cosine": spearman_scorer(jd_table["Score"], resume_cos),
        },
        name="zero-shot Spearman",
    )
    # Primary anchor (MEMORY.md max-cosine) feeds the comparison table below
    zero_shot_spearman = zero_shot_table["MEMORY.md max-cosine"]
    zero_shot_table
    return (zero_shot_spearman,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### BoW vs embeddings — Spearman comparison
    """)
    return


@app.cell
def _(
    pd,
    study_cat_boost,
    study_elastic_net,
    study_elastic_net_emb,
    study_nusvr,
    study_nusvr_emb,
    study_ridge,
    study_ridge_emb,
    zero_shot_spearman,
):
    comparison = pd.DataFrame(
        {
            "BoW / TF-IDF": {
                "Ridge": study_ridge.best_value,
                "ElasticNet": study_elastic_net.best_value,
                "nuSVR": study_nusvr.best_value,
                "CatBoost": study_cat_boost.best_value,
                "zero-shot cosine": float("nan"),
            },
            "nomic embeddings": {
                "Ridge": study_ridge_emb.best_value,
                "ElasticNet": study_elastic_net_emb.best_value,
                "nuSVR": study_nusvr_emb.best_value,
                "CatBoost": float("nan"),
                "zero-shot cosine": zero_shot_spearman,
            },
        }
    )
    comparison
    return


if __name__ == "__main__":
    app.run()
