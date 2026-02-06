#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIRP REVIEW PROJECT — Topic Modeling Pipeline
============================================

CIRP BERTopic v12.0 (BALANCED, R-ready)
---------------------------------------
Design goals for CIRP Annals + CIRP JMST (2000–2025), research papers only:
- Topic granularity: manageable & interpretable (no fixed target nr_topics; "natural" from UMAP+HDBSCAN)
- Outliers: aim ~10% (controlled, conservative; never force to 0)
- Quality metrics: Coherence (C_v), Topic Diversity, Topic Count, Outlier %, Stability (ARI bootstrap)
- Topic-over-time RAW exports for R (streamgraph / joyplot): long + full grid + wide (counts/shares)
- Robust across BERTopic versions: NO fragile imports (e.g., SentenceTransformerBackend)

Run outputs:
- results/run_YYYYMMDD_HHMMSS/*
- figures/run_YYYYMMDD_HHMMSS/*
- results/run_YYYYMMDD_HHMMSS.zip

Notes:
- If you already have a research-only CSV, the filter keeps only "article" if a document-type column exists.
- Outlier tuning is optional and does NOT change the number of topics; it reassigns only noise points (-1)
  using probabilities and/or c-TF-IDF in a controlled way to land inside a target band.

Author: ChatGPT (project assistance)
"""

from __future__ import annotations

import os
import re
import json
import random
import zipfile
import warnings
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer

from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

from umap import UMAP
from hdbscan import HDBSCAN
import hdbscan  # approximate_predict

from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


# -----------------------------
# CONFIG
# -----------------------------
@dataclass
class Config:
    # Reproducibility
    seed: int = 76

    # Data
    data_path: str = "data/CIRP_researchonly.csv"
    out_root_results: str = "results"
    out_root_figures: str = "figures"

    # Column candidates (robust to naming differences)
    title_col_candidates: Tuple[str, ...] = ("Title", "title")
    abstract_col_candidates: Tuple[str, ...] = ("Abstract", "abstract")
    year_col_candidates: Tuple[str, ...] = ("Year", "year", "Publication Year", "pubyear")
    doi_col_candidates: Tuple[str, ...] = ("DOI", "doi")
    doc_type_col_candidates: Tuple[str, ...] = ("Document Type", "document_type", "doctype", "DocumentType")
    source_col_candidates: Tuple[str, ...] = ("Source title", "source_title", "Source Title", "journal")
    issue_col_candidates: Tuple[str, ...] = ("Issue", "issue")

    # Filters
    keep_document_types: Tuple[str, ...] = ("article",)   # research papers
    exclude_annals_issue2: bool = False  # set True only if your CSV includes Issue/Source columns & you need this

    # Text preprocessing
    min_chars: int = 50
    # Optional lemmatization (falls back gracefully if spaCy unavailable)
    use_spacy_lemmatization: bool = True
    spacy_model: str = "en_core_web_sm"

    # Embeddings
    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    max_seq_length: int = 512
    normalize_embeddings: bool = True
    batch_size_gpu: int = 64
    batch_size_cpu: int = 32
    cache_dir: str = "data/cache_embeddings"

    # UMAP (granularity driver)
    umap_n_neighbors: int = 30
    umap_n_components: int = 5
    umap_min_dist: float = 0.05
    umap_metric: str = "cosine"

    # HDBSCAN (granularity + outliers)
    hdb_min_cluster_size: int = 25
    hdb_min_samples: int = 2
    hdb_cluster_selection_method: str = "eom"  # "eom" tends to fewer, larger clusters than "leaf"

    # Vectorizer / c-TF-IDF (interpretability driver)
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int = 5
    max_df: float = 0.35
    max_features: int = 20000
    top_n_words: int = 10

    # c-TF-IDF weights
    bm25_weighting: bool = True
    reduce_frequent_words: bool = True

    # Domain stopwords (representation only; does not affect clustering)
    use_domain_stopwords: bool = True

    # Outlier control
    do_outlier_reduction: bool = True
    target_outlier_low: float = 8.0
    target_outlier_high: float = 12.0
    outlier_strategies: Tuple[str, ...] = ("probabilities", "c-tf-idf")  # conservative
    candidate_thresholds: Tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)

    # Metrics
    diversity_topk: int = 10
    coherence_dictionary_filter: bool = True
    coherence_no_below: int = 5
    coherence_no_above: float = 0.50

    compute_stability: bool = True
    stability_bootstrap_iters: int = 8
    stability_sample_frac: float = 0.80

    # Topic-over-time exports
    export_topic_over_time_raw: bool = True

    # Zip outputs
    create_zip: bool = True


CFG = Config()


# -----------------------------
# Utilities
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def pick_col(df: pd.DataFrame, candidates: Tuple[str, ...], required: bool = True) -> Optional[str]:
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    if required:
        raise KeyError(f"Missing required column. Tried {candidates}. Available columns sample: {list(df.columns)[:25]}")
    return None


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def token_pattern_for_engineering() -> str:
    # Keeps alphanumerics + symbols common in manufacturing terms (e.g., Ti-6Al-4V, LPBF, µm, etc.)
    # Requires at least one letter to reduce numeric-only tokens.
    return r"(?u)\b(?=\w*[A-Za-z])[\w\-\+\.]{2,}\b"


def build_domain_stopwords() -> set[str]:
    # Keep it conservative: remove cross-topic boilerplate terms that inflate overlap.
    return {
        "paper","study","research","work","result","results","method","methods","approach","approaches",
        "analysis","performance","experimental","experiment","experiments","developed","based","investigated",
        "demonstrated","different","proposed","propose","proposes","using","use","used","show","shows","shown",
        "new","novel","application","applications",
        "manufacturing","manufacture","manufactured","production","produce","process","processes","processing",
        "system","systems","model","models",
        "cirp","elsevier","ltd","author","authors","rights","reserved","published",
        "et","al",
    }


def outlier_percent(topics: List[int] | np.ndarray) -> float:
    t = np.asarray(topics)
    return float((t == -1).mean() * 100.0)


def safe_write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def zip_run(results_dir: Path, figures_dir: Path, zip_path: Path) -> None:
    files: List[Path] = []
    for base in (results_dir, figures_dir):
        if base.exists():
            files.extend([p for p in base.rglob("*") if p.is_file()])
    if not files:
        # Ensure zip isn't empty even if something went wrong late
        (results_dir / "README_EMPTY_ZIP.txt").write_text(
            "No files found at zip-time. Check console log for earlier errors.\n",
            encoding="utf-8"
        )
        files = [results_dir / "README_EMPTY_ZIP.txt"]

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            arc = f.relative_to(results_dir.parent)
            zf.write(f, arcname=str(arc))


# -----------------------------
# Optional spaCy lemmatization
# -----------------------------
def maybe_lemmatize(texts: List[str]) -> List[str]:
    if not CFG.use_spacy_lemmatization:
        return texts
    try:
        import spacy
        nlp = spacy.load(CFG.spacy_model, disable=["parser", "ner", "textcat"])
        nlp.max_length = 2_000_000
        out = []
        for doc in nlp.pipe(texts, batch_size=64):
            lemmas = []
            for tok in doc:
                if tok.is_space or tok.is_punct:
                    continue
                lemma = tok.lemma_.lower().strip()
                if lemma:
                    lemmas.append(lemma)
            out.append(" ".join(lemmas))
        return out
    except Exception as e:
        print(f"⚠️ spaCy lemmatization skipped: {e}")
        return texts


# -----------------------------
# Metrics
# -----------------------------
def compute_diversity(topic_words: List[List[str]], topk: int) -> float:
    if not topic_words:
        return 0.0
    flat = [w for tw in topic_words for w in tw[:topk] if w]
    denom = len(topic_words) * topk
    return float(len(set(flat)) / denom) if denom else 0.0


def compute_coherence_cv(tokenized_docs: List[List[str]], topic_words: List[List[str]]) -> float:
    if not topic_words:
        return 0.0
    dictionary = Dictionary(tokenized_docs)
    if CFG.coherence_dictionary_filter:
        dictionary.filter_extremes(no_below=CFG.coherence_no_below, no_above=CFG.coherence_no_above)
        dictionary.compactify()
    cm = CoherenceModel(topics=topic_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_v")
    return float(cm.get_coherence())


def compute_stability_ari(reduced_embeddings: np.ndarray, topics: np.ndarray, hdb_template: HDBSCAN) -> float:
    from sklearn.metrics import adjusted_rand_score

    n = reduced_embeddings.shape[0]
    if n < 200:
        return 0.0

    base = np.asarray(topics)
    mask_base = base != -1
    if mask_base.mean() < 0.5:
        mask_base = np.ones_like(mask_base, dtype=bool)

    rng = np.random.default_rng(CFG.seed)
    aris: List[float] = []
    for _ in range(CFG.stability_bootstrap_iters):
        idx = rng.choice(n, size=int(CFG.stability_sample_frac * n), replace=False)
        clusterer = HDBSCAN(
            min_cluster_size=hdb_template.min_cluster_size,
            min_samples=hdb_template.min_samples,
            metric=hdb_template.metric,
            cluster_selection_method=hdb_template.cluster_selection_method,
            prediction_data=True,
        )
        clusterer.fit(reduced_embeddings[idx])
        labels_b, _ = hdbscan.approximate_predict(clusterer, reduced_embeddings)

        mask = mask_base & (labels_b != -1)
        if mask.sum() < 200:
            continue
        aris.append(adjusted_rand_score(base[mask], labels_b[mask]))

    return float(np.mean(aris)) if aris else 0.0


# -----------------------------
# Outlier tuning (controlled)
# -----------------------------
def reduce_outliers_to_target(
    topic_model: BERTopic,
    docs: List[str],
    topics: List[int],
    probs: Optional[np.ndarray],
    embeddings: np.ndarray
) -> Tuple[List[int], Dict[str, Any]]:
    report: Dict[str, Any] = {"initial_outlier_pct": outlier_percent(topics), "steps": []}
    cur = list(topics)

    if not CFG.do_outlier_reduction or (-1 not in set(cur)):
        report["final_outlier_pct"] = outlier_percent(cur)
        return cur, report

    low, high = CFG.target_outlier_low, CFG.target_outlier_high

    for strat in CFG.outlier_strategies:
        if outlier_percent(cur) <= high:
            break

        results: List[Tuple[float, float, List[int]]] = []
        for thr in CFG.candidate_thresholds:
            if strat == "probabilities":
                if probs is None:
                    continue
                new_t = topic_model.reduce_outliers(docs, cur, probabilities=probs, strategy=strat, threshold=thr)
            elif strat == "c-tf-idf":
                new_t = topic_model.reduce_outliers(docs, cur, strategy=strat, threshold=thr)
            elif strat == "embeddings":
                new_t = topic_model.reduce_outliers(docs, cur, strategy=strat, embeddings=embeddings, threshold=thr)
            else:
                raise ValueError(strat)

            pct = outlier_percent(new_t)
            results.append((float(thr), float(pct), list(new_t)))

        if not results:
            continue

        # Prefer solutions inside range; otherwise choose closest above high (conservative)
        in_range = [r for r in results if low <= r[1] <= high]
        if in_range:
            best = min(in_range, key=lambda x: abs(x[1] - high))
        else:
            above = [r for r in results if r[1] > high]
            best = min(above, key=lambda x: x[1]) if above else min(results, key=lambda x: abs(x[1] - high))

        thr, pct, new_t = best
        report["steps"].append({"strategy": strat, "threshold": thr, "outlier_pct": pct})
        cur = new_t

    report["final_outlier_pct"] = outlier_percent(cur)
    return cur, report


# -----------------------------
# Topic-over-time RAW exports
# -----------------------------
def export_topics_over_time_raw(years: List[int], topics: List[int], out_dir: Path) -> None:
    tmp = pd.DataFrame({"year": years, "topic": topics})

    counts = tmp.groupby(["year", "topic"], as_index=False).size().rename(columns={"size": "count"})
    totals = tmp.groupby("year", as_index=False).size().rename(columns={"size": "total_year"})
    long = counts.merge(totals, on="year", how="left")
    long["share"] = long["count"] / long["total_year"]
    long = long.sort_values(["year", "topic"]).reset_index(drop=True)
    long.to_csv(out_dir / "06_Topics_Over_Time_Raw_Long.csv", index=False)

    years_u = sorted(tmp["year"].unique().tolist())
    topics_u = sorted(tmp["topic"].unique().tolist())
    grid = pd.MultiIndex.from_product([years_u, topics_u], names=["year", "topic"]).to_frame(index=False)
    full = grid.merge(long, on=["year", "topic"], how="left").fillna({"count": 0, "total_year": 0, "share": 0.0})
    full.to_csv(out_dir / "06a_Topics_Over_Time_Raw_FullGrid.csv", index=False)

    wide_counts = full.pivot_table(index="year", columns="topic", values="count", fill_value=0).reset_index()
    wide_counts.to_csv(out_dir / "06b_Topics_Over_Time_WideCount.csv", index=False)

    wide_shares = full.pivot_table(index="year", columns="topic", values="share", fill_value=0.0).reset_index()
    wide_shares.to_csv(out_dir / "06c_Topics_Over_Time_WideShare.csv", index=False)


# -----------------------------
# Main
# -----------------------------
def main() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    set_seed(CFG.seed)

    run_id = now_tag()
    results_dir = Path(CFG.out_root_results) / f"run_{run_id}"
    figures_dir = Path(CFG.out_root_figures) / f"run_{run_id}"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("CIRP BERTopic v12.0 — BALANCED & R-READY")
    print(f"Seed: {CFG.seed}")
    print(f"Output: {results_dir}")
    print("=" * 80)

    df = pd.read_csv(CFG.data_path)

    title_col = pick_col(df, CFG.title_col_candidates)
    abstract_col = pick_col(df, CFG.abstract_col_candidates)
    year_col = pick_col(df, CFG.year_col_candidates)
    doi_col = pick_col(df, CFG.doi_col_candidates, required=False)
    doc_type_col = pick_col(df, CFG.doc_type_col_candidates, required=False)
    source_col = pick_col(df, CFG.source_col_candidates, required=False)
    issue_col = pick_col(df, CFG.issue_col_candidates, required=False)

    if doc_type_col is not None:
        before = len(df)
        df = df[df[doc_type_col].astype(str).str.lower().isin(CFG.keep_document_types)].copy()
        print(f"🧹 Filter articles: kept {len(df)}/{before}")

    if CFG.exclude_annals_issue2 and (source_col is not None) and (issue_col is not None):
        is_annals = df[source_col].astype(str).str.strip().str.lower().eq("cirp annals")
        before = len(df)
        df = df[~is_annals | df[issue_col].isna() | (df[issue_col].astype(str) == "1")].copy()
        if len(df) != before:
            print(f"🧹 Exclude Annals Issue 2: kept {len(df)}/{before}")

    # Build text
    df[title_col] = df[title_col].fillna("")
    df[abstract_col] = df[abstract_col].fillna("")
    df[year_col] = pd.to_numeric(df[year_col], errors="coerce").astype("Int64")
    df = df[df[year_col].notna()].copy()
    df[year_col] = df[year_col].astype(int)

    df["raw_text"] = (df[title_col].astype(str) + ". " + df[abstract_col].astype(str)).apply(clean_text)
    df = df[df["raw_text"].str.len() >= CFG.min_chars].copy()

    years = df[year_col].tolist()
    print(f"✅ Docs: {len(df)} | Years: {min(years)}–{max(years)}")

    # Prepare text for embeddings and for c-TF-IDF
    docs_embed = df["raw_text"].tolist()
    docs_ctfidf = [d.lower() for d in docs_embed]

    # Optional lemmatization for c-TF-IDF only (keeps embeddings on raw text)
    if CFG.use_spacy_lemmatization:
        print("🧠 spaCy lemmatization (for vectorizer/c-TF-IDF only)...")
        docs_ctfidf = maybe_lemmatize(docs_ctfidf)

    # Vectorizer + stopwords (representation)
    stopwords = set(ENGLISH_STOP_WORDS)
    if CFG.use_domain_stopwords:
        stopwords |= build_domain_stopwords()

    vectorizer_model = CountVectorizer(
        stop_words=sorted(stopwords),
        ngram_range=CFG.ngram_range,
        min_df=CFG.min_df,
        max_df=CFG.max_df,
        max_features=CFG.max_features,
        token_pattern=token_pattern_for_engineering(),
        strip_accents="unicode",
    )

    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=CFG.bm25_weighting,
        reduce_frequent_words=CFG.reduce_frequent_words,
    )

    # Embeddings (cached)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size = CFG.batch_size_gpu if device == "cuda" else CFG.batch_size_cpu
    cache_dir = Path(CFG.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    emb_cache = cache_dir / f"emb_{CFG.embedding_model_name.replace('/', '_')}_v12_{len(docs_embed)}.npy"

    print(f"🚀 Embeddings: {CFG.embedding_model_name} | device={device} | batch={batch_size}")
    embedding_model = SentenceTransformer(CFG.embedding_model_name, device=device)
    try:
        embedding_model.max_seq_length = CFG.max_seq_length
    except Exception:
        pass

    if emb_cache.exists():
        embeddings = np.load(emb_cache)
        print(f"💾 Loaded embeddings cache: {emb_cache}")
    else:
        # SentenceTransformers API varies slightly across versions
        try:
            embeddings = embedding_model.encode(
                docs_embed,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=CFG.normalize_embeddings,
            )
        except TypeError:
            embeddings = embedding_model.encode(
                docs_embed,
                batch_size=batch_size,
                show_progress_bar=True,
                convert_to_numpy=True,
            )
            if CFG.normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
                embeddings = embeddings / norms
        np.save(emb_cache, embeddings)
        print(f"💾 Saved embeddings cache: {emb_cache}")

    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # UMAP + HDBSCAN
    umap_model = UMAP(
        n_neighbors=CFG.umap_n_neighbors,
        n_components=CFG.umap_n_components,
        min_dist=CFG.umap_min_dist,
        metric=CFG.umap_metric,
        random_state=CFG.seed,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=CFG.hdb_min_cluster_size,
        min_samples=CFG.hdb_min_samples,
        metric="euclidean",
        cluster_selection_method=CFG.hdb_cluster_selection_method,
        prediction_data=True,
    )

    # Fit (representation added AFTER outlier tuning for stability)
    topic_model = BERTopic(
        embedding_model=embedding_model,  # IMPORTANT: required for KeyBERTInspired update_topics
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=None,
        top_n_words=CFG.top_n_words,
        calculate_probabilities=True,
        verbose=True,
    )

    print("🧠 Fit BERTopic...")
    topics, probs = topic_model.fit_transform(docs_ctfidf, embeddings)

    init_out = outlier_percent(topics)
    init_topic_count = len([t for t in set(topics) if t != -1])
    print(f"📈 Initial: topics={init_topic_count}, outliers={init_out:.2f}%")

    # Outlier tuning (controlled)
    if CFG.do_outlier_reduction:
        topics, outlier_report = reduce_outliers_to_target(topic_model, docs_ctfidf, list(topics), probs, embeddings)
    else:
        outlier_report = {"initial_outlier_pct": init_out, "steps": [], "final_outlier_pct": init_out}

    print(f"✅ Final outliers: {outlier_percent(topics):.2f}% (target {CFG.target_outlier_low}–{CFG.target_outlier_high}%)")
    safe_write_json(results_dir / "09_Outlier_Tuning_Report.json", outlier_report)

    # Update topics representation (KeyBERTInspired + MMR)
    print("🧩 update_topics (KeyBERTInspired + MMR)...")
    representation_model = [KeyBERTInspired(random_state=CFG.seed), MaximalMarginalRelevance(diversity=0.35)]
    try:
        topic_model.update_topics(
            docs_ctfidf,
            topics=topics,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            top_n_words=CFG.top_n_words,
        )
    except Exception as e:
        # Do not fail the run if representation refinement has API/compat issues
        print(f"⚠️ update_topics failed (continuing): {e}")

    # Exports
    df_out = df.copy()
    df_out["topic"] = topics

    # Probability max (if available)
    try:
        probs_np = np.asarray(probs) if probs is not None else None
        if probs_np is not None and probs_np.ndim == 2 and probs_np.shape[0] == len(df_out):
            df_out["topic_prob_max"] = probs_np.max(axis=1)
        else:
            df_out["topic_prob_max"] = np.nan
    except Exception:
        df_out["topic_prob_max"] = np.nan

    keep_cols = [c for c in [doi_col, year_col, title_col] if c is not None and c in df_out.columns]
    keep_cols += ["topic", "topic_prob_max", "raw_text"]
    df_out[keep_cols].to_csv(results_dir / "01_Full_Document_Assignments.csv", index=False)
    df_out[df_out["topic"] == -1][keep_cols].to_csv(results_dir / "01b_Outlier_Documents.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(results_dir / "03_Topic_Metadata_Keywords.csv", index=False)

    # Summary stats
    topic_sizes = pd.Series(topics).value_counts(dropna=False).sort_index()
    stats = topic_sizes.rename("n_docs").reset_index().rename(columns={"index": "topic"})
    stats["pct_docs"] = stats["n_docs"] / len(topics) * 100.0
    stats.to_csv(results_dir / "04_Summary_Statistics.csv", index=False)

    # Metrics
    print("📈 Computing quality metrics...")
    analyzer = vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(d) for d in docs_ctfidf]

    topic_ids = [t for t in sorted(set(topics)) if t != -1]
    topic_words: List[List[str]] = []
    for tid in topic_ids:
        t = topic_model.get_topic(tid) or []
        topic_words.append([w for w, _ in t[:max(CFG.top_n_words, CFG.diversity_topk)]])

    coherence_cv = compute_coherence_cv(tokenized_docs, topic_words)
    diversity = compute_diversity(topic_words, topk=CFG.diversity_topk)

    stability_ari = 0.0
    if CFG.compute_stability:
        try:
            reduced = getattr(topic_model.umap_model, "embedding_", None)
            if isinstance(reduced, np.ndarray) and reduced.shape[0] == len(topics):
                stability_ari = compute_stability_ari(reduced, np.asarray(topics), hdbscan_model)
        except Exception as e:
            print(f"⚠️ Stability computation skipped: {e}")
            stability_ari = 0.0

    metrics = pd.DataFrame([
        {"metric": "topic_coherence_c_v", "value": coherence_cv},
        {"metric": "topic_diversity", "value": diversity},
        {"metric": "topic_count", "value": len(topic_ids)},
        {"metric": "outlier_pct", "value": outlier_percent(topics)},
        {"metric": "docs", "value": len(docs_ctfidf)},
        {"metric": "stability_ari_bootstrap", "value": stability_ari},
    ])
    metrics.to_csv(results_dir / "05_Quality_Metrics.csv", index=False)

    # Topic over time exports (RAW for R)
    if CFG.export_topic_over_time_raw:
        print("🕒 Export Topic Over Time (RAW for R)...")
        export_topics_over_time_raw(years=years, topics=list(topics), out_dir=results_dir)

    # Visualizations (best-effort)
    print("📊 Saving BERTopic visuals (best effort)...")
    try:
        topic_model.visualize_topics().write_html(str(figures_dir / "Viz_01_Intertopic_Distance_Map.html"))
    except Exception:
        pass
    try:
        topic_model.visualize_barchart(top_n_topics=40).write_html(str(figures_dir / "Viz_02_Topic_Word_Scores.html"))
    except Exception:
        pass
    try:
        topic_model.visualize_hierarchy().write_html(str(figures_dir / "Viz_03_Hierarchical_Clustering.html"))
    except Exception:
        pass
    try:
        topic_model.visualize_heatmap().write_html(str(figures_dir / "Viz_04_Topic_Similarity_Heatmap.html"))
    except Exception:
        pass

    # Save model (optional; can fail across versions)
    try:
        topic_model.save(
            results_dir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=CFG.embedding_model_name
        )
    except Exception:
        pass

    # Run report
    report = {
        "run_id": run_id,
        "docs": len(docs_ctfidf),
        "years": [min(years), max(years)],
        "topic_count": len(topic_ids),
        "outlier_pct": outlier_percent(topics),
        "coherence_c_v": coherence_cv,
        "diversity": diversity,
        "stability_ari_bootstrap": stability_ari,
        "embeddings_cache": str(emb_cache),
        "config": CFG.__dict__,
        "outlier_tuning": outlier_report,
    }
    safe_write_json(results_dir / "00_Run_Report.json", report)

    # Zip
    if CFG.create_zip:
        zip_path = results_dir.parent / f"run_{run_id}.zip"
        zip_run(results_dir, figures_dir, zip_path)
        print(f"📦 ZIP: {zip_path}")

    print("=" * 80)
    print("DONE ✅")
    print(f"Results: {results_dir}")
    print(f"Figures: {figures_dir}")
    print("Key metrics:")
    print(f"  topics:   {report['topic_count']}")
    print(f"  outliers: {report['outlier_pct']:.2f}% (target {CFG.target_outlier_low}-{CFG.target_outlier_high}%)")
    print(f"  C_v:      {report['coherence_c_v']:.3f}")
    print(f"  diversity:{report['diversity']:.3f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
