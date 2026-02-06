#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CIRP BERTopic v7.1 — BALANCED
Fix: update_topics works with precomputed embeddings by keeping embedding_model backend.
Fix: exports + zip always happen (update_topics is wrapped).
"""

from __future__ import annotations

import json
import os
import re
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    RANDOM_SEED: int = 76

    DATA_PATH: str = "data/CIRP_researchonly.csv"
    TEXT_FIELDS: Tuple[str, ...] = ("Title", "Abstract")
    YEAR_COL: str = "Year"
    DOC_TYPE_COL: str = "Document Type"

    RESULTS_DIR: str = "results"
    FIGURES_DIR: str = "figures"

    KEEP_DOC_TYPES: Tuple[str, ...] = ("article",)
    EXCLUDE_DOC_TYPES: Tuple[str, ...] = ("review", "editorial", "erratum", "correction", "retracted", "note")

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMB_BATCH_SIZE: int = 64
    EMB_NORMALIZE: bool = True

    # UMAP
    UMAP_N_NEIGHBORS: int = 30
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.10
    UMAP_METRIC: str = "cosine"

    # HDBSCAN (moderato: non troppo rigido)
    HDB_MIN_CLUSTER_SIZE: int = 20
    HDB_MIN_SAMPLES: int = 2
    HDB_CLUSTER_SELECTION_METHOD: str = "eom"

    # Vectorizer (chiave per diversity/interpretability)
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    MIN_DF: int = 10
    MAX_DF: float = 0.35
    MAX_FEATURES: int = 20000

    TOP_N_WORDS: int = 10

    # c-TF-IDF
    BM25_WEIGHTING: bool = True
    REDUCE_FREQUENT_WORDS: bool = True

    # Outlier control
    ENABLE_OUTLIER_CONTROL: bool = True
    TARGET_OUTLIER_PCT_RANGE: Tuple[float, float] = (8.0, 15.0)
    OUTLIER_PROB_THRESHOLD: float = 0.05  # nel tuo log questo ha portato a ~13%

    # Representation
    USE_MMR: bool = True
    MMR_DIVERSITY: float = 0.30


def now_tag() -> str:
    return datetime.utcnow().strftime("%Y%m%d_%H%M%S")


def safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def normalize_doctype(x) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


def basic_clean(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text


def build_text(row: pd.Series, fields: Tuple[str, ...]) -> str:
    parts = []
    for f in fields:
        v = row.get(f, "")
        if pd.notna(v):
            parts.append(str(v))
    return basic_clean(" . ".join(parts))


def zip_folder(folder: Path, zip_path: Path) -> None:
    files = [p for p in folder.rglob("*") if p.is_file()]
    if not files:
        readme = folder / "README_EMPTY_ZIP.txt"
        readme.write_text("Run folder contained no files at zip-time. Check earlier errors.\n", encoding="utf-8")
        files = [readme]
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files:
            z.write(f, arcname=str(f.relative_to(folder)))


def compute_topic_diversity(topic_words: Dict[int, List[str]], top_n: int) -> float:
    topics = [t for t in topic_words.keys() if t != -1]
    if not topics:
        return 0.0
    words = []
    for t in topics:
        words.extend([str(x) for x in (topic_words.get(t, [])[:top_n]) if x])
    if not words:
        return 0.0
    return len(set(words)) / (len(topics) * top_n)


def topics_over_time_raw(assignments: pd.DataFrame, year_col: str, topic_col: str = "Topic") -> pd.DataFrame:
    df = assignments[[year_col, topic_col]].copy().dropna()
    df[year_col] = df[year_col].astype(int)

    years = sorted(df[year_col].unique())
    topics = sorted(df[topic_col].unique())

    counts = df.groupby([year_col, topic_col]).size().reset_index(name="Count")
    full = pd.MultiIndex.from_product([years, topics], names=[year_col, topic_col]).to_frame(index=False)
    full = full.merge(counts, on=[year_col, topic_col], how="left")
    full["Count"] = full["Count"].fillna(0).astype(int)

    year_tot = full.groupby(year_col)["Count"].transform("sum").replace(0, np.nan)
    full["Share"] = (full["Count"] / year_tot).fillna(0.0)
    return full


def wide_from_long(full: pd.DataFrame, year_col: str, topic_col: str, value_col: str) -> pd.DataFrame:
    wide = full.pivot(index=year_col, columns=topic_col, values=value_col).fillna(0.0).sort_index()
    wide.columns = [f"T{int(c)}" for c in wide.columns]
    return wide.reset_index()


def main():
    import warnings
    warnings.filterwarnings("ignore")

    np.random.seed(Config.RANDOM_SEED)

    run = f"run_{now_tag()}"
    results_dir = Path(Config.RESULTS_DIR) / run
    figures_dir = Path(Config.FIGURES_DIR) / run
    safe_mkdir(results_dir)
    safe_mkdir(figures_dir)

    print("=" * 80)
    print("CIRP BERTopic v7.1 BALANCED (fix update_topics + zip)")
    print(f"Seed: {Config.RANDOM_SEED}")
    print(f"Output: {results_dir}")
    print("=" * 80)

    # -------------------------
    # Load + filter
    # -------------------------
    df = pd.read_csv(Config.DATA_PATH)
    print(f"📂 Loading: {Config.DATA_PATH}")

    if Config.DOC_TYPE_COL in df.columns:
        dt = df[Config.DOC_TYPE_COL].apply(normalize_doctype)
        keep_mask = dt.isin(set(Config.KEEP_DOC_TYPES)) & (~dt.isin(set(Config.EXCLUDE_DOC_TYPES)))
        df = df.loc[keep_mask].copy()
        print(f"🧹 Filter articles: kept {len(df)}/{len(dt)}")

    df["text"] = df.apply(lambda r: build_text(r, Config.TEXT_FIELDS), axis=1)
    df = df[df["text"].str.len() > 0].copy()

    df[Config.YEAR_COL] = pd.to_numeric(df[Config.YEAR_COL], errors="coerce")
    df = df.dropna(subset=[Config.YEAR_COL]).copy()
    df[Config.YEAR_COL] = df[Config.YEAR_COL].astype(int)

    print(f"✅ Docs: {len(df)} | Years: {df[Config.YEAR_COL].min()}–{df[Config.YEAR_COL].max()}")

    # -------------------------
    # Embeddings + backend (CRITICAL FIX)
    # -------------------------
    from sentence_transformers import SentenceTransformer
    from bertopic.backend import SentenceTransformerBackend

    cache_dir = Path("data/cache_embeddings")
    safe_mkdir(cache_dir)
    cache_path = cache_dir / f"emb_{Config.EMBEDDING_MODEL_NAME.replace('/', '_')}_v7_1_{len(df)}.npy"

    print(f"🚀 Embeddings: {Config.EMBEDDING_MODEL_NAME}")
    st_model = SentenceTransformer(Config.EMBEDDING_MODEL_NAME)
    backend = SentenceTransformerBackend(st_model)  # ✅ provides embed_documents for KeyBERTInspired

    if cache_path.exists():
        embeddings = np.load(cache_path)
        print(f"♻️ Loaded cache: {cache_path}")
    else:
        embeddings = st_model.encode(
            df["text"].tolist(),
            batch_size=Config.EMB_BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=Config.EMB_NORMALIZE,
        )
        embeddings = np.asarray(embeddings, dtype=np.float32)
        np.save(cache_path, embeddings)
        print(f"💾 Saved cache: {cache_path}")

    print(f"✅ Embeddings shape: {embeddings.shape}")

    # -------------------------
    # BERTopic build
    # -------------------------
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer
    import umap
    import hdbscan

    umap_model = umap.UMAP(
        n_neighbors=Config.UMAP_N_NEIGHBORS,
        n_components=Config.UMAP_N_COMPONENTS,
        min_dist=Config.UMAP_MIN_DIST,
        metric=Config.UMAP_METRIC,
        random_state=Config.RANDOM_SEED,
    )

    hdb_model = hdbscan.HDBSCAN(
        min_cluster_size=Config.HDB_MIN_CLUSTER_SIZE,
        min_samples=Config.HDB_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=Config.HDB_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )

    vectorizer = CountVectorizer(
        ngram_range=Config.NGRAM_RANGE,
        min_df=Config.MIN_DF,
        max_df=Config.MAX_DF,
        max_features=Config.MAX_FEATURES,
        stop_words="english",
    )

    ctfidf = ClassTfidfTransformer(
        bm25_weighting=Config.BM25_WEIGHTING,
        reduce_frequent_words=Config.REDUCE_FREQUENT_WORDS,
    )

    rep_models = [KeyBERTInspired()]
    if Config.USE_MMR:
        rep_models.append(MaximalMarginalRelevance(diversity=Config.MMR_DIVERSITY))

    topic_model = BERTopic(
        embedding_model=backend,  # ✅ keep backend even if embeddings are passed
        umap_model=umap_model,
        hdbscan_model=hdb_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf,
        representation_model=rep_models,
        top_n_words=Config.TOP_N_WORDS,
        calculate_probabilities=True,
        verbose=True,
    )

    docs = df["text"].tolist()
    print("🧠 Fit BERTopic...")
    topics, probs = topic_model.fit_transform(docs, embeddings=embeddings)

    topics_arr = np.asarray(topics)
    out_init = 100.0 * float(np.sum(topics_arr == -1)) / len(topics_arr)
    n_topics_init = len(set([t for t in topics_arr if t != -1]))
    print(f"📈 Initial: topics={n_topics_init}, outliers={out_init:.2f}%")

    # -------------------------
    # Outlier control (optional)
    # -------------------------
    if Config.ENABLE_OUTLIER_CONTROL and probs is not None:
        low, high = Config.TARGET_OUTLIER_PCT_RANGE
        thr = float(Config.OUTLIER_PROB_THRESHOLD)

        def out_pct(tp):
            tp = np.asarray(tp)
            return 100.0 * float(np.sum(tp == -1)) / len(tp)

        try:
            topics2 = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=thr)
            p2 = out_pct(topics2)
            print(f"🧹 reduce_outliers[probabilities] thr={thr:.2f} -> outliers={p2:.2f}%")

            if p2 > high:
                thr2 = min(0.20, thr + 0.05)
                topics3 = topic_model.reduce_outliers(docs, topics, probabilities=probs, strategy="probabilities", threshold=thr2)
                p3 = out_pct(topics3)
                print(f"🧹 adjust thr={thr2:.2f} -> outliers={p3:.2f}%")
                topics2, p2 = topics3, p3

            topics = topics2
            print(f"✅ Final outliers: {p2:.2f}% (target {low}–{high}%)")
        except Exception as e:
            print(f"⚠️ Outlier control skipped due to error: {e}")

    # -------------------------
    # update_topics (wrapped: NEVER blocks exports)
    # -------------------------
    print("🧩 update_topics (representation refresh)")
    try:
        if getattr(topic_model, "embedding_model", None) is None:
            topic_model.embedding_model = backend  # extra safety
        topic_model.update_topics(
            docs,
            topics=topics,
            vectorizer_model=vectorizer,
            ctfidf_model=ctfidf,
            representation_model=rep_models,
        )
    except Exception as e:
        print(f"⚠️ update_topics failed, continuing: {e}")

    # -------------------------
    # Exports
    # -------------------------
    out = df.copy()
    out["Topic"] = topics

    out.to_csv(results_dir / "01_Full_Document_Assignments.csv", index=False)
    out[out["Topic"] == -1].to_csv(results_dir / "01b_Outlier_Documents.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(results_dir / "03_Topic_Metadata_Keywords.csv", index=False)

    # Summary
    summary = (
        out.groupby("Topic")
           .agg(Docs=("Topic", "size"), Year_min=(Config.YEAR_COL, "min"), Year_max=(Config.YEAR_COL, "max"))
           .reset_index()
    )
    summary["Share"] = summary["Docs"] / len(out)
    summary.to_csv(results_dir / "04_Summary_Statistics.csv", index=False)

    # Metrics (light)
    topic_words = {}
    for tid in topic_info["Topic"].tolist():
        tid = int(tid)
        if tid == -1:
            continue
        topic_words[tid] = [w for w, _ in (topic_model.get_topic(tid) or [])]

    diversity = compute_topic_diversity(topic_words, Config.TOP_N_WORDS)
    out_final = 100.0 * float(np.sum(np.asarray(topics) == -1)) / len(topics)
    topic_count_final = len(set([t for t in topics if t != -1]))

    metrics = pd.DataFrame([{
        "Docs": len(out),
        "Topic_Count": int(topic_count_final),
        "Outlier_%": float(out_final),
        "Topic_Diversity": float(diversity),
        "Embedding_Model": Config.EMBEDDING_MODEL_NAME,
        "Outlier_Control_Enabled": Config.ENABLE_OUTLIER_CONTROL,
        "Outlier_Prob_Threshold": Config.OUTLIER_PROB_THRESHOLD if Config.ENABLE_OUTLIER_CONTROL else None,
    }])
    metrics.to_csv(results_dir / "05_Quality_Metrics.csv", index=False)

    # Topics over time (RAW for R)
    long_full = topics_over_time_raw(out, year_col=Config.YEAR_COL, topic_col="Topic")
    long_full.to_csv(results_dir / "06a_Topics_Over_Time_FullGrid.csv", index=False)
    wide_count = wide_from_long(long_full, year_col=Config.YEAR_COL, topic_col="Topic", value_col="Count")
    wide_count.to_csv(results_dir / "06b_Topics_Over_Time_WideCount.csv", index=False)
    wide_share = wide_from_long(long_full, year_col=Config.YEAR_COL, topic_col="Topic", value_col="Share")
    wide_share.to_csv(results_dir / "06c_Topics_Over_Time_WideShare.csv", index=False)

    report = {
        "run": run,
        "docs": int(len(out)),
        "years": [int(out[Config.YEAR_COL].min()), int(out[Config.YEAR_COL].max())],
        "topic_count": int(topic_count_final),
        "outliers_pct": float(out_final),
        "diversity": float(diversity),
        "embeddings_cache": str(cache_path),
    }
    (results_dir / "00_Analysis_Report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    # Zip ALWAYS at the end
    zip_path = results_dir.with_suffix(".zip")  # results/run_xxx.zip
    zip_folder(results_dir, zip_path)
    print(f"📦 Zipped run folder: {zip_path}")

    print("=" * 80)
    print("DONE")
    print("=" * 80)


if __name__ == "__main__":
    main()
