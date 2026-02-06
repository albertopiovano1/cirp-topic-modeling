#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CIRP Topic Modeling (BERTopic) — v7.0 BALANCED (paper-ready)

Goal
----
- Keep *HDBSCAN topics/outliers as the base truth* ("natural")
- But allow *controlled* outlier reduction into a target band (e.g., 8–15%)
  without forcing 0% outliers (avoids topic contamination).
- Improve interpretability/metrics via:
  - scientific embeddings (default: bge-large) with caching
  - BM25 + reduce_frequent_words (c-TF-IDF)
  - domain + umbrella stopwords + max_df control
  - KeyBERTInspired + MMR (single update_topics at the end)
- Export *raw Topics-over-Time* (long + wide) for R streamgraph/joyplot.

This script is inspired by your older "Optimized v2" pipeline (Dec 2025)
which achieved: ~44 topics, ~14% outliers, high word-diversity. It keeps the
same principles (stopwords, max_df, BM25, controlled outlier reduction) while
adding run-stamped outputs, robust metrics, and R-friendly time exports.

Notes
-----
- If you truly want *zero interventions*, set DO_OUTLIER_REDUCTION=False.
- If you want "~10% outliers" consistently, keep DO_OUTLIER_REDUCTION=True and
  set TARGET_OUTLIER_PCT_RANGE.

"""

from __future__ import annotations

import os
import re
import json
import time
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# sklearn
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

# embeddings
import torch
from sentence_transformers import SentenceTransformer

# topic modeling
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance

# coherence
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel


# -----------------------------
# 0) CONFIG
# -----------------------------
@dataclass
class Config:
    SEED: int = 76

    # Data
    DATA_PATH: str = "data/CIRP_researchonly.csv"  # update if needed

    # Output
    OUTPUT_ROOT: str = "results"
    FIG_ROOT: str = "figures"

    # Filters
    KEEP_ONLY_ARTICLES: bool = True
    EXCLUDE_ANNALS_ISSUE2: bool = True  # remove Annals Issue 2 (keynotes/special)
    INCLUDE_JMST: bool = True

    # Embeddings
    EMBEDDING_MODEL_NAME: str = "BAAI/bge-large-en-v1.5"
    EMB_BATCH_SIZE: int = 64
    EMB_NORMALIZE: bool = True
    EMB_CACHE_DIR: str = "data/cache_embeddings"

    # UMAP
    UMAP_N_NEIGHBORS: int = 25
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.0
    UMAP_METRIC: str = "cosine"

    # HDBSCAN (main levers for outliers)
    HDB_MIN_CLUSTER_SIZE: int = 20
    HDB_MIN_SAMPLES: int = 2  # lower => fewer outliers
    HDB_CLUSTER_SELECTION: str = "eom"  # eom: robust; leaf: more granular

    # Vectorizer
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    MIN_DF: int = 5
    MAX_DF: float = 0.35  # aggressive -> reduces umbrella words
    MAX_FEATURES: int = 20000
    TOKEN_PATTERN: str = r"(?u)\b(?=\w*[a-zA-Z])[\w\-\+\.]{2,}\b"  # keeps cnc, 3d, ti-6al-4v

    # Outlier control (recommended for paper-ready assignments)
    DO_OUTLIER_REDUCTION: bool = True
    TARGET_OUTLIER_PCT_RANGE: Tuple[float, float] = (8.0, 15.0)  # aim ~10–15%
    OUTLIER_STRATEGIES: Tuple[str, ...] = ("probabilities", "c-tf-idf")
    OUTLIER_THRESHOLDS: Tuple[float, ...] = (0.00, 0.02, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30)

    # Representation
    TOP_N_WORDS: int = 10
    MMR_DIVERSITY: float = 0.35

    # Stability (fast + reproducible)
    STABILITY_BOOTSTRAPS: int = 5
    STABILITY_SAMPLE_FRAC: float = 0.8

    # Topics over time exports
    INCLUDE_OUTLIER_IN_TOT: bool = False  # in time exports; usually False


CFG = Config()


# -----------------------------
# 1) DETERMINISM
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# -----------------------------
# 2) HELPERS
# -----------------------------
_BOILERPLATE_CUT_PATTERNS = [
    r"©",
    r"All rights reserved",
    r"Published by",
    r"Elsevier",
]


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"<[^>]+>", " ", text)  # strip markup
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text).strip()

    cut_positions = []
    for pat in _BOILERPLATE_CUT_PATTERNS:
        m = re.search(pat, text)
        if m:
            cut_positions.append(m.start())
    if cut_positions:
        text = text[: min(cut_positions)].strip()

    return text


def get_col(df: pd.DataFrame, candidates: List[str]) -> str:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    raise KeyError(f"Missing required column. Tried: {candidates}. Available: {list(df.columns)[:30]}...")


def outlier_percent(topics: List[int]) -> float:
    t = np.asarray(topics)
    return float((t == -1).mean() * 100.0)


def run_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


# -----------------------------
# 3) STOPWORDS
# -----------------------------
DOMAIN_STOPWORDS = {
    # academic umbrella
    "paper", "study", "research", "work", "result", "results", "method", "methods", "approach", "approaches",
    "analysis", "performance", "experimental", "experiment", "experiments", "developed", "based", "investigated",
    "demonstrated", "different", "proposed", "propose", "proposes", "using", "use", "used", "show", "shows", "shown",
    "new", "novel", "application", "applications",

    # CIRP-wide umbrella
    "manufacturing", "manufacture", "manufactured", "production", "produce", "process", "processes", "processing",
    "system", "systems", "model", "models",

    # publisher noise
    "cirp", "elsevier", "ltd", "author", "authors", "rights", "reserved", "published",

    # citation noise
    "et", "al",
}
CUSTOM_STOPWORDS = sorted(set(ENGLISH_STOP_WORDS).union(DOMAIN_STOPWORDS))


# -----------------------------
# 4) LOAD + FILTER
# -----------------------------

def load_and_filter() -> Tuple[pd.DataFrame, List[str], List[int], List[str]]:
    print(f"📂 Loading dataset: {CFG.DATA_PATH}")
    df = pd.read_csv(CFG.DATA_PATH)

    col_title = get_col(df, ["Title"]) 
    col_abs = get_col(df, ["Abstract"]) 
    col_year = get_col(df, ["Year", "Publication Year", "PY"]) 

    # optional columns
    col_dtype = None
    for c in ["Document Type", "DocumentType", "DT"]:
        if c.lower() in {x.lower() for x in df.columns}:
            col_dtype = get_col(df, [c])
            break

    col_source = None
    for c in ["Source title", "Source Title", "Journal", "Source"]:
        if c.lower() in {x.lower() for x in df.columns}:
            col_source = get_col(df, [c])
            break

    col_issue = None
    for c in ["Issue", "issue"]:
        if c.lower() in {x.lower() for x in df.columns}:
            col_issue = get_col(df, [c])
            break

    # keep only articles
    if CFG.KEEP_ONLY_ARTICLES and col_dtype is not None:
        before = len(df)
        df = df[df[col_dtype].astype(str).str.lower().eq("article")].copy()
        print(f"🧹 Filter articles: kept {len(df)}/{before}")

    # include/exclude JMST
    if not CFG.INCLUDE_JMST and col_source is not None:
        df = df[df[col_source].astype(str).str.strip().eq("CIRP Annals")].copy()

    # exclude Annals Issue 2
    if CFG.EXCLUDE_ANNALS_ISSUE2 and col_source is not None and col_issue is not None:
        is_annals = df[col_source].astype(str).str.strip().eq("CIRP Annals")
        before = len(df)
        df = df[~is_annals | df[col_issue].isna() | (pd.to_numeric(df[col_issue], errors="coerce") == 1)].copy()
        print(f"🧹 Exclude Annals Issue 2: kept {len(df)}/{before}")

    # build docs
    df[col_title] = df[col_title].fillna("")
    df[col_abs] = df[col_abs].fillna("")
    df["raw_text"] = (df[col_title] + ". " + df[col_abs]).astype(str)
    df["text_embed"] = df["raw_text"].apply(clean_text)

    df = df[df["text_embed"].str.len() > 30].copy()

    years = pd.to_numeric(df[col_year], errors="coerce").fillna(-1).astype(int)
    df = df[years >= 0].copy()
    years = years.loc[df.index].tolist()

    docs_embed = df["text_embed"].tolist()
    docs_ctfidf = [re.sub(r"\s+", " ", d.lower()).strip() for d in docs_embed]

    print(f"✅ Docs after filters: {len(df)} | Years: {min(years)}–{max(years)}")
    return df, docs_embed, years, docs_ctfidf


# -----------------------------
# 5) EMBEDDINGS (cached)
# -----------------------------

def embed_docs(docs_embed: List[str], cache_tag: str) -> Tuple[np.ndarray, str]:
    ensure_dir(Path(CFG.EMB_CACHE_DIR))

    cache_name = f"emb_{CFG.EMBEDDING_MODEL_NAME.replace('/', '_')}_{cache_tag}_{len(docs_embed)}.npy"
    cache_path = Path(CFG.EMB_CACHE_DIR) / cache_name

    if cache_path.exists():
        emb = np.load(cache_path)
        return emb, str(cache_path)

    print(f"🚀 Embeddings: {CFG.EMBEDDING_MODEL_NAME}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(CFG.EMBEDDING_MODEL_NAME, device=device)

    try:
        model.max_seq_length = 512
    except Exception:
        pass

    try:
        emb = model.encode(
            docs_embed,
            batch_size=CFG.EMB_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=CFG.EMB_NORMALIZE,
        )
    except TypeError:
        emb = model.encode(
            docs_embed,
            batch_size=CFG.EMB_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        if CFG.EMB_NORMALIZE:
            norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
            emb = emb / norms

    np.save(cache_path, emb)
    return emb, str(cache_path)


# -----------------------------
# 6) OUTLIER REDUCTION (to target band)
# -----------------------------

def pick_threshold_for_target(
    model: BERTopic,
    docs: List[str],
    topics_base: List[int],
    probs_base: Optional[np.ndarray],
    strategy: str,
    target_low: float,
    target_high: float,
    thresholds: Tuple[float, ...],
) -> Tuple[float, float, List[int]]:
    results: List[Tuple[float, float, List[int]]] = []

    for thr in thresholds:
        if strategy == "probabilities":
            if probs_base is None:
                continue
            new_t = model.reduce_outliers(
                docs,
                topics_base,
                probabilities=probs_base,
                strategy=strategy,
                threshold=thr,
            )
        elif strategy == "c-tf-idf":
            new_t = model.reduce_outliers(
                docs,
                topics_base,
                strategy=strategy,
                threshold=thr,
            )
        else:
            raise ValueError(strategy)

        pct = outlier_percent(new_t)
        results.append((thr, pct, list(new_t)))

    in_range = [r for r in results if target_low <= r[1] <= target_high]
    if in_range:
        best = min(in_range, key=lambda x: abs(x[1] - target_high))
        return best

    # else choose closest, penalize going below target_low heavily
    def score(item: Tuple[float, float, List[int]]) -> float:
        _, pct, _ = item
        if pct < target_low:
            return (target_low - pct) + 1000.0
        if pct > target_high:
            return (pct - target_high)
        return 0.0

    best = min(results, key=score)
    return best


# -----------------------------
# 7) STABILITY (bootstrap ARI)
# -----------------------------

def bootstrap_ari(
    umap_model: UMAP,
    hdbscan_kwargs: Dict[str, Any],
    embeddings: np.ndarray,
    base_labels: np.ndarray,
    n_boot: int,
    frac: float,
    seed: int,
) -> float:
    from sklearn.metrics import adjusted_rand_score

    rng = np.random.default_rng(seed)
    n = embeddings.shape[0]
    m = int(max(10, round(frac * n)))

    scores: List[float] = []
    for b in range(n_boot):
        idx = rng.choice(n, size=m, replace=False)
        X = embeddings[idx]
        y_base = base_labels[idx]

        # fit new UMAP + HDBSCAN on the bootstrap sample
        um = UMAP(
            n_neighbors=umap_model.n_neighbors,
            n_components=umap_model.n_components,
            min_dist=umap_model.min_dist,
            metric=umap_model.metric,
            random_state=seed + 1000 + b,
        )
        Xr = um.fit_transform(X)

        hdb = HDBSCAN(**hdbscan_kwargs)
        y_boot = hdb.fit_predict(Xr)

        scores.append(float(adjusted_rand_score(y_base, y_boot)))

    return float(np.mean(scores)) if scores else 0.0


# -----------------------------
# 8) TOPICS OVER TIME (raw exports for R)
# -----------------------------

def export_topics_over_time(
    outdir: Path,
    years: List[int],
    topics: List[int],
    topic_labels: Dict[int, str],
    include_outlier: bool,
) -> None:
    df_t = pd.DataFrame({"Year": years, "Topic": topics})
    if not include_outlier:
        df_t = df_t[df_t["Topic"] != -1].copy()

    # Long counts
    long_counts = (
        df_t.groupby(["Year", "Topic"]).size().reset_index(name="Count")
    )

    # Complete grid (explicit zeros)
    all_years = sorted(df_t["Year"].unique().tolist())
    all_topics = sorted(df_t["Topic"].unique().tolist())

    grid = pd.MultiIndex.from_product([all_years, all_topics], names=["Year", "Topic"]).to_frame(index=False)
    full = grid.merge(long_counts, on=["Year", "Topic"], how="left")
    full["Count"] = full["Count"].fillna(0).astype(int)

    # Shares
    year_totals = full.groupby("Year")["Count"].transform("sum").replace(0, np.nan)
    full["Share"] = (full["Count"] / year_totals).fillna(0.0)

    # add labels
    full["Topic_Name"] = full["Topic"].map(topic_labels).fillna(full["Topic"].astype(str))

    # stable ranks by total volume
    totals = full.groupby("Topic")["Count"].sum().sort_values(ascending=False)
    rank_map = {t: i + 1 for i, t in enumerate(totals.index.tolist())}
    full["Topic_Rank"] = full["Topic"].map(rank_map).fillna(9999).astype(int)

    # exports
    full.to_csv(outdir / "06a_Topics_Over_Time_FullGrid.csv", index=False)

    wide_count = full.pivot(index="Year", columns="Topic", values="Count").fillna(0).astype(int)
    wide_count.columns = [f"T{c}" for c in wide_count.columns]
    wide_count.to_csv(outdir / "06b_Topics_Over_Time_WideCount.csv")

    wide_share = full.pivot(index="Year", columns="Topic", values="Share").fillna(0.0)
    wide_share.columns = [f"T{c}" for c in wide_share.columns]
    wide_share.to_csv(outdir / "06c_Topics_Over_Time_WideShare.csv")


# -----------------------------
# 9) MAIN
# -----------------------------

def main() -> None:
    set_seed(CFG.SEED)

    stamp = run_stamp()
    outdir = Path(CFG.OUTPUT_ROOT) / f"run_{stamp}"
    figdir = Path(CFG.FIG_ROOT) / f"run_{stamp}"
    ensure_dir(outdir)
    ensure_dir(figdir)

    print("=" * 80)
    print("CIRP BIBLIOMETRIC ANALYSIS — BERTopic v7.0 BALANCED")
    print("=" * 80)
    print(f"Seed: {CFG.SEED}")
    print(f"Output: {outdir}")

    # Load
    df, docs_embed, years, docs_ctfidf = load_and_filter()

    # Embeddings
    emb, emb_cache = embed_docs(docs_embed, cache_tag=f"v7_0")
    print(f"✅ Embeddings shape: {emb.shape}")
    print(f"💾 Embeddings cache: {emb_cache}")

    # Models
    umap_model = UMAP(
        n_neighbors=CFG.UMAP_N_NEIGHBORS,
        n_components=CFG.UMAP_N_COMPONENTS,
        min_dist=CFG.UMAP_MIN_DIST,
        metric=CFG.UMAP_METRIC,
        random_state=CFG.SEED,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=CFG.HDB_MIN_CLUSTER_SIZE,
        min_samples=CFG.HDB_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=CFG.HDB_CLUSTER_SELECTION,
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=CUSTOM_STOPWORDS,
        ngram_range=CFG.NGRAM_RANGE,
        min_df=CFG.MIN_DF,
        max_df=CFG.MAX_DF,
        max_features=CFG.MAX_FEATURES,
        token_pattern=CFG.TOKEN_PATTERN,
        strip_accents="unicode",
    )

    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=True,
        reduce_frequent_words=True,
    )

    # Fit without representation (compute once later)
    topic_model = BERTopic(
        embedding_model=None,  # we pass precomputed embeddings
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=None,
        calculate_probabilities=True,
        verbose=True,
    )

    print("🧠 Fit BERTopic...")
    topics, probs = topic_model.fit_transform(docs_ctfidf, embeddings=emb)

    base_out = outlier_percent(topics)
    n_topics = len([t for t in set(topics) if t != -1])
    print(f"📈 Initial: topics={n_topics}, outliers={base_out:.2f}%")

    topics_final = list(topics)
    outlier_reduction_log: List[Dict[str, Any]] = []

    if CFG.DO_OUTLIER_REDUCTION and (-1 in set(topics_final)):
        target_low, target_high = CFG.TARGET_OUTLIER_PCT_RANGE

        for strat in CFG.OUTLIER_STRATEGIES:
            cur_out = outlier_percent(topics_final)
            if cur_out <= target_high:
                break

            thr, pct, new_topics = pick_threshold_for_target(
                model=topic_model,
                docs=docs_ctfidf,
                topics_base=topics_final,
                probs_base=probs,
                strategy=strat,
                target_low=target_low,
                target_high=target_high,
                thresholds=CFG.OUTLIER_THRESHOLDS,
            )

            outlier_reduction_log.append({
                "strategy": strat,
                "threshold": float(thr),
                "outliers_pct": float(pct),
            })
            topics_final = new_topics
            print(f"🧹 reduce_outliers[{strat}] thr={thr:.2f} -> outliers={pct:.2f}%")

    final_out = outlier_percent(topics_final)
    print(f"✅ Final outliers: {final_out:.2f}% (target {CFG.TARGET_OUTLIER_PCT_RANGE[0]}–{CFG.TARGET_OUTLIER_PCT_RANGE[1]}%)")

    # Update topic representations ONCE
    representation_model = [
        KeyBERTInspired(random_state=CFG.SEED),
        MaximalMarginalRelevance(diversity=CFG.MMR_DIVERSITY),
    ]

    print("🧩 update_topics (representation + stopwords-safe)")
    topic_model.update_topics(
        docs=docs_ctfidf,
        topics=topics_final,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=CFG.TOP_N_WORDS,
    )

    # Labels
    topic_labels: Dict[int, str] = {}
    try:
        labels = topic_model.generate_topic_labels(nr_words=4, separator=" | ", topic_prefix=False)
        topic_model.set_topic_labels(labels)
        # map
        info = topic_model.get_topic_info()
        for _, row in info.iterrows():
            tid = int(row["Topic"]) if "Topic" in row else int(row[0])
            if tid == -1:
                continue
            topic_labels[tid] = str(row.get("Name", tid))
    except Exception:
        # fallback: build from top words
        for tid in sorted(set(topics_final)):
            if tid == -1:
                continue
            ws = [w for w, _ in (topic_model.get_topic(tid) or [])][:4]
            topic_labels[tid] = " | ".join(ws) if ws else str(tid)

    # Metrics
    print("📈 Computing metrics...")
    analyzer = vectorizer_model.build_analyzer()
    tokenized_docs = [analyzer(d) for d in docs_ctfidf]

    topic_ids = [t for t in sorted(set(topics_final)) if t != -1]
    topic_words = [[str(w) for w, _ in (topic_model.get_topic(t) or [])[:CFG.TOP_N_WORDS]] for t in topic_ids]

    # coherence
    dictionary = Dictionary(tokenized_docs)
    # keep only words in dictionary for stability
    vocab = set(dictionary.token2id.keys())
    topic_words_clean = [[w for w in tw if w in vocab] for tw in topic_words]
    topic_words_clean = [tw for tw in topic_words_clean if len(tw) >= 2]

    if topic_words_clean:
        cv = float(CoherenceModel(
            topics=topic_words_clean,
            texts=tokenized_docs,
            dictionary=dictionary,
            coherence="c_v",
        ).get_coherence())
    else:
        cv = 0.0

    # diversity
    unique_words = set([w for tw in topic_words for w in tw])
    diversity = float(len(unique_words) / (len(topic_ids) * CFG.TOP_N_WORDS)) if topic_ids else 0.0

    # stability
    base_labels = np.asarray(topics_final)
    hdb_kwargs = dict(
        min_cluster_size=CFG.HDB_MIN_CLUSTER_SIZE,
        min_samples=CFG.HDB_MIN_SAMPLES,
        metric="euclidean",
        cluster_selection_method=CFG.HDB_CLUSTER_SELECTION,
        prediction_data=False,
    )
    ari = bootstrap_ari(
        umap_model=umap_model,
        hdbscan_kwargs=hdb_kwargs,
        embeddings=emb,
        base_labels=base_labels,
        n_boot=CFG.STABILITY_BOOTSTRAPS,
        frac=CFG.STABILITY_SAMPLE_FRAC,
        seed=CFG.SEED,
    )

    # Topic count (excluding -1)
    topic_count = len([t for t in set(topics_final) if t != -1])

    metrics = {
        "Topic Coherence (C_v)": cv,
        "Topic Diversity": diversity,
        "Topic Count": topic_count,
        "Outlier %": final_out,
        "Docs": len(docs_ctfidf),
        "Stability (ARI)": ari,
        "Embedding Model": CFG.EMBEDDING_MODEL_NAME,
        "UMAP": json.dumps({"n_neighbors": CFG.UMAP_N_NEIGHBORS, "n_components": CFG.UMAP_N_COMPONENTS, "min_dist": CFG.UMAP_MIN_DIST, "metric": CFG.UMAP_METRIC}),
        "HDBSCAN": json.dumps({"min_cluster_size": CFG.HDB_MIN_CLUSTER_SIZE, "min_samples": CFG.HDB_MIN_SAMPLES, "cluster_selection": CFG.HDB_CLUSTER_SELECTION}),
        "Vectorizer": json.dumps({"ngram_range": CFG.NGRAM_RANGE, "min_df": CFG.MIN_DF, "max_df": CFG.MAX_DF, "max_features": CFG.MAX_FEATURES}),
    }

    pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}).to_csv(outdir / "05_Quality_Metrics.csv", index=False)
    if outlier_reduction_log:
        pd.DataFrame(outlier_reduction_log).to_csv(outdir / "05b_Outlier_Reduction_Log.csv", index=False)

    print("\n📌 METRICS")
    print(pd.DataFrame({"Metric": list(metrics.keys()), "Value": list(metrics.values())}))

    # Exports
    df_out = df.copy()
    df_out["Topic"] = topics_final
    df_out.to_csv(outdir / "01_Full_Document_Assignments.csv", index=False)

    # outliers file
    out_df = df_out[df_out["Topic"] == -1].copy()
    out_df.to_csv(outdir / "01b_Outlier_Documents.csv", index=False)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(outdir / "03_Topic_Info.csv", index=False)

    # keywords/metadata
    rows = []
    for tid in sorted(set(topics_final)):
        if tid == -1:
            continue
        kw = topic_model.get_topic(tid) or []
        rows.append({
            "Topic": tid,
            "Name": topic_labels.get(tid, str(tid)),
            "Top_Words": ", ".join([w for w, _ in kw[:CFG.TOP_N_WORDS]]),
        })
    pd.DataFrame(rows).to_csv(outdir / "03_Topic_Metadata_Keywords.csv", index=False)

    # top papers per topic (by citations if available else by year)
    # We keep it simple: select 5 docs with longest abstracts as proxy if no citations.
    col_cit = None
    for c in ["Cited by", "Citations", "Times Cited", "Citedby"]:
        if c.lower() in {x.lower() for x in df_out.columns}:
            col_cit = get_col(df_out, [c])
            break

    top_rows = []
    for tid in sorted(set(topics_final)):
        if tid == -1:
            continue
        sub = df_out[df_out["Topic"] == tid].copy()
        if sub.empty:
            continue
        if col_cit is not None:
            sub[col_cit] = pd.to_numeric(sub[col_cit], errors="coerce").fillna(0)
            sub = sub.sort_values(col_cit, ascending=False)
        else:
            sub["__abs_len"] = sub["Abstract"].astype(str).str.len()
            sub = sub.sort_values("__abs_len", ascending=False)
        sub = sub.head(5)
        for _, r in sub.iterrows():
            top_rows.append({
                "Topic": tid,
                "Topic_Name": topic_labels.get(tid, str(tid)),
                "Title": r.get("Title", ""),
                "Year": r.get("Year", ""),
                "DOI": r.get("DOI", ""),
                "Citations": r.get(col_cit, "") if col_cit else "",
            })
    pd.DataFrame(top_rows).to_csv(outdir / "02_Top_Papers_Per_Topic.csv", index=False)

    # Topics over time exports for R
    export_topics_over_time(
        outdir=outdir,
        years=years,
        topics=topics_final,
        topic_labels=topic_labels,
        include_outlier=CFG.INCLUDE_OUTLIER_IN_TOT,
    )

    # Save model (optional). Some environments struggle with saving embedding_model.
    try:
        topic_model.save(
            outdir / "bertopic_model",
            serialization="safetensors",
            save_ctfidf=True,
            save_embedding_model=False,
        )
    except Exception as e:
        print(f"⚠️ Could not save model: {e}")

    # Minimal visuals (optional)
    try:
        topic_model.visualize_topics().write_html(str(figdir / "Viz_01_Intertopic_Distance_Map.html"))
        topic_model.visualize_barchart(top_n_topics=min(30, topic_count)).write_html(str(figdir / "Viz_02_Topic_Word_Scores.html"))
        topic_model.visualize_hierarchy().write_html(str(figdir / "Viz_03_Hierarchical_Clustering.html"))
    except Exception as e:
        print(f"⚠️ Skipped visuals (plotly/kaleido missing?): {e}")

    # Report
    report = {
        "run": stamp,
        "seed": CFG.SEED,
        "docs": len(docs_ctfidf),
        "topic_count": topic_count,
        "outliers_pct": final_out,
        "metrics": metrics,
        "outlier_reduction": outlier_reduction_log,
        "embedding_cache": emb_cache,
    }
    (outdir / "00_Analysis_Report.json").write_text(json.dumps(report, indent=2))

    print("\n✅ Done.")
    print(f"Run folder: {outdir}")


if __name__ == "__main__":
    main()
