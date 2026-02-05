#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v6.5 (natural, fixed coherence + stopwords)
================================================================================
Goal: Paper-ready topic modelling for CIRP Annals + CIRP JMST research papers (2000–2025)

Key properties (as requested):
- NO forced nr_topics (no reduce_topics / target topics)
- Outliers are NATURAL from HDBSCAN (topic = -1), no reassignment by default
- Research-only filtering (exclude reviews/editorials/etc.) before modelling
- Robust exports + robust Topics-over-Time (table for ALL topics + heatmap)
- Stability FIXED using clustering stability (ARI on bootstrap subsamples) — no BERTopic backend calls
- Run-stamped output folders to avoid mixing results across runs
- Optional embedding cache to iterate hyperparameters quickly

================================================================================
"""

import os
import re
import json
import time
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import adjusted_rand_score
from sklearn.utils import resample
from tqdm import tqdm
from umap import UMAP

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")


# =============================================================================
# 1) CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class Config:
    # Paths
    DATA_DIR: Path = Path("data")
    OUTPUT_BASE: Path = Path("results")
    FIGURES_BASE: Path = Path("figures")

    # Run stamping (prevents mixing runs)
    RUN_ID: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Input file
    INPUT_FILE: str = "CIRP_researchonly.csv"

    # Reproducibility
    SEED: int = 76

    # --- Corpus filtering (research-only) ---
    INCLUDE_DOC_TYPES: Tuple[str, ...] = ("article",)
    EXCLUDE_DOC_TYPES: Tuple[str, ...] = ("review", "editorial", "erratum", "correction", "retracted", "note")

    # Text preprocessing
    MIN_WORDS_PER_DOC: int = 10
    MIN_ABSTRACT_LENGTH: int = 50

    # Stopwords / synonyms
    STOPWORDS_FILE: str = "custom_stopwords.txt"   # under DATA_DIR; optional
    SYNONYMS_FILE: str = "synonyms_map.json"       # under DATA_DIR; optional
    AUTO_STOPWORD_TOP_N: int = 30                  # add top-N most frequent tokens in corpus (after lemmatization)
    AUTO_STOPWORD_MIN_LEN: int = 4                 # avoid removing short technical tokens (e.g., "mql", "edm")

    # Vectorizer extra stopwords (post-lemmatization tokens) to improve topic diversity/interpretability
    # NOTE: affects only topic representation (c-TF-IDF), not clustering/outliers.
    VECTORIZER_EXTRA_STOPWORDS: Tuple[str, ...] = (
        "surface", "tool", "design", "material", "machine", "product", "control", "quality", "high", "time",
        "micro", "structure", "mechanical", "property", "dynamic", "develop", "stress", "component", "condition", "test",
        "increase", "speed", "temperature", "force", "measurement"
    )


    # --- Embeddings ---
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"
    EMBEDDING_BATCH_SIZE: int = 32
    USE_EMBEDDING_CACHE: bool = True
    CACHE_TAG: str = "v6_5"   # bump this if you change preprocessing in a way that invalidates cached embeddings

    # --- BERTopic parameters (NATURAL topics/outliers via UMAP+HDBSCAN) ---
    # UMAP: more local structure than v6.3 to avoid "too few topics"
    UMAP_N_NEIGHBORS: int = 20
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.0
    UMAP_METRIC: str = "cosine"

    # HDBSCAN: allow more clusters, but keep a floor to avoid micro-noise
    HDBSCAN_MIN_CLUSTER_SIZE: int = 20
    HDBSCAN_MIN_SAMPLES: int = 2
    HDBSCAN_CLUSTER_SELECTION_METHOD: str = "leaf"   # "leaf" -> more (natural) clusters than default "eom"

    # Vectorizer (interpretability)
    VECTORIZER_NGRAM_RANGE: Tuple[int, int] = (1, 2)
    VECTORIZER_MIN_DF: int = 10
    VECTORIZER_MAX_FEATURES: int = 12000

    TOP_N_WORDS: int = 12

    # Outlier handling — NATURAL by default
    ENABLE_OUTLIER_REASSIGN: bool = False
    OUTLIER_REASSIGN_STRATEGY: str = "embeddings"
    OUTLIER_REASSIGN_THRESHOLD: float = 0.30

    # Analysis parameters
    TIME_CUTOFF_YEAR: int = 2025
    TOP_N_TOPICS_LINEPLOT: int = 16        # lineplot shows top-N by volume (for readability)
    INCLUDE_OUTLIERS_IN_TIMEPLOTS: bool = True

    TOP_N_PAPERS_PER_TOPIC: int = 5

    # Stability (clustering stability, ARI)
    STABILITY_BOOTSTRAPS: int = 8
    STABILITY_SAMPLE_FRAC: float = 0.80
    STABILITY_IGNORE_OUTLIERS: bool = True   # ignore topic=-1 when computing ARI

    # Exports
    EXPORT_QUALITY_METRICS: bool = True
    EXPORT_TOP_TOKENS: bool = True
    TOP_TOKENS_K: int = 80


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_slug(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s)
    return s.strip("_")


def ensure_dirs(results_dir: Path, figures_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2) DATA LOADING + RESEARCH-ONLY FILTER
# =============================================================================

class DataLoader:
    @staticmethod
    def load(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        return df

    @staticmethod
    def filter_research_only(df: pd.DataFrame) -> pd.DataFrame:
        if "Document Type" not in df.columns:
            return df

        def norm(x: str) -> str:
            return str(x).strip().lower()

        before = len(df)
        types = df["Document Type"].map(norm)

        mask = pd.Series(True, index=df.index)

        if Config.INCLUDE_DOC_TYPES:
            include = set([t.lower() for t in Config.INCLUDE_DOC_TYPES])
            mask &= types.isin(include)

        if Config.EXCLUDE_DOC_TYPES:
            exclude = set([t.lower() for t in Config.EXCLUDE_DOC_TYPES])
            mask &= ~types.isin(exclude)

        df2 = df[mask].copy()
        removed = before - len(df2)
        print(f"🧹 Research-only filter: kept {len(df2):,}/{before:,} docs; removed {removed:,} (reviews/editorials/other).")
        return df2


# =============================================================================
# 3) TEXT PREPROCESSING
# =============================================================================

class TextPreprocessor:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        self.syn_map: Dict[str, str] = {}
        self.auto_stopwords_added: List[str] = []
        self._load_synonyms()
        self._load_stopwords()

    def _load_synonyms(self) -> None:
        syn_path = Config.DATA_DIR / Config.SYNONYMS_FILE
        if syn_path.exists():
            try:
                self.syn_map = json.loads(syn_path.read_text(encoding="utf-8"))
                # normalize keys/values
                self.syn_map = {str(k).lower().strip(): str(v).lower().strip() for k, v in self.syn_map.items()}
                print(f"📝 Synonym normalization: file mode ({len(self.syn_map)} mappings)")
                return
            except Exception as e:
                print(f"⚠️  Could not read synonyms_map.json ({e}). Falling back to SAFE mode.")
        # SAFE fallback (minimal)
        self.syn_map = {
            "additive manufacturing": "additive_manufacturing",
            "machine tool": "machine_tool",
            "friction stir welding": "friction_stir_welding",
            "digital twin": "digital_twin",
            "finite element": "finite_element",
            "laser powder bed fusion": "lpbf",
        }
        print(f"📝 Synonym normalization: SAFE mode ({len(self.syn_map)} mappings)")

    def _load_stopwords(self) -> None:
        # Base academic filler + manufacturing-generic (expanded vs v6.3)
        stopwords_academic = [
            "aim", "objective", "paper", "study", "result", "results", "method", "methods",
            "approach", "analysis", "analyses", "investigate", "investigation", "propose",
            "proposed", "present", "presented", "provide", "provided", "show", "shown",
            "demonstrate", "demonstrated", "discuss", "discussed", "conclude", "concluded",
            "conclusion", "conclusions", "future", "work", "based", "using", "use", "used",
            "via", "within", "without", "among", "across", "however", "therefore",
            "et", "al", "etc"
        ]
        stopwords_domain = [
            "manufacturing", "process", "processes", "production", "industrial", "industry",
            "system", "systems", "approach", "model", "models", "method", "methods",
            "technology", "technologies", "application", "applications", "performance",
            "effect", "effects", "parameter", "parameters", "optimization", "optimisation",
            "experiment", "experiments", "experimental", "simulation", "simulations",
            "validation", "validated", "case", "cases", "based", "proposed", "paper",
            "journal", "article"
        ]

        custom = set([w.lower().strip() for w in (stopwords_academic + stopwords_domain)])

        # Optional file-based stopwords (recommended)
        sw_path = Config.DATA_DIR / Config.STOPWORDS_FILE
        if sw_path.exists():
            try:
                extra = [ln.strip().lower() for ln in sw_path.read_text(encoding="utf-8").splitlines()
                         if ln.strip() and not ln.strip().startswith("#")]
                custom |= set(extra)
                print(f"🛑 Custom stopwords: file loaded (+{len(extra)} terms)")
            except Exception as e:
                print(f"⚠️  Could not read custom_stopwords.txt ({e}). Using built-in list only.")

        self.nlp.Defaults.stop_words |= custom
        self.base_custom_stopwords = custom
        print(f"🛑 Custom stopwords loaded: {len(custom)} terms")

    @staticmethod
    def clean_text_for_embedding(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.lower().strip()

    @staticmethod
    def clean_text_for_vectorizer(text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""
        text = TextPreprocessor.clean_text_for_embedding(text)
        text = re.sub(r"[^\w\s-]", " ", text)
        text = re.sub(r"\b\d+\b", "", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def lemmatize_and_normalize(self, text: str) -> str:
        text = self.clean_text_for_vectorizer(text)
        if not text:
            return ""
        # apply phrase-level synonyms first
        for k, v in self.syn_map.items():
            text = text.replace(k, v)
        doc = self.nlp(text)
        toks = []
        for t in doc:
            if t.is_stop or t.is_punct or t.like_num:
                continue
            lemma = t.lemma_.lower().strip()
            if not lemma or lemma in self.nlp.Defaults.stop_words:
                continue
            toks.append(lemma)
        return " ".join(toks)

    def build_corpus(self, df: pd.DataFrame) -> Tuple[List[str], List[str], pd.DataFrame]:
        # Build raw combined text
        def combine(row) -> str:
            parts = []
            for col in ["Title", "Abstract", "Author Keywords", "Index Keywords"]:
                if col in df.columns and isinstance(row.get(col, ""), str):
                    parts.append(row.get(col, ""))
            return " ".join(parts)

        raw = [combine(r) for _, r in df.iterrows()]
        text_embed = [self.clean_text_for_embedding(t) for t in raw]
        text_vec = [self.lemmatize_and_normalize(t) for t in tqdm(raw, desc="Lemmatizing", total=len(raw))]

        df2 = df.copy()
        df2["text_embed"] = text_embed
        df2["text_vec"] = text_vec

        # Filter docs
        mask = df2["text_vec"].apply(lambda x: isinstance(x, str) and len(x.split()) >= Config.MIN_WORDS_PER_DOC)
        df2 = df2[mask].copy()

        # Abstract length gate (optional)
        if "Abstract" in df2.columns:
            abs_len = df2["Abstract"].astype(str).apply(lambda x: len(str(x).split()))
            df2 = df2[abs_len >= Config.MIN_ABSTRACT_LENGTH].copy()

        df2 = df2.reset_index(drop=True)
        return df2["text_embed"].tolist(), df2["text_vec"].tolist(), df2

    def auto_stopwords_from_corpus(self, texts_vec: List[str], results_dir: Path) -> None:
        """Add top-N most frequent tokens (post-lemmatization) to stopwords to reduce generic overlap."""
        if Config.AUTO_STOPWORD_TOP_N <= 0:
            return
        counts: Dict[str, int] = {}
        for doc in texts_vec:
            for tok in doc.split():
                if len(tok) < Config.AUTO_STOPWORD_MIN_LEN:
                    continue
                counts[tok] = counts.get(tok, 0) + 1
        # take most frequent, excluding existing stopwords
        items = sorted(counts.items(), key=lambda x: x[1], reverse=True)
        added = []
        for tok, c in items:
            if tok in self.nlp.Defaults.stop_words:
                continue
            added.append(tok)
            if len(added) >= Config.AUTO_STOPWORD_TOP_N:
                break
        self.auto_stopwords_added = added.copy()
        self.nlp.Defaults.stop_words |= set(added)
        if added:
            print(f"🧯 Auto-stopwords added: {len(added)} terms (top corpus frequency)")

        if Config.EXPORT_TOP_TOKENS:
            out = results_dir / "07_Top_Tokens.csv"
            topk = items[:Config.TOP_TOKENS_K]
            pd.DataFrame(topk, columns=["token", "count"]).to_csv(out, index=False)
            print(f"✅ Saved top tokens: {out}")


# =============================================================================
# 4) EMBEDDINGS (with optional caching)
# =============================================================================

class EmbeddingGenerator:
    def __init__(self, results_dir: Path):
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        self.results_dir = results_dir

    def _cache_path(self, n_docs: int) -> Path:
        cache_dir = Config.DATA_DIR / "cache_embeddings"
        cache_dir.mkdir(parents=True, exist_ok=True)
        tag = safe_slug(Config.CACHE_TAG)
        model = safe_slug(Config.EMBEDDING_MODEL.split("/")[-1])
        return cache_dir / f"emb_{model}_{tag}_{n_docs}.npy"

    def encode(self, texts_embed: List[str]) -> np.ndarray:
        n = len(texts_embed)
        cache_path = self._cache_path(n)
        if Config.USE_EMBEDDING_CACHE and cache_path.exists():
            try:
                emb = np.load(cache_path)
                if emb.shape[0] == n:
                    print(f"⚡ Loaded cached embeddings: {cache_path}")
                    return emb
            except Exception:
                pass

        print(f"🔢 Generating embeddings for {n:,} documents...")
        emb = self.encoder.encode(
            texts_embed,
            batch_size=Config.EMBEDDING_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        if Config.USE_EMBEDDING_CACHE:
            np.save(cache_path, emb)
            print(f"💾 Saved embeddings cache: {cache_path}")
        return emb


# =============================================================================
# 5) MODEL BUILDING + TRAINING (NATURAL topics/outliers)
# =============================================================================

def build_model(encoder: SentenceTransformer, vectorizer_stopwords: Optional[List[str]] = None) -> BERTopic:
    print("\n" + "=" * 80)
    print("BERTOPIC MODEL CONFIGURATION")
    print("=" * 80)
    print(f"[1/6] Embedding model: {Config.EMBEDDING_MODEL} (pre-encoded)")

    print("\n[2/6] Configuring UMAP:")
    umap_model = UMAP(
        n_neighbors=Config.UMAP_N_NEIGHBORS,
        n_components=Config.UMAP_N_COMPONENTS,
        min_dist=Config.UMAP_MIN_DIST,
        metric=Config.UMAP_METRIC,
        random_state=Config.SEED,
    )
    print(f"    • n_neighbors: {Config.UMAP_N_NEIGHBORS}")
    print(f"    • n_components: {Config.UMAP_N_COMPONENTS}")
    print(f"    • min_dist: {Config.UMAP_MIN_DIST}")
    print(f"    • metric: {Config.UMAP_METRIC}")

    print("\n[3/6] Configuring HDBSCAN:")
    hdbscan_model = HDBSCAN(
        min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=Config.HDBSCAN_MIN_SAMPLES,
        cluster_selection_method=Config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        prediction_data=True,
    )
    print(f"    • min_cluster_size: {Config.HDBSCAN_MIN_CLUSTER_SIZE}")
    print(f"    • min_samples: {Config.HDBSCAN_MIN_SAMPLES}")
    print(f"    • cluster_selection_method: {Config.HDBSCAN_CLUSTER_SELECTION_METHOD}")

    print("\n[4/6] Configuring CountVectorizer:")
    vectorizer_model = CountVectorizer(
        ngram_range=Config.VECTORIZER_NGRAM_RANGE,
        min_df=Config.VECTORIZER_MIN_DF,
        max_features=Config.VECTORIZER_MAX_FEATURES,
        stop_words=vectorizer_stopwords,
    )
    print(f"    • ngram_range: {Config.VECTORIZER_NGRAM_RANGE}")
    print(f"    • min_df: {Config.VECTORIZER_MIN_DF}")
    print(f"    • max_features: {Config.VECTORIZER_MAX_FEATURES}")
    if vectorizer_stopwords:
        print(f"    • stop_words: {len(vectorizer_stopwords)} (extra domain+auto)")

    print("\n[5/6] Configuring c-TF-IDF:")
    ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

    print("\n[6/6] Configuring representation:")
    representation_model = KeyBERTInspired()
    print("    • Model: KeyBERTInspired")

    model = BERTopic(
        embedding_model=encoder,  # only for internal representations/visuals; embeddings are passed explicitly
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        top_n_words=Config.TOP_N_WORDS,
        verbose=True,
        calculate_probabilities=True,
        language="english",
    )
    return model


class ModelTrainer:
    def __init__(self, model: BERTopic):
        self.model = model
        self.topics: Optional[List[int]] = None
        self.probs: Optional[np.ndarray] = None

    def train(self, texts_vec: List[str], embeddings: np.ndarray) -> Tuple[List[int], np.ndarray]:
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        self.topics, self.probs = self.model.fit_transform(texts_vec, embeddings=embeddings)

        n_out = int(np.sum(np.array(self.topics) == -1))
        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        print("\n📈 Initial Results:")
        print(f"   • Topics discovered: {n_topics}")
        print(f"   • Outliers (topic=-1): {n_out:,} ({n_out/len(self.topics)*100:.1f}%)")
        return self.topics, self.probs

    def maybe_reduce_outliers(self, texts_vec: List[str], embeddings: np.ndarray) -> List[int]:
        if not Config.ENABLE_OUTLIER_REASSIGN:
            return self.topics
        if self.topics is None:
            raise RuntimeError("Train before reducing outliers.")
        if -1 not in self.topics:
            print("✅ No outliers to reduce.")
            return self.topics
        print("\n🔧 Controlled outlier reassignment...")
        new_topics = self.model.reduce_outliers(
            texts_vec,
            self.topics,
            strategy=Config.OUTLIER_REASSIGN_STRATEGY,
            embeddings=embeddings,
            threshold=Config.OUTLIER_REASSIGN_THRESHOLD,
        )
        self.model.update_topics(texts_vec, new_topics)
        before = int(np.sum(np.array(self.topics) == -1))
        after = int(np.sum(np.array(new_topics) == -1))
        print(f"   ✅ Outliers: {before:,} → {after:,}")
        self.topics = new_topics
        return new_topics


# =============================================================================
# 6) METRICS (Diversity, Coherence, Stability = ARI bootstrap)
# =============================================================================

class MetricsCalculator:
    def __init__(self, model: BERTopic, texts_vec: List[str], topics: List[int], embeddings: np.ndarray):
        self.model = model
        self.texts = texts_vec
        self.topics = np.asarray(topics)
        self.embeddings = embeddings
        self.metrics: Dict[str, float] = {}

    def calculate_all(self) -> Dict[str, float]:
        print("\n" + "=" * 80)
        print("QUALITY METRICS CALCULATION")
        print("=" * 80)

        tokenized_docs = [doc.split() for doc in self.texts]
        dictionary = Dictionary(tokenized_docs)

        topic_info = self.model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        top_words_per_topic: List[List[str]] = []
        for tid in valid_topics:
            terms = self.model.get_topic(tid) or []
            words: List[str] = []
            for item in terms:
                try:
                    w = item[0] if isinstance(item, (list, tuple)) and len(item) > 0 else item
                    if w is None:
                        continue
                    words.append(str(w))
                except Exception:
                    continue
            top_words_per_topic.append(words)

        print("\n[1/3] Calculating Topic Diversity...")
        self.metrics["diversity"] = self._diversity(top_words_per_topic)
        print(f"    🟣 Topic Diversity: {self.metrics['diversity']:.4f}")

        print("\n[2/3] Calculating Coherence C_v...")
        self.metrics["coherence_cv"] = self._coherence(top_words_per_topic, tokenized_docs, dictionary)
        print(f"    🟢 Coherence C_v: {self.metrics['coherence_cv']:.4f}")

        print("\n[3/3] Calculating Stability (bootstrap ARI on clustering)...")
        self.metrics["stability_ari"] = self._stability_ari()
        print(f"    🟠 Topic Stability (ARI): {self.metrics['stability_ari']:.4f}")

        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        print(f"  Diversity:      {self.metrics['diversity']:.4f}")
        print(f"  Coherence C_v:  {self.metrics['coherence_cv']:.4f}")
        print(f"  Stability (ARI):{self.metrics['stability_ari']:.4f}")
        print("=" * 80 + "\n")
        return self.metrics

    @staticmethod
    def _diversity(top_words_per_topic: List[List[str]]) -> float:
        all_words = [w for topic in top_words_per_topic for w in topic[:Config.TOP_N_WORDS]]
        unique_words = set(all_words)
        total_slots = max(1, len(top_words_per_topic) * Config.TOP_N_WORDS)
        return len(unique_words) / total_slots

    @staticmethod
    def _coherence(top_words: List[List[str]], tokenized_docs: List[List[str]], dictionary: Dictionary) -> float:
        """
        Gensim C_v coherence on cleaned topic token lists.
        We aggressively sanitize topics to avoid the common gensim error:
        'unable to interpret topic as either a list of tokens or a list of ids'.
        """
        try:
            cleaned: List[List[str]] = []
            tok2id = getattr(dictionary, 'token2id', {}) or {}

            for topic in top_words:
                # Ensure iterable of tokens
                if not isinstance(topic, (list, tuple)):
                    continue
                toks: List[str] = []
                for t in topic[:Config.TOP_N_WORDS]:
                    if t is None:
                        continue
                    # Coerce to str and strip
                    s = str(t).strip()
                    if not s:
                        continue
                    toks.append(s)

                # Keep only tokens present in dictionary (gensim is much happier)
                toks = [t for t in toks if t in tok2id]
                if len(toks) >= 2:
                    cleaned.append(toks)

            if len(cleaned) < 3:
                print(f"    ⚠️  Coherence skipped: too few valid topics after cleaning ({len(cleaned)}).")
                return 0.0

            cm = CoherenceModel(topics=cleaned, texts=tokenized_docs, dictionary=dictionary, coherence="c_v")
            return float(cm.get_coherence())
        except Exception as e:
            print(f"    ⚠️  Error calculating coherence: {e}")
            return 0.0

    def _stability_ari(self) -> float:
        """
        Stability via bootstrap: re-fit UMAP+HDBSCAN on subsamples of embeddings
        and compare labels to the original labels on the same documents using ARI.
        This avoids BERTopic backend pitfalls and gives a reproducible robustness score.
        """
        try:
            labels_base = self.topics.copy()

            if Config.STABILITY_IGNORE_OUTLIERS:
                keep = labels_base != -1
                labels_base = labels_base[keep]
                emb_base = self.embeddings[keep]
            else:
                emb_base = self.embeddings

            n = len(labels_base)
            if n < 50:
                return 0.0

            bs_n = max(50, int(n * Config.STABILITY_SAMPLE_FRAC))
            aris: List[float] = []

            for b in range(Config.STABILITY_BOOTSTRAPS):
                rs = Config.SEED + 1000 + b
                idx = resample(np.arange(n), n_samples=bs_n, replace=False, random_state=rs)
                emb_b = emb_base[idx]
                labels_ref = labels_base[idx]

                umap_b = UMAP(
                    n_neighbors=Config.UMAP_N_NEIGHBORS,
                    n_components=Config.UMAP_N_COMPONENTS,
                    min_dist=Config.UMAP_MIN_DIST,
                    metric=Config.UMAP_METRIC,
                    random_state=rs,
                )
                red_b = umap_b.fit_transform(emb_b)

                hdb_b = HDBSCAN(
                    min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE,
                    min_samples=Config.HDBSCAN_MIN_SAMPLES,
                    cluster_selection_method=Config.HDBSCAN_CLUSTER_SELECTION_METHOD,
                    prediction_data=False,
                )
                labels_b = hdb_b.fit_predict(red_b)

                if Config.STABILITY_IGNORE_OUTLIERS:
                    m = labels_b != -1
                    if m.sum() < 30:
                        continue
                    ari = adjusted_rand_score(labels_ref[m], labels_b[m])
                else:
                    ari = adjusted_rand_score(labels_ref, labels_b)

                aris.append(float(ari))

            return float(np.mean(aris)) if aris else 0.0
        except Exception as e:
            print(f"    ⚠️  Error calculating stability: {e}")
            return 0.0


# =============================================================================
# 7) EXPORTS
# =============================================================================

class Exporter:
    def __init__(self, model: BERTopic, df: pd.DataFrame, topics: List[int], probs: Optional[np.ndarray], results_dir: Path):
        self.model = model
        self.df = df.reset_index(drop=True)
        self.topics = topics
        self.probs = probs
        self.results_dir = results_dir

    def export_all(self):
        print("\n" + "=" * 80)
        print("DATA EXPORT")
        print("=" * 80)

        df_aligned = self.df.copy()
        df_aligned["Topic"] = self.topics

        info = self.model.get_topic_info()
        name_map = dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))
        df_aligned["Topic_Name"] = df_aligned["Topic"].map(name_map).fillna("Outlier")

        df_aligned["Probability"] = self._extract_probabilities()

        self._export_full_assignments(df_aligned)
        self._export_outliers(df_aligned)
        self._export_top_papers(df_aligned)
        self._export_topic_metadata()
        self._export_summary_stats(df_aligned)

        print("\n✅ All exports complete")
        print("=" * 80 + "\n")

    def _extract_probabilities(self) -> List[float]:
        if self.probs is None:
            return [0.0] * len(self.topics)
        if getattr(self.probs, "ndim", 1) == 1:
            return [float(x) for x in self.probs]
        # robust: use max probability as "relevance"
        out = []
        for row, t in zip(self.probs, self.topics):
            out.append(float(np.max(row)) if t != -1 else 0.0)
        return out

    def _export_full_assignments(self, df: pd.DataFrame):
        out = self.results_dir / "01_Full_Document_Assignments.csv"
        cols = ["Title", "Year", "Authors", "Abstract", "Document Type",
                "Topic", "Topic_Name", "Probability", "DOI", "Cited by",
                "Author Keywords", "Index Keywords"]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(out, index=False)
        print(f"\n[1/5] ✅ Saved: {out}")
        print(f"      {len(df):,} documents with topic assignments")

    def _export_outliers(self, df: pd.DataFrame):
        outliers = df[df["Topic"] == -1].copy()
        if outliers.empty:
            print("[2/5] ✅ No outliers to export (topic=-1 not present).")
            return
        out = self.results_dir / "01b_Outlier_Documents.csv"
        cols = ["Title", "Year", "Authors", "Abstract", "DOI", "Cited by", "Probability"]
        cols = [c for c in cols if c in outliers.columns]
        outliers[cols].to_csv(out, index=False)
        print(f"[2/5] ✅ Saved: {out} ({len(outliers):,} outliers)")

    def _export_top_papers(self, df: pd.DataFrame):
        out = self.results_dir / "02_Top_Papers_Per_Topic.csv"
        info = self.model.get_topic_info()
        valid = info[info["Topic"] != -1]["Topic"].tolist()

        top_docs = []
        for tid in valid:
            d = df[df["Topic"] == tid].copy()
            d = d.sort_values("Probability", ascending=False).head(Config.TOP_N_PAPERS_PER_TOPIC)
            d["Topic_ID"] = tid
            d["Rank"] = range(1, len(d) + 1)
            top_docs.append(d)

        if top_docs:
            df_top = pd.concat(top_docs, ignore_index=True)
            cols = ["Topic_ID", "Topic_Name", "Rank", "Title", "Year", "Authors", "Abstract",
                    "Probability", "Cited by", "DOI"]
            cols = [c for c in cols if c in df_top.columns]
            df_top[cols].to_csv(out, index=False)
            print(f"\n[3/5] ✅ Saved: {out}")
            print(f"      Top {Config.TOP_N_PAPERS_PER_TOPIC} papers × {len(valid)} topics")
        else:
            print("\n[3/5] ⚠️ No topics to export top papers.")

    def _export_topic_metadata(self):
        out = self.results_dir / "03_Topic_Metadata_Keywords.csv"
        info = self.model.get_topic_info().copy()
        reps = []
        repdocs = []
        for tid in info["Topic"].tolist():
            topic_terms = self.model.get_topic(tid)
            reps.append([w for w, _ in topic_terms] if topic_terms else [])
            docs = self.model.get_representative_docs(tid)
            repdocs.append(docs[:3] if isinstance(docs, list) else [])
        info["Representation"] = reps
        info["Representative_Docs"] = repdocs
        info.to_csv(out, index=False)
        print(f"\n[4/5] ✅ Saved: {out}")

    def _export_summary_stats(self, df: pd.DataFrame):
        out = self.results_dir / "04_Summary_Statistics.csv"
        rows = []
        for tid, g in df.groupby("Topic"):
            name = g["Topic_Name"].iloc[0] if "Topic_Name" in g.columns else str(tid)
            rows.append({
                "Topic_ID": int(tid),
                "Topic_Name": name,
                "N_Documents": int(len(g)),
                "Percentage": float(len(g) / len(df) * 100.0),
                "Avg_Probability": float(g["Probability"].mean()) if "Probability" in g.columns else 0.0,
                "Avg_Citations": float(g["Cited by"].mean()) if "Cited by" in g.columns else 0.0,
                "Year_Range": f"{int(g['Year'].min())}-{int(g['Year'].max())}" if "Year" in g.columns else "",
                "Peak_Year": int(g["Year"].value_counts().idxmax()) if "Year" in g.columns else None,
            })
        pd.DataFrame(rows).sort_values(["Topic_ID"]).to_csv(out, index=False)
        print(f"\n[5/5] ✅ Saved: {out}")


# =============================================================================
# 8) TOPICS OVER TIME (ROBUST) + VISUALS
# =============================================================================

def topics_over_time_table(df_assign: pd.DataFrame, results_dir: Path) -> pd.DataFrame:
    df = df_assign.copy()
    df = df[df["Year"].between(2000, Config.TIME_CUTOFF_YEAR)].copy()
    if not Config.INCLUDE_OUTLIERS_IN_TIMEPLOTS:
        df = df[df["Topic"] != -1].copy()

    grp = df.groupby(["Year", "Topic", "Topic_Name"]).size().reset_index(name="Count")
    totals = df.groupby("Year").size().reset_index(name="Total")
    out = grp.merge(totals, on="Year", how="left")
    out["Share"] = out["Count"] / out["Total"]
    out_path = results_dir / "06_Topics_Over_Time.csv"
    out.to_csv(out_path, index=False)
    print(f"✅ Saved topics over time table: {out_path}")
    return out


def make_time_plots(table: pd.DataFrame, figures_dir: Path):
    # Avoid matplotlib; use plotly if available
    try:
        import plotly.express as px

        # 1) Top-N line plot (readable)
        df = table.copy()
        # Rank topics by total volume across years (excluding outliers unless requested)
        rank = df.groupby(["Topic", "Topic_Name"])["Count"].sum().reset_index()
        if not Config.INCLUDE_OUTLIERS_IN_TIMEPLOTS:
            rank = rank[rank["Topic"] != -1]
        rank = rank.sort_values("Count", ascending=False)
        top = rank.head(Config.TOP_N_TOPICS_LINEPLOT)["Topic"].tolist()

        df_top = df[df["Topic"].isin(top)].copy()
        fig = px.line(df_top, x="Year", y="Share", color="Topic_Name",
                      title=f"Topics over time (Top {len(top)} by volume) — up to {Config.TIME_CUTOFF_YEAR}")
        fig.write_html(figures_dir / f"Viz_05_Topics_Over_Time_Top{len(top)}_to_{Config.TIME_CUTOFF_YEAR}.html")

        # 2) Heatmap for ALL topics (solves 'few topics shown' complaint)
        # Pivot Share (Topic x Year)
        piv = df.pivot_table(index=["Topic", "Topic_Name"], columns="Year", values="Share", fill_value=0.0)
        # Sort topics by overall volume
        totals = df.groupby(["Topic", "Topic_Name"])["Count"].sum()
        piv = piv.loc[totals.sort_values(ascending=False).index]

        piv2 = piv.reset_index()
        melt = piv2.melt(id_vars=["Topic", "Topic_Name"], var_name="Year", value_name="Share")
        fig2 = px.density_heatmap(
            melt, x="Year", y="Topic_Name", z="Share",
            title=f"Topics over time (ALL topics heatmap) — up to {Config.TIME_CUTOFF_YEAR}"
        )
        fig2.update_layout(yaxis={"categoryorder": "array", "categoryarray": melt["Topic_Name"].unique()})
        fig2.write_html(figures_dir / f"Viz_06_Topics_Over_Time_Heatmap_to_{Config.TIME_CUTOFF_YEAR}.html")

        print("✅ Saved time series visuals (Top-N line + ALL-topics heatmap)")
    except Exception as e:
        print(f"⚠️  Could not generate plotly time visuals: {e}")


# =============================================================================
# 9) REPORT + QUALITY METRICS EXPORT
# =============================================================================

def export_quality_metrics(df_assign: pd.DataFrame, metrics: Dict[str, float], results_dir: Path) -> None:
    if not Config.EXPORT_QUALITY_METRICS:
        return
    topics = df_assign["Topic"].tolist()
    n_out = int(np.sum(np.array(topics) == -1))
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    row = {
        "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "seed": Config.SEED,
        "run_id": Config.RUN_ID,
        "n_documents": int(len(df_assign)),
        "n_topics": int(n_topics),
        "n_outliers": int(n_out),
        "outlier_pct": float(n_out/len(df_assign)*100.0),
        "diversity": float(metrics.get("diversity", 0.0)),
        "coherence_cv": float(metrics.get("coherence_cv", 0.0)),
        "stability_ari": float(metrics.get("stability_ari", 0.0)),
        "umap_n_neighbors": Config.UMAP_N_NEIGHBORS,
        "umap_n_components": Config.UMAP_N_COMPONENTS,
        "umap_min_dist": Config.UMAP_MIN_DIST,
        "umap_metric": Config.UMAP_METRIC,
        "hdbscan_min_cluster_size": Config.HDBSCAN_MIN_CLUSTER_SIZE,
        "hdbscan_min_samples": Config.HDBSCAN_MIN_SAMPLES,
        "hdbscan_cluster_selection_method": Config.HDBSCAN_CLUSTER_SELECTION_METHOD,
        "vectorizer_ngram_range": str(Config.VECTORIZER_NGRAM_RANGE),
        "vectorizer_min_df": Config.VECTORIZER_MIN_DF,
        "vectorizer_max_features": Config.VECTORIZER_MAX_FEATURES,
        "outlier_reassign_enabled": Config.ENABLE_OUTLIER_REASSIGN,
        "outlier_reassign_strategy": Config.OUTLIER_REASSIGN_STRATEGY,
        "outlier_reassign_threshold": Config.OUTLIER_REASSIGN_THRESHOLD,
    }
    out = results_dir / "05_Quality_Metrics.csv"
    pd.DataFrame([row]).to_csv(out, index=False)
    print(f"✅ Saved quality metrics: {out}")


def write_final_report(results_dir: Path, figures_dir: Path, n_docs: int, n_topics: int, n_out: int, metrics: Dict[str, float]) -> None:
    out = results_dir / "00_Analysis_Report.txt"
    with out.open("w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CIRP BIBLIOMETRIC ANALYSIS - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Pipeline Version: v6.5 (natural, fixed coherence + stopwords)\n")
        f.write(f"Random Seed: {Config.SEED}\n")
        f.write(f"Run ID: {Config.RUN_ID}\n\n")

        f.write("=" * 80 + "\nDATA SUMMARY\n" + "=" * 80 + "\n")
        f.write(f"Total documents analyzed: {n_docs}\n")
        f.write(f"Time coverage: 2000 - {Config.TIME_CUTOFF_YEAR}\n\n")

        f.write("=" * 80 + "\nTOPIC MODELING RESULTS\n" + "=" * 80 + "\n")
        f.write(f"Topics discovered (final, excl. -1): {n_topics}\n")
        f.write(f"Outlier documents (topic=-1): {n_out} ({n_out/n_docs*100:.1f}%)\n\n")

        f.write("=" * 80 + "\nQUALITY METRICS\n" + "=" * 80 + "\n")
        f.write(f"Topic Diversity: {metrics.get('diversity', 0.0):.4f}\n")
        f.write(f"Coherence C_v: {metrics.get('coherence_cv', 0.0):.4f}\n")
        f.write(f"Topic Stability (ARI): {metrics.get('stability_ari', 0.0):.4f}\n\n")

        f.write("=" * 80 + "\nMODEL CONFIGURATION\n" + "=" * 80 + "\n")
        f.write(f"Embedding Model: {Config.EMBEDDING_MODEL}\n")
        f.write(f"UMAP: n_neighbors={Config.UMAP_N_NEIGHBORS}, n_components={Config.UMAP_N_COMPONENTS}, "
                f"min_dist={Config.UMAP_MIN_DIST}, metric={Config.UMAP_METRIC}\n")
        f.write(f"HDBSCAN: min_cluster_size={Config.HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={Config.HDBSCAN_MIN_SAMPLES}, "
                f"cluster_selection_method={Config.HDBSCAN_CLUSTER_SELECTION_METHOD}\n")
        f.write(f"Vectorizer: ngram_range={Config.VECTORIZER_NGRAM_RANGE}, min_df={Config.VECTORIZER_MIN_DF}, "
                f"max_features={Config.VECTORIZER_MAX_FEATURES}\n")
        f.write(f"Outlier reassignment: {Config.ENABLE_OUTLIER_REASSIGN} (strategy={Config.OUTLIER_REASSIGN_STRATEGY}, "
                f"threshold={Config.OUTLIER_REASSIGN_THRESHOLD})\n\n")

        f.write("=" * 80 + "\nOUTPUT FILES\n" + "=" * 80 + "\n")
        f.write("Results:\n")
        f.write("  • 00_Analysis_Report.txt\n")
        f.write("  • 01_Full_Document_Assignments.csv\n")
        f.write("  • 01b_Outlier_Documents.csv (if outliers exist)\n")
        f.write("  • 02_Top_Papers_Per_Topic.csv\n")
        f.write("  • 03_Topic_Metadata_Keywords.csv\n")
        f.write("  • 04_Summary_Statistics.csv\n")
        f.write("  • 05_Quality_Metrics.csv\n")
        f.write("  • 06_Topics_Over_Time.csv\n")
        f.write("  • 07_Top_Tokens.csv (optional)\n\n")

        f.write("Visualizations:\n")
        f.write("  • Viz_01_Intertopic_Distance_Map.html\n")
        f.write("  • Viz_02_Topic_Word_Scores.html\n")
        f.write("  • Viz_03_Hierarchical_Clustering.html\n")
        f.write("  • Viz_04_Topic_Similarity_Heatmap.html\n")
        f.write(f"  • Viz_05_Topics_Over_Time_Top{Config.TOP_N_TOPICS_LINEPLOT}_to_{Config.TIME_CUTOFF_YEAR}.html\n")
        f.write(f"  • Viz_06_Topics_Over_Time_Heatmap_to_{Config.TIME_CUTOFF_YEAR}.html\n")

    print(f"📄 Final report saved: {out}")


# =============================================================================
# 10) MAIN
# =============================================================================

def main():
    print("\n" + "=" * 80)
    print("CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v6.5")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    set_seed(Config.SEED)
    print(f"🔒 Random seed set to {Config.SEED}")

    results_dir = Config.OUTPUT_BASE / f"run_{Config.RUN_ID}"
    figures_dir = Config.FIGURES_BASE / f"run_{Config.RUN_ID}"
    ensure_dirs(results_dir, figures_dir)
    print(f"📁 Output directories ready: {results_dir}, {figures_dir}")

    # Load + filter
    data_path = Config.DATA_DIR / Config.INPUT_FILE
    print(f"📂 Loading dataset: {data_path}")
    df = DataLoader.load(data_path)
    df = DataLoader.filter_research_only(df)

    # Basic validation
    years = df["Year"] if "Year" in df.columns else pd.Series([])
    if len(years) > 0:
        print(f"📅 Temporal Coverage: {int(years.min())} - {int(years.max())}")

    # Preprocess
    print("\n" + "=" * 80)
    print("TEXT PREPROCESSING")
    print("=" * 80)
    pre = TextPreprocessor()
    texts_embed, texts_vec, df2 = pre.build_corpus(df)
    pre.auto_stopwords_from_corpus(texts_vec, results_dir)

    print("\n✅ Preprocessing complete:")
    print(f"   • Final corpus size: {len(df2):,} documents")
    print("=" * 80 + "\n")

    # Embeddings
    emb_gen = EmbeddingGenerator(results_dir)
    embeddings = emb_gen.encode(texts_embed)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Model
    # Vectorizer stopwords: boost diversity/interpretability (does NOT affect clustering)
    vector_sw = sorted(set(pre.auto_stopwords_added) | set(Config.VECTORIZER_EXTRA_STOPWORDS))
    model = build_model(emb_gen.encoder, vectorizer_stopwords=vector_sw)
    trainer = ModelTrainer(model)
    topics, probs = trainer.train(texts_vec, embeddings)
    topics = trainer.maybe_reduce_outliers(texts_vec, embeddings)

    # Metrics
    metrics = MetricsCalculator(model, texts_vec, topics, embeddings).calculate_all()

    # Export
    exporter = Exporter(model, df2, topics, probs, results_dir)
    exporter.export_all()

    df_assign = pd.read_csv(results_dir / "01_Full_Document_Assignments.csv")

    export_quality_metrics(df_assign, metrics, results_dir)

    # Visuals (BERTopic html)
    print("\n" + "=" * 80)
    print("VISUALIZATION GENERATION")
    print("=" * 80)
    try:
        model.visualize_topics().write_html(figures_dir / "Viz_01_Intertopic_Distance_Map.html")
        model.visualize_barchart(top_n_topics=30).write_html(figures_dir / "Viz_02_Topic_Word_Scores.html")
        model.visualize_hierarchy().write_html(figures_dir / "Viz_03_Hierarchical_Clustering.html")
        model.visualize_heatmap().write_html(figures_dir / "Viz_04_Topic_Similarity_Heatmap.html")
        print("✅ Saved BERTopic visuals (01-04)")
    except Exception as e:
        print(f"⚠️  Error generating BERTopic visuals: {e}")

    # Topics over time (robust)
    tot = topics_over_time_table(df_assign, results_dir)
    make_time_plots(tot, figures_dir)

    # Final report
    n_out = int((df_assign["Topic"] == -1).sum())
    n_topics = df_assign["Topic"].nunique() - (1 if -1 in df_assign["Topic"].unique() else 0)
    write_final_report(results_dir, figures_dir, len(df_assign), n_topics, n_out, metrics)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {Config.SEED}")
    print(f"Run folder: {results_dir}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
