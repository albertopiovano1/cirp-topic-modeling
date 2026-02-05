"""
================================================================================
CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v6.3 (natural topics/outliers)
================================================================================
Advanced Topic Modeling for Manufacturing Science & Technology Research
Target: CIRP Journal of Manufacturing Science and Technology (JMST)

Main upgrades vs v5.0
- Enforced "research-only" corpus: reviews/editorials/errata removed (configurable)
- Fewer, more interpretable topics (stronger clustering; **no forced topic count**)
- Metrics: diversity + coherence + FIXED stability (bootstrap on topic centroids)
- Outliers left to HDBSCAN by default (no forced 0% outliers)
================================================================================
"""

import os
import re
import random
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import spacy
import torch
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from tqdm import tqdm
from umap import UMAP

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from bertopic.vectorizers import ClassTfidfTransformer
from sentence_transformers import SentenceTransformer

warnings.filterwarnings("ignore")

# ============================================================================
# 1) CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class Config:
    # Paths
    DATA_DIR: Path = Path("data")
    RESULTS_DIR: Path = Path("results")
    FIGURES_DIR: Path = Path("figures")

    # Input file
    INPUT_FILE: str = "CIRP_researchonly.csv"

    # Reproducibility
    SEED: int = 76

    # --- Corpus filtering (research-only) ---
    # Keep only these types (case-insensitive); if empty, do not "include" filter.
    INCLUDE_DOC_TYPES: Tuple[str, ...] = ("article",)
    # Remove these types (case-insensitive) even if present in INCLUDE_DOC_TYPES
    EXCLUDE_DOC_TYPES: Tuple[str, ...] = ("review", "editorial", "erratum", "correction", "retracted", "note")

    # Text preprocessing
    MIN_WORDS_PER_DOC: int = 10          # Minimum words after cleaning
    MIN_ABSTRACT_LENGTH: int = 50        # Minimum abstract length in words

    # --- BERTopic parameters (biased toward fewer, cleaner topics) ---
    EMBEDDING_MODEL: str = "BAAI/bge-large-en-v1.5"

    # UMAP (a bit more global structure; less fragmentation)
    UMAP_N_NEIGHBORS: int = 25
    UMAP_N_COMPONENTS: int = 5
    UMAP_MIN_DIST: float = 0.05

    # HDBSCAN (fewer micro-topics, higher purity)
    HDBSCAN_MIN_CLUSTER_SIZE: int = 30
    HDBSCAN_MIN_SAMPLES: int = 2

    # Vectorizer (avoid noisy trigrams by default)
    VECTORIZER_NGRAM_RANGE: Tuple[int, int] = (1, 2)
    VECTORIZER_MIN_DF: int = 10
    VECTORIZER_MAX_FEATURES: int = 12000

    TOP_N_WORDS: int = 12

    # Topic count is **not** forced.
    # The number of topics is determined naturally by the embedding space + HDBSCAN.
    # (No post-hoc reduction / merging.)

    # Outlier handling
    # Outliers are also determined naturally by HDBSCAN.
    # If you later want a conservative reassignment, switch this on and keep a high threshold.
    ENABLE_OUTLIER_REASSIGN: bool = False
    OUTLIER_REASSIGN_STRATEGY: str = "embeddings"   # "embeddings" usually best for scientific corpora
    OUTLIER_REASSIGN_THRESHOLD: float = 0.30        # more conservative: keeps true outliers unless very clear

    # Analysis parameters
    TIME_CUTOFF_YEAR: int = 2025
    TOP_N_TOPICS_VISUALIZATION: int = 16
    TOP_N_PAPERS_PER_TOPIC: int = 5

    # Stability bootstrap (centroid similarity)
    STABILITY_BOOTSTRAPS: int = 5
    STABILITY_SAMPLE_FRAC: float = 0.80
    STABILITY_RANDOM_STATE_OFFSET: int = 1000

    # Exports (paper-ready bookkeeping)
    EXPORT_QUALITY_METRICS: bool = True
    QUALITY_METRICS_FILE: str = "05_Quality_Metrics.csv"
    TOPICS_OVER_TIME_CSV: str = "06_Topics_Over_Time.csv"
    TOPICS_OVER_TIME_TOP_N: int = 12


def set_seed(seed: int = Config.SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"🔒 Random seed set to {seed}")


def create_directories():
    for directory in [Config.DATA_DIR, Config.RESULTS_DIR, Config.FIGURES_DIR]:
        directory.mkdir(exist_ok=True)
    print(f"📁 Output directories ready: {Config.RESULTS_DIR}, {Config.FIGURES_DIR}")


def filter_research_only(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce research-only corpus using Document Type column (case-insensitive)."""
    if "Document Type" not in df.columns:
        print("⚠️  'Document Type' column not found → skipping research-only filter.")
        return df

    dt = df["Document Type"].fillna("").astype(str).str.strip().str.lower()

    if Config.INCLUDE_DOC_TYPES:
        include_mask = dt.isin(set(Config.INCLUDE_DOC_TYPES))
    else:
        include_mask = pd.Series(True, index=df.index)

    exclude_mask = dt.apply(lambda x: any(bad in x for bad in Config.EXCLUDE_DOC_TYPES))
    kept = df[include_mask & (~exclude_mask)].copy()

    removed = len(df) - len(kept)
    print(f"🧹 Research-only filter: kept {len(kept):,}/{len(df):,} docs; removed {removed:,} (reviews/editorials/other).")
    return kept


# ============================================================================
# 2) DATA VALIDATION
# ============================================================================

class DataValidator:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report: Dict = {}

    def validate_and_report(self) -> Dict:
        print("\n" + "=" * 80)
        print("DATA VALIDATION & QUALITY REPORT")
        print("=" * 80)

        self._check_basic_structure()
        self._check_coverage()
        self._check_document_types()
        self._check_duplicates()
        self._check_temporal_distribution()
        self._check_content_quality()
        self._print_summary()
        return self.report

    def _check_basic_structure(self):
        required_cols = ["Title", "Abstract", "Year"]
        missing = [c for c in required_cols if c not in self.df.columns]

        self.report["total_documents"] = len(self.df)
        self.report["total_columns"] = len(self.df.columns)
        self.report["missing_required_columns"] = missing

        print("\n📊 Basic Structure:")
        print(f"  • Total documents: {len(self.df):,}")
        print(f"  • Total columns: {len(self.df.columns)}")
        if missing:
            print(f"  ⚠️  Missing required columns: {missing}")

    def _check_coverage(self):
        key_fields = ["Title", "Abstract", "Author Keywords", "Index Keywords", "DOI"]
        coverage = {}

        print("\n📈 Data Coverage:")
        for field in key_fields:
            if field in self.df.columns:
                non_null = self.df[field].notna().sum()
                pct = (non_null / len(self.df)) * 100
                coverage[field] = {"count": int(non_null), "percentage": round(pct, 2)}
                print(f"  • {field}: {non_null:,} ({pct:.1f}%)")

        self.report["coverage"] = coverage

    def _check_document_types(self):
        if "Document Type" in self.df.columns:
            dist = self.df["Document Type"].fillna("NA").value_counts().to_dict()
            self.report["document_types"] = dist
            print("\n📑 Document Types:")
            for k, v in dist.items():
                print(f"  • {k}: {v:,} ({v/len(self.df)*100:.1f}%)")

    def _check_duplicates(self):
        duplicates_doi = int(self.df["DOI"].duplicated().sum()) if "DOI" in self.df.columns else 0
        duplicates_title = int(self.df["Title"].duplicated().sum()) if "Title" in self.df.columns else 0

        self.report["duplicates"] = {"by_doi": duplicates_doi, "by_title": duplicates_title}

        print("\n🔍 Duplicate Detection:")
        print(f"  • Duplicate DOIs: {duplicates_doi}")
        print(f"  • Duplicate Titles: {duplicates_title}")

    def _check_temporal_distribution(self):
        if "Year" not in self.df.columns:
            return
        year_stats = {
            "min": int(self.df["Year"].min()),
            "max": int(self.df["Year"].max()),
            "mean": round(float(self.df["Year"].mean()), 1),
            "median": int(self.df["Year"].median()),
        }
        self.report["temporal_distribution"] = year_stats

        print("\n📅 Temporal Coverage:")
        print(f"  • Period: {year_stats['min']} - {year_stats['max']}")
        print(f"  • Mean year: {year_stats['mean']}")
        print(f"  • Median year: {year_stats['median']}")

        tmp = self.df.copy()
        tmp["Decade"] = (tmp["Year"] // 10) * 10
        decade_dist = tmp["Decade"].value_counts().sort_index()
        print("\n  Distribution by decade:")
        for decade, count in decade_dist.items():
            print(f"    {decade}s: {count:,} papers")

    def _check_content_quality(self):
        self.df["Abstract_Length"] = self.df["Abstract"].fillna("").astype(str).str.split().str.len()
        length_stats = {
            "mean": round(float(self.df["Abstract_Length"].mean()), 1),
            "median": int(self.df["Abstract_Length"].median()),
            "min": int(self.df["Abstract_Length"].min()),
            "max": int(self.df["Abstract_Length"].max()),
        }
        short_abs = int((self.df["Abstract_Length"] < Config.MIN_ABSTRACT_LENGTH).sum())

        self.report["content_quality"] = {"abstract_length": length_stats, "short_abstracts": short_abs}

        print("\n📝 Content Quality:")
        print("  • Abstract length (words):")
        print(f"    - Mean: {length_stats['mean']}")
        print(f"    - Median: {length_stats['median']}")
        print(f"    - Range: {length_stats['min']} - {length_stats['max']}")
        print(f"  • Short abstracts (<{Config.MIN_ABSTRACT_LENGTH} words): {short_abs}")

    def _print_summary(self):
        print("\n" + "=" * 80)
        print("VALIDATION SUMMARY")
        print("=" * 80)

        issues = 0
        if self.report.get("missing_required_columns"):
            issues += 1
        if self.report.get("duplicates", {}).get("by_doi", 0) > 10:
            issues += 1
        if self.report.get("content_quality", {}).get("short_abstracts", 0) > len(self.df) * 0.05:
            issues += 1

        if issues == 0:
            print("✅ Dataset quality: EXCELLENT - Ready for analysis")
        elif issues == 1:
            print("⚠️  Dataset quality: GOOD - Minor issues detected")
        else:
            print("❌ Dataset quality: FAIR - Review recommended before proceeding")

        print("=" * 80 + "\n")


# ============================================================================
# 3) TEXT PREPROCESSING (dual output)
# ============================================================================

class TextPreprocessor:
    def __init__(self, synonym_mode: str = "safe"):
        self.synonym_mode = synonym_mode
        self._load_spacy()
        self._setup_synonym_maps()
        self._setup_stopwords()

    def _load_spacy(self):
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            print("✅ spaCy model loaded: en_core_web_sm")
        except Exception:
            print("❌ spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

    def _setup_synonym_maps(self):
        self.safe_synonyms = {
            "ml": "machine_learning",
            "ai": "artificial_intelligence",
            "dl": "deep_learning",
            "nn": "neural_network",
            "iot": "internet_of_things",
            "iiot": "industrial_internet_of_things",
            "cps": "cyber_physical_system",
            "am": "additive_manufacturing",
            "3d_printing": "additive_manufacturing",
            "3d_print": "additive_manufacturing",
            "dt": "digital_twin",
            "dts": "digital_twin",
            "digital_twins": "digital_twin",
        }
        self.aggressive_synonyms = {
            "optimisation": "optimize",
            "optimization": "optimize",
            "modelling": "modeling",
            "analyse": "analyze",
            "analysis": "analyze",
        }

        if self.synonym_mode == "safe":
            self.active_synonyms = self.safe_synonyms
            print(f"📝 Synonym normalization: SAFE mode ({len(self.safe_synonyms)} mappings)")
        elif self.synonym_mode == "aggressive":
            self.active_synonyms = {**self.safe_synonyms, **self.aggressive_synonyms}
            print(f"📝 Synonym normalization: AGGRESSIVE mode ({len(self.active_synonyms)} mappings)")
        else:
            self.active_synonyms = {}
            print("📝 Synonym normalization: OFF")

    def _setup_stopwords(self):
        stopwords_academic = [
            "aim", "paper", "study", "result", "conclusion", "method", "methodology",
            "data", "review", "literature", "respectively", "significant", "significantly",
            "show", "demonstrate", "include", "obtain", "use", "using", "utilize",
            "proposed", "based", "approach", "application", "conducted", "investigated",
            "discussed", "finding", "suggest", "present", "introduction", "future", "work",
            "limitation", "implication", "case", "research", "journal", "article",
            "systematic", "survey", "et", "al", "etc"
        ]
        stopwords_domain = ["industry", "product", "system", "solution", "process", "production", "company", "business"]

        custom_stopwords = set([w.lower().strip() for w in (stopwords_academic + stopwords_domain)])
        self.nlp.Defaults.stop_words |= custom_stopwords
        self.stopwords_list = list(self.nlp.Defaults.stop_words)
        print(f"🛑 Custom stopwords loaded: {len(custom_stopwords)} terms")

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
        doc = self.nlp(text)
        tokens = []
        for token in doc:
            if not (token.is_alpha or "_" in token.text):
                continue
            if token.is_stop:
                continue
            if len(token.text) < 3:
                continue
            lemma = token.lemma_.lower()
            if lemma in self.active_synonyms:
                lemma = self.active_synonyms[lemma]
            tokens.append(lemma)
        return " ".join(tokens)

    def process_corpus(self, titles: List[str], abstracts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        print("\n" + "=" * 80)
        print("TEXT PREPROCESSING")
        print("=" * 80)

        combined = []
        for t, a in zip(titles, abstracts):
            combined.append(f"{str(t) if pd.notna(t) else ''}. {str(a) if pd.notna(a) else ''}")

        print(f"📄 Processing {len(combined):,} documents...")

        print("  [1/4] Basic cleaning...")
        cleaned_embed = [self.clean_text_for_embedding(x) for x in tqdm(combined, desc="Cleaning")]
        cleaned_vec = [self.clean_text_for_vectorizer(x) for x in tqdm(combined, desc="Deep cleaning")]

        print(f"  [2/4] Filtering documents (min {Config.MIN_WORDS_PER_DOC} words)...")
        valid_idx = [i for i, x in enumerate(cleaned_embed) if len(x.split()) >= Config.MIN_WORDS_PER_DOC]
        print(f"    ✅ Valid documents: {len(valid_idx):,} / {len(combined):,}")
        print(f"    ❌ Filtered out: {len(combined) - len(valid_idx):,}")

        print("  [3/4] Lemmatization & normalization...")
        text_embed, text_vec = [], []
        for idx in tqdm(valid_idx, desc="Lemmatizing"):
            text_embed.append(self.lemmatize_and_normalize(cleaned_embed[idx]))
            text_vec.append(self.lemmatize_and_normalize(cleaned_vec[idx]))

        print("  [4/4] Final validation...")
        final_idx, final_embed, final_vec = [], [], []
        for te, tv, idx in zip(text_embed, text_vec, valid_idx):
            if len(te.split()) >= 5 and len(tv.split()) >= 5:
                final_embed.append(te)
                final_vec.append(tv)
                final_idx.append(idx)

        print("\n✅ Preprocessing complete:")
        print(f"   • Final corpus size: {len(final_embed):,} documents")
        print(f"   • Dual outputs ready: text_embed + text_vec")
        print("=" * 80 + "\n")
        return final_embed, final_vec, final_idx


# ============================================================================
# 4) MODEL BUILDING
# ============================================================================

class BERTopicModelBuilder:
    def __init__(self, stopwords_list: List[str]):
        self.stopwords_list = stopwords_list
        self.model: Optional[BERTopic] = None
        self.encoder: Optional[SentenceTransformer] = None

    def build_model(self) -> Tuple[BERTopic, SentenceTransformer]:
        print("\n" + "=" * 80)
        print("BERTOPIC MODEL CONFIGURATION")
        print("=" * 80)

        print(f"\n[1/6] Loading embedding model: {Config.EMBEDDING_MODEL}")
        self.encoder = SentenceTransformer(Config.EMBEDDING_MODEL)
        print(f"    ✅ Encoder loaded: {self.encoder.get_sentence_embedding_dimension()}D embeddings")

        print("\n[2/6] Configuring UMAP:")
        print(f"    • n_neighbors: {Config.UMAP_N_NEIGHBORS}")
        print(f"    • n_components: {Config.UMAP_N_COMPONENTS}")
        print(f"    • min_dist: {Config.UMAP_MIN_DIST}")
        umap_model = UMAP(
            n_neighbors=Config.UMAP_N_NEIGHBORS,
            n_components=Config.UMAP_N_COMPONENTS,
            min_dist=Config.UMAP_MIN_DIST,
            metric="cosine",
            random_state=Config.SEED,
        )

        print("\n[3/6] Configuring HDBSCAN:")
        print(f"    • min_cluster_size: {Config.HDBSCAN_MIN_CLUSTER_SIZE}")
        print(f"    • min_samples: {Config.HDBSCAN_MIN_SAMPLES}")
        hdbscan_model = HDBSCAN(
            min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=Config.HDBSCAN_MIN_SAMPLES,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
        )

        print("\n[4/6] Configuring CountVectorizer:")
        print(f"    • ngram_range: {Config.VECTORIZER_NGRAM_RANGE}")
        print(f"    • min_df: {Config.VECTORIZER_MIN_DF}")
        print(f"    • max_features: {Config.VECTORIZER_MAX_FEATURES:,}")
        vectorizer_model = CountVectorizer(
            stop_words=self.stopwords_list,
            ngram_range=Config.VECTORIZER_NGRAM_RANGE,
            min_df=Config.VECTORIZER_MIN_DF,
            max_features=Config.VECTORIZER_MAX_FEATURES,
        )

        print("\n[5/6] Configuring c-TF-IDF:")
        ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)

        print("\n[6/6] Configuring representation:")
        representation_model = KeyBERTInspired()

        print("\n🏗️  Building BERTopic model...")
        self.model = BERTopic(
            embedding_model=self.encoder,  # we still pass it for internal use/visuals
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

        print("✅ Model configuration complete")
        print("=" * 80 + "\n")
        return self.model, self.encoder


# ============================================================================
# 5) TRAINING + OUTLIERS + TOPIC REDUCTION
# ============================================================================

class ModelTrainer:
    def __init__(self, model: BERTopic):
        self.model = model
        self.topics: Optional[List[int]] = None
        self.probs: Optional[np.ndarray] = None

    def train(self, texts_vec: List[str], embeddings: np.ndarray) -> Tuple[List[int], np.ndarray]:
        print("\n" + "=" * 80)
        print("MODEL TRAINING")
        print("=" * 80)
        print(f"📊 Training on {len(texts_vec):,} documents...")
        print(f"   Embedding shape: {embeddings.shape}")

        self.topics, self.probs = self.model.fit_transform(texts_vec, embeddings=embeddings)

        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        n_outliers = sum(1 for t in self.topics if t == -1)
        print("\n📈 Initial Results:")
        print(f"   • Topics discovered: {n_topics}")
        print(f"   • Outliers: {n_outliers:,} ({n_outliers/len(self.topics)*100:.1f}%)")
        return self.topics, self.probs

    def reduce_outliers(self, texts_vec: List[str], embeddings: np.ndarray) -> List[int]:
        if not Config.ENABLE_OUTLIER_REASSIGN:
            return self.topics

        if self.topics is None:
            raise RuntimeError("Train the model before reducing outliers.")

        if -1 not in self.topics:
            print("✅ No outliers to reduce")
            return self.topics

        print("\n🔧 Controlled outlier reassignment...")
        try:
            new_topics = self.model.reduce_outliers(
                texts_vec,
                self.topics,
                strategy=Config.OUTLIER_REASSIGN_STRATEGY,
                embeddings=embeddings,
                threshold=Config.OUTLIER_REASSIGN_THRESHOLD,
            )
            self.model.update_topics(texts_vec, new_topics)
            before = sum(1 for t in self.topics if t == -1)
            after = sum(1 for t in new_topics if t == -1)
            print(f"   ✅ Outliers: {before:,} → {after:,} (kept {after:,} true outliers)")
            self.topics = new_topics
            return new_topics
        except Exception as e:
            print(f"   ⚠️  Error during outlier reassignment: {e}")
            return self.topics

    # NOTE: No forced post-hoc topic reduction.
    # If you later decide to merge topics, do it *after* interpretation and with a documented rationale.

# ============================================================================
# 6) METRICS (Diversity, Coherence, Stability FIXED)
# ============================================================================

class MetricsCalculator:
    def __init__(self, model: BERTopic, texts_vec: List[str], topics: List[int], embeddings: np.ndarray):
        self.model = model
        self.texts = texts_vec
        self.topics = topics
        self.embeddings = embeddings
        self.metrics: Dict[str, float] = {}

    def calculate_all_metrics(self) -> Dict[str, float]:
        print("\n" + "=" * 80)
        print("QUALITY METRICS CALCULATION")
        print("=" * 80)

        tokenized_docs = [doc.split() for doc in self.texts]
        dictionary = Dictionary(tokenized_docs)

        topic_info = self.model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        top_words_per_topic = [[w for w, _ in self.model.get_topic(tid)] for tid in valid_topics]

        print("\n[1/3] Calculating Topic Diversity...")
        self.metrics["diversity"] = self._calculate_diversity(top_words_per_topic)
        print(f"    🟣 Topic Diversity: {self.metrics['diversity']:.4f}")

        print("\n[2/3] Calculating Coherence C_v...")
        self.metrics["coherence_cv"] = self._calculate_coherence(top_words_per_topic, tokenized_docs, dictionary)
        print(f"    🟢 Coherence C_v: {self.metrics['coherence_cv']:.4f}")

        print("\n[3/3] Calculating Stability (bootstrap on centroids)...")
        self.metrics["stability"] = self._calculate_stability_centroids()
        print(f"    🟠 Topic Stability: {self.metrics['stability']:.4f}")

        print("\n" + "=" * 80)
        print("METRICS SUMMARY")
        print("=" * 80)
        print(f"  Diversity:  {self.metrics['diversity']:.4f}")
        print(f"  Coherence:  {self.metrics['coherence_cv']:.4f}")
        print(f"  Stability:  {self.metrics['stability']:.4f}")
        print("=" * 80 + "\n")
        return self.metrics

    @staticmethod
    def _calculate_diversity(top_words_per_topic: List[List[str]]) -> float:
        all_words = [w for topic in top_words_per_topic for w in topic[:Config.TOP_N_WORDS]]
        unique_words = set(all_words)
        total_slots = max(1, len(top_words_per_topic) * Config.TOP_N_WORDS)
        return len(unique_words) / total_slots

    @staticmethod
    def _calculate_coherence(top_words: List[List[str]], tokenized_docs: List[List[str]], dictionary: Dictionary) -> float:
        try:
            cm = CoherenceModel(topics=top_words, texts=tokenized_docs, dictionary=dictionary, coherence="c_v")
            return float(cm.get_coherence())
        except Exception as e:
            print(f"    ⚠️  Error calculating coherence: {e}")
            return 0.0

    @staticmethod
    def _centroids_from_doc_embeddings(topics: List[int], embeddings: np.ndarray) -> Dict[int, np.ndarray]:
        centroids: Dict[int, np.ndarray] = {}
        topics_arr = np.asarray(topics)
        for tid in np.unique(topics_arr):
            if tid == -1:
                continue
            idx = np.where(topics_arr == tid)[0]
            if len(idx) < 2:
                continue
            centroids[int(tid)] = embeddings[idx].mean(axis=0)
        return centroids

    def _calculate_stability_centroids(self) -> float:
        """Bootstrap stability: mean best-match cosine similarity between topic centroids."""
        try:
            base_centroids = self._centroids_from_doc_embeddings(self.topics, self.embeddings)
            if not base_centroids:
                return 0.0

            sims_all: List[float] = []
            n = len(self.texts)
            bs_n = max(10, int(n * Config.STABILITY_SAMPLE_FRAC))

            for b in range(Config.STABILITY_BOOTSTRAPS):
                rs = Config.SEED + Config.STABILITY_RANDOM_STATE_OFFSET + b
                idx = resample(np.arange(n), n_samples=bs_n, replace=True, random_state=rs)

                docs_b = [self.texts[i] for i in idx]
                emb_b = self.embeddings[idx]

                # Fit bootstrap model with embeddings (avoid encode/backend issues)
                model_b = BERTopic(
                    embedding_model=None,
                    umap_model=self.model.umap_model,
                    hdbscan_model=self.model.hdbscan_model,
                    vectorizer_model=self.model.vectorizer_model,
                    ctfidf_model=self.model.ctfidf_model,
                    representation_model=self.model.representation_model,
                    top_n_words=Config.TOP_N_WORDS,
                    verbose=False,
                    calculate_probabilities=False,
                    language="english",
                )
                topics_b, _ = model_b.fit_transform(docs_b, embeddings=emb_b)

                centroids_b = self._centroids_from_doc_embeddings(topics_b, emb_b)
                if not centroids_b:
                    continue

                # Best-match similarity for each original centroid
                for _, v in base_centroids.items():
                    best = 0.0
                    for _, vb in centroids_b.items():
                        s = float(cosine_similarity(v.reshape(1, -1), vb.reshape(1, -1))[0][0])
                        if s > best:
                            best = s
                    sims_all.append(best)

            return float(np.mean(sims_all)) if sims_all else 0.0
        except Exception as e:
            print(f"    ⚠️  Error calculating stability: {e}")
            return 0.0


# ============================================================================
# 7) EXPORTS (robust topic name + prob)
# ============================================================================


# ============================================================================
# 6b) QUALITY METRICS EXPORT (CSV)
# ============================================================================

class QualityExporter:
    @staticmethod
    def export(metrics: Dict[str, float], topics: List[int], validation_report: Dict):
        """Save a one-row CSV with run-level quality metrics + key configuration."""
        if not getattr(Config, "EXPORT_QUALITY_METRICS", True):
            return

        n_docs = len(topics)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        n_outliers = sum(1 for t in topics if t == -1)
        outlier_pct = (n_outliers / n_docs * 100) if n_docs else 0.0

        sizes = pd.Series([t for t in topics if t != -1]).value_counts()
        mean_size = float(sizes.mean()) if len(sizes) else 0.0
        median_size = float(sizes.median()) if len(sizes) else 0.0
        min_size = int(sizes.min()) if len(sizes) else 0
        max_size = int(sizes.max()) if len(sizes) else 0

        td = validation_report.get("temporal_distribution", {})
        row = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "seed": Config.SEED,
            "n_documents": n_docs,
            "n_topics": n_topics,
            "n_outliers": n_outliers,
            "outlier_pct": round(outlier_pct, 4),
            "topic_size_mean": round(mean_size, 4),
            "topic_size_median": round(median_size, 4),
            "topic_size_min": min_size,
            "topic_size_max": max_size,
            "year_min": td.get("min", None),
            "year_max": td.get("max", None),
            "diversity": round(float(metrics.get("diversity", 0.0)), 6),
            "coherence_cv": round(float(metrics.get("coherence_cv", 0.0)), 6),
            "stability": round(float(metrics.get("stability", 0.0)), 6),
            "embedding_model": Config.EMBEDDING_MODEL,
            "umap_n_neighbors": Config.UMAP_N_NEIGHBORS,
            "umap_n_components": Config.UMAP_N_COMPONENTS,
            "umap_min_dist": Config.UMAP_MIN_DIST,
            "hdbscan_min_cluster_size": Config.HDBSCAN_MIN_CLUSTER_SIZE,
            "hdbscan_min_samples": Config.HDBSCAN_MIN_SAMPLES,
            "vectorizer_ngram_range": str(Config.VECTORIZER_NGRAM_RANGE),
            "vectorizer_min_df": Config.VECTORIZER_MIN_DF,
            "vectorizer_max_features": Config.VECTORIZER_MAX_FEATURES,
            "outlier_reassign_enabled": Config.ENABLE_OUTLIER_REASSIGN,
            "outlier_reassign_strategy": Config.OUTLIER_REASSIGN_STRATEGY,
            "outlier_reassign_threshold": Config.OUTLIER_REASSIGN_THRESHOLD,
        }

        out_path = Config.RESULTS_DIR / getattr(Config, "QUALITY_METRICS_FILE", "05_Quality_Metrics.csv")
        pd.DataFrame([row]).to_csv(out_path, index=False)
        print(f"✅ Saved quality metrics: {out_path}")


class DataExporter:
    def __init__(self, model: BERTopic, df_original: pd.DataFrame, valid_indices: List[int],
                 topics: List[int], probs: Optional[np.ndarray]):
        self.model = model
        self.df_original = df_original
        self.valid_indices = valid_indices
        self.topics = topics
        self.probs = probs

    def export_all(self):
        print("\n" + "=" * 80)
        print("DATA EXPORT")
        print("=" * 80)

        df_aligned = self.df_original.iloc[self.valid_indices].reset_index(drop=True)
        df_aligned["Topic"] = self.topics

        # Topic name mapping (robust)
        info = self.model.get_topic_info()
        name_map = dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))
        df_aligned["Topic_Name"] = df_aligned["Topic"].map(name_map).fillna("Outlier")

        # Probability: robust fallback = max probability row
        df_aligned["Probability"] = self._extract_probabilities()

        self._export_full_assignments(df_aligned)
        self._export_top_papers(df_aligned)
        self._export_topic_metadata()
        self._export_summary_stats(df_aligned)

        print("\n✅ All exports complete")
        print("=" * 80 + "\n")

    def _extract_probabilities(self) -> List[float]:
        if self.probs is None:
            return [0.0] * len(self.topics)
        if self.probs.ndim == 1:
            return [float(x) for x in self.probs]
        # Safe: assigned topic probability can be unreliable across versions; use max as relevance score.
        return [float(np.max(row)) if t != -1 else 0.0 for row, t in zip(self.probs, self.topics)]

    def _export_full_assignments(self, df: pd.DataFrame):
        out = Config.RESULTS_DIR / "01_Full_Document_Assignments.csv"
        cols = ["Title", "Year", "Authors", "Abstract", "Document Type",
                "Topic", "Topic_Name", "Probability", "DOI", "Cited by",
                "Author Keywords", "Index Keywords"]
        cols = [c for c in cols if c in df.columns]
        df[cols].to_csv(out, index=False)
        print(f"\n[1/4] ✅ Saved: {out}")
        print(f"      {len(df):,} documents with topic assignments")

    def _export_top_papers(self, df: pd.DataFrame):
        out = Config.RESULTS_DIR / "02_Top_Papers_Per_Topic.csv"
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
            print(f"\n[2/4] ✅ Saved: {out}")
            print(f"      Top {Config.TOP_N_PAPERS_PER_TOPIC} papers × {len(valid)} topics")

    def _export_topic_metadata(self):
        out = Config.RESULTS_DIR / "03_Topic_Metadata_Keywords.csv"
        self.model.get_topic_info().to_csv(out, index=False)
        print(f"\n[3/4] ✅ Saved: {out}")

    def _export_summary_stats(self, df: pd.DataFrame):
        out = Config.RESULTS_DIR / "04_Summary_Statistics.csv"
        stats = []
        for tid in sorted(df["Topic"].unique()):
            if tid == -1:
                continue
            d = df[df["Topic"] == tid]
            stats.append({
                "Topic_ID": int(tid),
                "Topic_Name": d["Topic_Name"].iloc[0],
                "N_Documents": int(len(d)),
                "Percentage": float(len(d) / len(df) * 100),
                "Avg_Probability": float(d["Probability"].mean()),
                "Avg_Citations": float(d["Cited by"].mean()) if "Cited by" in d.columns else 0.0,
                "Year_Range": f"{int(d['Year'].min())}-{int(d['Year'].max())}" if "Year" in d.columns else "",
                "Peak_Year": int(d["Year"].mode().values[0]) if "Year" in d.columns and len(d) else None,
            })
        pd.DataFrame(stats).sort_values("N_Documents", ascending=False).to_csv(out, index=False)
        print(f"\n[4/4] ✅ Saved: {out}")


# ============================================================================
# 8) VISUALS (unchanged API calls; safe-guarded)
# ============================================================================

class VisualizationGenerator:
    def __init__(self, model: BERTopic, texts: List[str], df_aligned: pd.DataFrame):
        self.model = model
        self.texts = texts
        self.df_aligned = df_aligned

    def generate_all(self):
        print("\n" + "=" * 80)
        print("VISUALIZATION GENERATION")
        print("=" * 80)

        self._save_html(self.model.visualize_topics, Config.FIGURES_DIR / "Viz_01_Intertopic_Distance_Map.html")
        self._save_html(lambda: self.model.visualize_barchart(top_n_topics=Config.TOP_N_TOPICS_VISUALIZATION),
                        Config.FIGURES_DIR / "Viz_02_Topic_Word_Scores.html")
        self._save_html(self.model.visualize_hierarchy, Config.FIGURES_DIR / "Viz_03_Hierarchical_Clustering.html")
        self._save_html(self.model.visualize_heatmap, Config.FIGURES_DIR / "Viz_04_Topic_Similarity_Heatmap.html")

        self._generate_topics_over_time()

        print("\n✅ All visualizations generated")
        print("=" * 80 + "\n")

    @staticmethod
    def _save_html(fig_fn, out_path: Path):
        try:
            fig = fig_fn()
            fig.write_html(str(out_path))
            print(f"✅ Saved: {out_path}")
        except Exception as e:
            print(f"⚠️  Error saving {out_path.name}: {e}")

    def _generate_topics_over_time(self):
        """Robust Topics-over-time.
        Always exports CSV; tries HTML via Plotly (falls back gracefully).
        """
        if "Year" not in self.df_aligned.columns:
            print("⚠️  Skipping time series: Year column not found")
            return

        df = self.df_aligned[["Year", "Topic"]].copy()
        df["Year"] = pd.to_numeric(df["Year"], errors="coerce").astype("Int64")
        df = df.dropna(subset=["Year"])
        df = df[df["Year"] <= Config.TIME_CUTOFF_YEAR]

        if df.empty:
            print("⚠️  Skipping time series: no valid years after filtering")
            return

        info = self.model.get_topic_info()
        name_map = dict(zip(info["Topic"].astype(int), info["Name"].astype(str)))
        df["Topic_Name"] = df["Topic"].map(name_map).fillna("Outlier")

        counts = df.groupby(["Year", "Topic", "Topic_Name"]).size().reset_index(name="Count")
        totals = df.groupby("Year").size().reset_index(name="Total")
        tot = counts.merge(totals, on="Year", how="left")
        tot["Share"] = tot["Count"] / tot["Total"]

        out_csv = Config.RESULTS_DIR / getattr(Config, "TOPICS_OVER_TIME_CSV", "06_Topics_Over_Time.csv")
        tot.sort_values(["Year", "Count"], ascending=[True, False]).to_csv(out_csv, index=False)
        print(f"✅ Saved topics over time table: {out_csv}")

        # HTML: top-N topics by overall count (exclude outliers)
        try:
            import plotly.express as px

            df_no_out = tot[tot["Topic"] != -1].copy()
            if df_no_out.empty:
                return

            top_n = getattr(Config, "TOPICS_OVER_TIME_TOP_N", 12)
            top_topics = (
                df_no_out.groupby(["Topic", "Topic_Name"])["Count"].sum()
                .sort_values(ascending=False)
                .head(top_n)
                .reset_index()
            )
            plot_df = df_no_out.merge(top_topics[["Topic"]], on="Topic", how="inner")

            fig = px.line(
                plot_df,
                x="Year",
                y="Share",
                color="Topic_Name",
                title=f"Topics Over Time (share per year) — up to {Config.TIME_CUTOFF_YEAR}",
            )
            out_html = Config.FIGURES_DIR / f"Viz_05_Topics_Over_Time_to_{Config.TIME_CUTOFF_YEAR}.html"
            fig.write_html(str(out_html))
            print(f"✅ Saved: {out_html}")
        except Exception as e:
            print(f"⚠️  Error generating HTML time series: {e}")

# ============================================================================
# 9) MAIN
# ============================================================================

def main():
    print("\n" + "=" * 80)
    print("CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v6.3 (natural)")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    set_seed()
    create_directories()

    input_path = Config.DATA_DIR / Config.INPUT_FILE
    if not input_path.exists():
        alt = Path("/mnt/user-data/uploads") / Config.INPUT_FILE
        if alt.exists():
            input_path = alt
        else:
            raise FileNotFoundError(f"Dataset not found: {Config.INPUT_FILE}")

    print(f"📂 Loading dataset: {input_path}")
    df = pd.read_csv(input_path)

    # Enforce research-only BEFORE validation/modeling
    df = filter_research_only(df)

    validator = DataValidator(df)
    validation_report = validator.validate_and_report()

    # Preprocess
    preprocessor = TextPreprocessor(synonym_mode="safe")
    text_embed, text_vec, valid_indices = preprocessor.process_corpus(
        df["Title"].fillna(""),
        df["Abstract"].fillna("")
    )
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)

    # Build model + encoder
    model_builder = BERTopicModelBuilder(preprocessor.stopwords_list)
    model, encoder = model_builder.build_model()

    # Embeddings from text_embed using raw encoder (avoids backend API drift)
    print("\n" + "=" * 80)
    print("EMBEDDING GENERATION")
    print("=" * 80)
    print(f"🔢 Generating embeddings for {len(text_embed):,} documents...")
    embeddings = encoder.encode(
        text_embed,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    embeddings = np.asarray(embeddings)
    print(f"✅ Embeddings shape: {embeddings.shape}")

    # Train
    trainer = ModelTrainer(model)
    topics, probs = trainer.train(text_vec, embeddings)

    # Controlled outlier reassignment
    topics = trainer.reduce_outliers(text_vec, embeddings)

    # No forced topic merging: keep the topic count as discovered by HDBSCAN.

    # Metrics on FINAL topics
    metrics_calc = MetricsCalculator(model, text_vec, topics, embeddings)
    metrics = metrics_calc.calculate_all_metrics()
    # Save run-level quality metrics CSV
    QualityExporter.export(metrics, topics, validation_report)

    # Export
    exporter = DataExporter(model, df, valid_indices, topics, probs)
    exporter.export_all()

    # Visuals
    df_aligned = df_filtered.copy()
    df_aligned["Topic"] = topics
    viz = VisualizationGenerator(model, text_vec, df_aligned)
    viz.generate_all()

    _save_final_report(validation_report, metrics, topics)

    print("\n" + "=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {Config.SEED}")
    print("\n📊 Results ready for publication:")
    print(f"   • CSV files: {Config.RESULTS_DIR}")
    print(f"   • Visualizations: {Config.FIGURES_DIR}")
    print("=" * 80 + "\n")


def _save_final_report(validation_report: Dict, metrics: Dict, topics: List[int]):
    report_path = Config.RESULTS_DIR / "00_Analysis_Report.txt"

    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("CIRP BIBLIOMETRIC ANALYSIS - FINAL REPORT\n")
        f.write("=" * 80 + "\n\n")

        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("Pipeline Version: v6.3 (natural)\n")
        f.write(f"Random Seed: {Config.SEED}\n\n")

        f.write("=" * 80 + "\nDATA SUMMARY\n" + "=" * 80 + "\n")
        f.write(f"Total documents analyzed: {validation_report.get('total_documents', 0):,}\n")
        f.write(f"Valid documents (after preprocessing): {len(topics):,}\n")
        if "temporal_distribution" in validation_report:
            td = validation_report["temporal_distribution"]
            f.write(f"Time coverage: {td['min']} - {td['max']}\n\n")

        f.write("=" * 80 + "\nTOPIC MODELING RESULTS\n" + "=" * 80 + "\n")
        f.write(f"Topics discovered (final): {n_topics}\n")
        f.write(f"Outlier documents (final): {n_outliers} ({(n_outliers/len(topics)*100):.1f}%)\n\n")

        f.write("=" * 80 + "\nQUALITY METRICS\n" + "=" * 80 + "\n")
        f.write(f"Topic Diversity: {metrics.get('diversity', 0.0):.4f}\n")
        f.write(f"Coherence C_v: {metrics.get('coherence_cv', 0.0):.4f}\n")
        f.write(f"Topic Stability: {metrics.get('stability', 0.0):.4f}\n\n")

        f.write("=" * 80 + "\nMODEL CONFIGURATION\n" + "=" * 80 + "\n")
        f.write(f"Embedding Model: {Config.EMBEDDING_MODEL}\n")
        f.write(f"UMAP: n_neighbors={Config.UMAP_N_NEIGHBORS}, n_components={Config.UMAP_N_COMPONENTS}, min_dist={Config.UMAP_MIN_DIST}\n")
        f.write(f"HDBSCAN: min_cluster_size={Config.HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={Config.HDBSCAN_MIN_SAMPLES}\n")
        f.write(f"Vectorizer: ngram_range={Config.VECTORIZER_NGRAM_RANGE}, min_df={Config.VECTORIZER_MIN_DF}, max_features={Config.VECTORIZER_MAX_FEATURES}\n")
        f.write(f"Outlier reassignment: {Config.ENABLE_OUTLIER_REASSIGN} (strategy={Config.OUTLIER_REASSIGN_STRATEGY}, threshold={Config.OUTLIER_REASSIGN_THRESHOLD})\n\n")

        f.write("=" * 80 + "\nOUTPUT FILES\n" + "=" * 80 + "\n")
        f.write("Results:\n")
        f.write("  • 00_Analysis_Report.txt\n")
        f.write("  • 01_Full_Document_Assignments.csv\n")
        f.write("  • 02_Top_Papers_Per_Topic.csv\n")
        f.write("  • 03_Topic_Metadata_Keywords.csv\n")
        f.write("  • 04_Summary_Statistics.csv\n")
        f.write("  • 05_Quality_Metrics.csv\n")
        f.write("  • 06_Topics_Over_Time.csv\n\n")
        f.write("Visualizations:\n")
        f.write("  • Viz_01_Intertopic_Distance_Map.html\n")
        f.write("  • Viz_02_Topic_Word_Scores.html\n")
        f.write("  • Viz_03_Hierarchical_Clustering.html\n")
        f.write("  • Viz_04_Topic_Similarity_Heatmap.html\n")
        f.write(f"  • Viz_05_Topics_Over_Time_to_{Config.TIME_CUTOFF_YEAR}.html\n")

    print(f"\n📄 Final report saved: {report_path}")


if __name__ == "__main__":
    main()
