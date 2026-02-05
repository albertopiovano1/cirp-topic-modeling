"""
================================================================================
CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v5.1 OPTIMIZED
================================================================================
Advanced Topic Modeling for Manufacturing Science & Technology Research
Target: CIRP Journal of Manufacturing Science and Technology (JMST)

VERSION 5.1 IMPROVEMENTS:
- Optimized parameters for fewer, more interpretable topics (target: 15-25)
- Fixed stability calculation bug
- Research papers only (excluded reviews)
- No forced outlier assignment (manual merge option)
- Extended time analysis to 2025
- Quality metrics CSV export
- Improved topic coherence and diversity

Author: Alberto Piovano
Institution: Politecnico di Torino - DIMEAS
Date: February 2026
================================================================================
"""

import pandas as pd
import numpy as np
import torch
import spacy
import re
import os
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

# BERTopic & Transformers
from bertopic import BERTopic
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer

# Metrics & Utils
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import resample
from gensim.corpora.dictionary import Dictionary
from gensim.models.coherencemodel import CoherenceModel

warnings.filterwarnings('ignore')

# ============================================================================
# 1. CONFIGURATION & SETUP
# ============================================================================

class Config:
    """Global configuration for the pipeline - OPTIMIZED FOR INTERPRETABILITY"""
    
    # Paths
    DATA_DIR = Path("data")
    RESULTS_DIR = Path("results")
    FIGURES_DIR = Path("figures")
    
    # Input file
    INPUT_FILE = "CIRP_researchonly.csv"
    
    # Reproducibility
    SEED = 76
    
    # Data filtering
    MIN_WORDS_PER_DOC = 10
    MIN_ABSTRACT_LENGTH = 50
    EXCLUDE_REVIEWS = True  # NEW: Only research papers
    
    # BERTopic parameters - OPTIMIZED
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    
    # UMAP - More global structure for fewer topics
    UMAP_N_NEIGHBORS = 25  # Increased from 15 → more global structure
    UMAP_N_COMPONENTS = 5
    UMAP_MIN_DIST = 0.0
    UMAP_METRIC = 'cosine'
    
    # HDBSCAN - Larger clusters for interpretability
    HDBSCAN_MIN_CLUSTER_SIZE = 50  # Increased from 10 → fewer, larger topics
    HDBSCAN_MIN_SAMPLES = 5  # Increased from 2 → more conservative
    HDBSCAN_METRIC = 'euclidean'
    HDBSCAN_CLUSTER_SELECTION = 'eom'
    
    # Vectorizer - Better keyword extraction
    VECTORIZER_NGRAM_RANGE = (1, 3)
    VECTORIZER_MIN_DF = 10  # Increased from 5 → filter rare terms
    VECTORIZER_MAX_DF = 0.5  # NEW: Remove terms in >50% documents
    VECTORIZER_MAX_FEATURES = 10000
    
    # Representation
    TOP_N_WORDS = 10
    
    # Outlier handling - NEW
    REDUCE_OUTLIERS = False  # Keep outliers separate for manual review
    
    # Analysis parameters
    TIME_CUTOFF_YEAR = 2025  # CHANGED: Include 2025 data
    TOP_N_TOPICS_VISUALIZATION = 20
    TOP_N_PAPERS_PER_TOPIC = 10  # Increased from 5 for better validation
    
    # Stability bootstrap
    STABILITY_N_SAMPLES = 1.0
    STABILITY_SEED_OFFSET = 1


def set_seed(seed: int = Config.SEED):
    """Set random seed for reproducibility"""
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
    """Create output directories if they don't exist"""
    for directory in [Config.DATA_DIR, Config.RESULTS_DIR, Config.FIGURES_DIR]:
        directory.mkdir(exist_ok=True)
    print(f"📁 Output directories ready: {Config.RESULTS_DIR}, {Config.FIGURES_DIR}")


# ============================================================================
# 2. DATA VALIDATION & QUALITY REPORTING
# ============================================================================

class DataValidator:
    """Comprehensive data validation and quality reporting"""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.report = {}
        
    def validate_and_report(self) -> Dict:
        """Run all validation checks and generate report"""
        print("\n" + "="*80)
        print("DATA VALIDATION & QUALITY REPORT")
        print("="*80)
        
        self._check_basic_structure()
        self._check_coverage()
        self._check_document_types()
        self._check_duplicates()
        self._check_temporal_distribution()
        self._check_content_quality()
        
        self._print_summary()
        return self.report
    
    def _check_basic_structure(self):
        """Check basic DataFrame structure"""
        required_cols = ['Title', 'Abstract', 'Year', 'Document Type']
        missing_cols = [col for col in required_cols if col not in self.df.columns]
        
        self.report['total_documents'] = len(self.df)
        self.report['total_columns'] = len(self.df.columns)
        self.report['missing_required_columns'] = missing_cols
        
        print(f"\n📊 Basic Structure:")
        print(f"  • Total documents: {len(self.df):,}")
        print(f"  • Total columns: {len(self.df.columns)}")
        if missing_cols:
            print(f"  ⚠️  Missing required columns: {missing_cols}")
    
    def _check_coverage(self):
        """Check data coverage and missingness"""
        key_fields = ['Title', 'Abstract', 'Author Keywords', 'Index Keywords', 'DOI']
        coverage = {}
        
        print(f"\n📈 Data Coverage:")
        for field in key_fields:
            if field in self.df.columns:
                non_null = self.df[field].notna().sum()
                pct = (non_null / len(self.df)) * 100
                coverage[field] = {'count': int(non_null), 'percentage': round(pct, 2)}
                print(f"  • {field}: {non_null:,} ({pct:.1f}%)")
        
        self.report['coverage'] = coverage
    
    def _check_document_types(self):
        """Analyze document type distribution"""
        if 'Document Type' in self.df.columns:
            doc_types = self.df['Document Type'].value_counts().to_dict()
            self.report['document_types'] = doc_types
            
            print(f"\n📑 Document Types:")
            for doc_type, count in doc_types.items():
                pct = (count / len(self.df)) * 100
                print(f"  • {doc_type}: {count:,} ({pct:.1f}%)")
                
            # Check for reviews
            if Config.EXCLUDE_REVIEWS:
                reviews = self.df[self.df['Document Type'] == 'Review']
                if len(reviews) > 0:
                    print(f"\n  ⚠️  {len(reviews)} reviews will be excluded (EXCLUDE_REVIEWS=True)")
    
    def _check_duplicates(self):
        """Check for duplicate entries"""
        if 'DOI' in self.df.columns:
            duplicates_doi = self.df['DOI'].duplicated().sum()
        else:
            duplicates_doi = 0
        
        duplicates_title = self.df['Title'].duplicated().sum()
        
        self.report['duplicates'] = {
            'by_doi': int(duplicates_doi),
            'by_title': int(duplicates_title)
        }
        
        print(f"\n🔍 Duplicate Detection:")
        print(f"  • Duplicate DOIs: {duplicates_doi}")
        print(f"  • Duplicate Titles: {duplicates_title}")
        
        if duplicates_doi > 0 or duplicates_title > 0:
            print(f"  ⚠️  Consider deduplication before analysis")
    
    def _check_temporal_distribution(self):
        """Analyze temporal distribution"""
        if 'Year' in self.df.columns:
            year_stats = {
                'min': int(self.df['Year'].min()),
                'max': int(self.df['Year'].max()),
                'mean': round(float(self.df['Year'].mean()), 1),
                'median': int(self.df['Year'].median())
            }
            
            self.report['temporal_distribution'] = year_stats
            
            print(f"\n📅 Temporal Coverage:")
            print(f"  • Period: {year_stats['min']} - {year_stats['max']}")
            print(f"  • Mean year: {year_stats['mean']}")
            print(f"  • Median year: {year_stats['median']}")
            
            # Distribution by decade
            self.df['Decade'] = (self.df['Year'] // 10) * 10
            decade_dist = self.df['Decade'].value_counts().sort_index()
            print(f"\n  Distribution by decade:")
            for decade, count in decade_dist.items():
                print(f"    {decade}s: {count:,} papers")
    
    def _check_content_quality(self):
        """Analyze content quality metrics"""
        self.df['Abstract_Length'] = self.df['Abstract'].fillna("").str.split().str.len()
        
        length_stats = {
            'mean': round(float(self.df['Abstract_Length'].mean()), 1),
            'median': int(self.df['Abstract_Length'].median()),
            'min': int(self.df['Abstract_Length'].min()),
            'max': int(self.df['Abstract_Length'].max())
        }
        
        short_abstracts = (self.df['Abstract_Length'] < Config.MIN_ABSTRACT_LENGTH).sum()
        
        self.report['content_quality'] = {
            'abstract_length': length_stats,
            'short_abstracts': int(short_abstracts)
        }
        
        print(f"\n📝 Content Quality:")
        print(f"  • Abstract length (words):")
        print(f"    - Mean: {length_stats['mean']}")
        print(f"    - Median: {length_stats['median']}")
        print(f"    - Range: {length_stats['min']} - {length_stats['max']}")
        print(f"  • Short abstracts (<{Config.MIN_ABSTRACT_LENGTH} words): {short_abstracts}")
        
        if short_abstracts > 0:
            pct = (short_abstracts / len(self.df)) * 100
            print(f"    ({pct:.1f}% of corpus - consider exclusion)")
    
    def _print_summary(self):
        """Print validation summary"""
        print("\n" + "="*80)
        print("VALIDATION SUMMARY")
        print("="*80)
        
        issues = 0
        if self.report.get('missing_required_columns'):
            issues += 1
        if self.report['duplicates']['by_doi'] > 10:
            issues += 1
        if self.report['content_quality']['short_abstracts'] > len(self.df) * 0.05:
            issues += 1
        
        if issues == 0:
            print("✅ Dataset quality: EXCELLENT - Ready for analysis")
        elif issues == 1:
            print("⚠️  Dataset quality: GOOD - Minor issues detected")
        else:
            print("❌ Dataset quality: FAIR - Review recommended before proceeding")
        
        print("="*80 + "\n")


# ============================================================================
# 3. TEXT PREPROCESSING WITH DUAL OUTPUT
# ============================================================================

class TextPreprocessor:
    """Advanced text preprocessing with dual output strategy"""
    
    def __init__(self, synonym_mode: str = "safe"):
        """
        Args:
            synonym_mode: 'safe', 'aggressive', or 'off'
        """
        self.synonym_mode = synonym_mode
        self._load_spacy()
        self._setup_synonym_maps()
        self._setup_stopwords()
        
    def _load_spacy(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            print("✅ spaCy model loaded: en_core_web_sm")
        except:
            print("❌ spaCy model not found. Installing...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    
    def _setup_synonym_maps(self):
        """Setup synonym normalization maps"""
        
        # SAFE SYNONYMS: Unambiguous acronyms and standard terms
        self.safe_synonyms = {
            # Data & Computing
            "datum": "data",
            "ml": "machine_learning",
            "ai": "artificial_intelligence",
            "dl": "deep_learning",
            "nn": "neural_network",
            
            # IoT & Industry 4.0
            "iot": "internet_of_things",
            "iiot": "industrial_internet_of_things",
            "cps": "cyber_physical_system",
            
            # Blockchain
            "bct": "blockchain",
            "blockchain_technology": "blockchain",
            
            # Manufacturing Technologies
            "am": "additive_manufacturing",
            "additive": "additive_manufacturing",
            "3d_print": "additive_manufacturing",
            "3d_printing": "additive_manufacturing",
            "print": "additive_manufacturing",
            "printing": "additive_manufacturing",
            
            # Digital Technologies
            "dt": "digital_twin",
            "dts": "digital_twin",
            "digital_twins": "digital_twin",
            
            # Supply Chain
            "fsc": "food_supply_chain",
            "sfsc": "short_food_supply_chain",
            "sc": "supply_chain",
            
            # Circular Economy
            "ce": "circular_economy",
            
            # Sensing & Identification
            "hsi": "hyperspectral_imaging",
            "nir": "near_infrared_spectroscopy",
            "rfid": "radio_frequency_identification",
            
            # Organizations
            "sme": "small_medium_enterprise",
            "smes": "small_medium_enterprise"
        }
        
        # AGGRESSIVE SYNONYMS: More ambiguous - use with caution
        self.aggressive_synonyms = {
            "optimization": "optimize",
            "optimisation": "optimize",
            "modelling": "modeling",
            "analyse": "analyze",
            "analysis": "analyze",
        }
        
        # Select active map based on mode
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
        """Setup comprehensive stopword list"""
        
        # Academic paper noise - EXPANDED for better filtering
        stopwords_academic = [
            "aim", "paper", "study", "result", "conclusion", "method", "methodology",
            "analysis", "data", "review", "literature", "author", "university", 
            "department", "example", "generic", "significant", "significantly",
            "respectively", "increase", "decrease", "show", "demonstrate",
            "include", "exclude", "obtain", "use", "using", "utilize", 
            "proposed", "propose", "based", "approach", "application", "applied",
            "conducted", "investigated", "discussed", "find", "finding",
            "evidence", "suggest", "present", "presentation", "introduction",
            "future", "work", "limitation", "practical", "implication",
            "theoretical", "context", "case", "research", "journal", "article",
            "systematic", "survey", "et", "al", "i.e", "e.g", "etc",
            "model", "method", "develop", "development", "propose", "proposed",
            "novel", "new", "different", "various", "several", "main", "major"
        ]
        
        # Publisher metadata noise
        stopwords_publisher = [
            "emerald publishing limited", "taylor francis group", 
            "licensee mdpi basel", "mdpi basel switzerland", "springer nature",
            "john wiley sons", "rights reserved", "authors exclusive license",
            "exclusive license", "trading taylor", "informa uk limited",
            "copyright", "reserved", "doi", "www", "http", "https", "com", "org",
            "elsevier", "wiley", "sage"
        ]
        
        # Generic domain terms - EXPANDED
        stopwords_domain = [
            "system", "process", "product", "solution",
            "company", "business", "industry", "sector",
            "technology", "technique", "tool", "equipment"
        ]
        
        # Combine all stopwords
        custom_stopwords = set([w.lower().strip() for w in 
                               stopwords_academic + stopwords_publisher + stopwords_domain])
        
        # Add to spaCy's default stopwords
        self.nlp.Defaults.stop_words |= custom_stopwords
        self.stopwords_list = list(self.nlp.Defaults.stop_words)
        
        print(f"🛑 Custom stopwords loaded: {len(custom_stopwords)} terms")
    
    def clean_text_for_embedding(self, text: str) -> str:
        """Conservative cleaning for embedding model"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = text.encode("ascii", "ignore").decode()
        text = re.sub(r'http\S+|www\S+', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.lower().strip()
    
    def clean_text_for_vectorizer(self, text: str) -> str:
        """Aggressive cleaning for c-TF-IDF vectorizer"""
        if not isinstance(text, str) or not text.strip():
            return ""
        
        text = self.clean_text_for_embedding(text)
        text = re.sub(r'[^\w\s-]', ' ', text)
        text = re.sub(r'\b\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def lemmatize_and_normalize(self, text: str, for_vectorizer: bool = False) -> str:
        """Lemmatization with optional synonym normalization"""
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
    
    def process_corpus(self, 
                      titles: List[str], 
                      abstracts: List[str]) -> Tuple[List[str], List[str], List[int]]:
        """Process entire corpus with dual output"""
        
        print("\n" + "="*80)
        print("TEXT PREPROCESSING")
        print("="*80)
        
        combined_texts = []
        for title, abstract in zip(titles, abstracts):
            title_str = str(title) if pd.notna(title) else ""
            abstract_str = str(abstract) if pd.notna(abstract) else ""
            combined_texts.append(f"{title_str}. {abstract_str}")
        
        print(f"📄 Processing {len(combined_texts):,} documents...")
        
        # Step 1: Basic cleaning
        print("  [1/4] Basic cleaning...")
        cleaned_embed = [self.clean_text_for_embedding(t) for t in tqdm(combined_texts, desc="Cleaning")]
        cleaned_vec = [self.clean_text_for_vectorizer(t) for t in tqdm(combined_texts, desc="Deep cleaning")]
        
        # Step 2: Filter valid documents
        print(f"  [2/4] Filtering documents (min {Config.MIN_WORDS_PER_DOC} words)...")
        valid_indices = []
        for i, text in enumerate(cleaned_embed):
            if len(text.split()) >= Config.MIN_WORDS_PER_DOC:
                valid_indices.append(i)
        
        print(f"    ✅ Valid documents: {len(valid_indices):,} / {len(combined_texts):,}")
        print(f"    ❌ Filtered out: {len(combined_texts) - len(valid_indices):,}")
        
        # Step 3: Lemmatize valid documents
        print("  [3/4] Lemmatization & normalization...")
        text_embed = []
        text_vec = []
        
        for idx in tqdm(valid_indices, desc="Lemmatizing"):
            lemma_embed = self.lemmatize_and_normalize(cleaned_embed[idx], for_vectorizer=False)
            text_embed.append(lemma_embed)
            
            lemma_vec = self.lemmatize_and_normalize(cleaned_vec[idx], for_vectorizer=True)
            text_vec.append(lemma_vec)
        
        # Step 4: Final validation
        print("  [4/4] Final validation...")
        final_valid_indices = []
        final_text_embed = []
        final_text_vec = []
        
        for i, (te, tv, idx) in enumerate(zip(text_embed, text_vec, valid_indices)):
            if len(te.split()) >= 5 and len(tv.split()) >= 5:
                final_text_embed.append(te)
                final_text_vec.append(tv)
                final_valid_indices.append(idx)
        
        print(f"\n✅ Preprocessing complete:")
        print(f"   • Final corpus size: {len(final_text_embed):,} documents")
        print(f"   • Dual outputs ready: text_embed + text_vec")
        print("="*80 + "\n")
        
        return final_text_embed, final_text_vec, final_valid_indices


# ============================================================================
# 4. BERTOPIC MODEL CONFIGURATION
# ============================================================================

class BERTopicModelBuilder:
    """Build and configure BERTopic model with optimized parameters"""
    
    def __init__(self, stopwords_list: List[str]):
        self.stopwords_list = stopwords_list
        self.model = None
        
    def build_model(self) -> BERTopic:
        """Build BERTopic model with optimized configuration"""
        
        print("\n" + "="*80)
        print("BERTOPIC MODEL CONFIGURATION - OPTIMIZED v5.1")
        print("="*80)
        
        # 1. Embedding Model
        print(f"\n[1/6] Loading embedding model: {Config.EMBEDDING_MODEL}")
        embedding_model = SentenceTransformer(Config.EMBEDDING_MODEL)
        print(f"    ✅ Model loaded: {embedding_model.get_sentence_embedding_dimension()}D embeddings")
        
        # 2. UMAP - OPTIMIZED for fewer topics
        print(f"\n[2/6] Configuring UMAP (OPTIMIZED):")
        print(f"    • n_neighbors: {Config.UMAP_N_NEIGHBORS} (↑ from 15 → more global structure)")
        print(f"    • n_components: {Config.UMAP_N_COMPONENTS}")
        print(f"    • min_dist: {Config.UMAP_MIN_DIST}")
        umap_model = UMAP(
            n_neighbors=Config.UMAP_N_NEIGHBORS,
            n_components=Config.UMAP_N_COMPONENTS,
            min_dist=Config.UMAP_MIN_DIST,
            metric=Config.UMAP_METRIC,
            random_state=Config.SEED
        )
        
        # 3. HDBSCAN - OPTIMIZED for larger clusters
        print(f"\n[3/6] Configuring HDBSCAN (OPTIMIZED):")
        print(f"    • min_cluster_size: {Config.HDBSCAN_MIN_CLUSTER_SIZE} (↑ from 10 → fewer topics)")
        print(f"    • min_samples: {Config.HDBSCAN_MIN_SAMPLES} (↑ from 2 → more conservative)")
        print(f"    • Keep outliers separate: {not Config.REDUCE_OUTLIERS}")
        hdbscan_model = HDBSCAN(
            min_cluster_size=Config.HDBSCAN_MIN_CLUSTER_SIZE,
            min_samples=Config.HDBSCAN_MIN_SAMPLES,
            metric=Config.HDBSCAN_METRIC,
            cluster_selection_method=Config.HDBSCAN_CLUSTER_SELECTION,
            prediction_data=True
        )
        
        # 4. CountVectorizer - OPTIMIZED
        print(f"\n[4/6] Configuring CountVectorizer (OPTIMIZED):")
        print(f"    • ngram_range: {Config.VECTORIZER_NGRAM_RANGE}")
        print(f"    • min_df: {Config.VECTORIZER_MIN_DF} (↑ from 5 → filter rare terms)")
        print(f"    • max_df: {Config.VECTORIZER_MAX_DF} (NEW → filter common terms)")
        print(f"    • max_features: {Config.VECTORIZER_MAX_FEATURES:,}")
        vectorizer_model = CountVectorizer(
            stop_words=self.stopwords_list,
            ngram_range=Config.VECTORIZER_NGRAM_RANGE,
            min_df=Config.VECTORIZER_MIN_DF,
            max_df=Config.VECTORIZER_MAX_DF,
            max_features=Config.VECTORIZER_MAX_FEATURES
        )
        
        # 5. c-TF-IDF Transformer
        print(f"\n[5/6] Configuring c-TF-IDF:")
        print(f"    • BM25 weighting: Enabled")
        print(f"    • Reduce frequent words: Enabled")
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=True,
            reduce_frequent_words=True
        )
        
        # 6. Representation Model
        print(f"\n[6/6] Configuring representation:")
        print(f"    • Model: KeyBERTInspired")
        print(f"    • Top N words: {Config.TOP_N_WORDS}")
        representation_model = KeyBERTInspired()
        
        # Build final model
        print(f"\n🏗️  Building BERTopic model...")
        self.model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            top_n_words=Config.TOP_N_WORDS,
            verbose=True,
            calculate_probabilities=True,
            language="english"
        )
        
        print("✅ Model configuration complete (OPTIMIZED FOR INTERPRETABILITY)")
        print("="*80 + "\n")
        
        return self.model


# ============================================================================
# 5. MODEL TRAINING & OUTLIER HANDLING
# ============================================================================

class ModelTrainer:
    """Handle model training with optional outlier reduction"""
    
    def __init__(self, model: BERTopic):
        self.model = model
        self.topics = None
        self.probs = None
        
    def train(self, texts: List[str], embeddings: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Train BERTopic model"""
        
        print("\n" + "="*80)
        print("MODEL TRAINING")
        print("="*80)
        
        print(f"📊 Training on {len(texts):,} documents...")
        print(f"   Embedding shape: {embeddings.shape}")
        
        self.topics, self.probs = self.model.fit_transform(texts, embeddings=embeddings)
        
        # Report initial results
        n_topics = len(set(self.topics)) - (1 if -1 in self.topics else 0)
        n_outliers = sum(1 for t in self.topics if t == -1)
        outlier_pct = (n_outliers / len(self.topics)) * 100
        
        print(f"\n📈 Initial Results:")
        print(f"   • Topics discovered: {n_topics}")
        print(f"   • Outliers: {n_outliers:,} ({outlier_pct:.1f}%)")
        
        if not Config.REDUCE_OUTLIERS and n_outliers > 0:
            print(f"\n   ℹ️  Outliers kept separate (REDUCE_OUTLIERS=False)")
            print(f"      → Review outliers manually in results CSV")
            print(f"      → Consider manual topic merging if needed")
        
        return self.topics, self.probs
    
    def reduce_outliers(self, texts: List[str], embeddings: np.ndarray):
        """Optionally reduce outliers using embedding strategy"""
        
        if not Config.REDUCE_OUTLIERS:
            print("\n⏭️  Skipping outlier reduction (REDUCE_OUTLIERS=False)")
            return self.topics
        
        if -1 not in self.topics:
            print("✅ No outliers to reduce")
            return self.topics
        
        print("\n🔧 Reducing outliers (embeddings strategy)...")
        try:
            new_topics = self.model.reduce_outliers(
                texts, 
                self.topics, 
                strategy="embeddings",
                embeddings=embeddings
            )
            
            self.model.update_topics(texts, new_topics)
            
            n_outliers_before = sum(1 for t in self.topics if t == -1)
            n_outliers_after = sum(1 for t in new_topics if t == -1)
            reduction = n_outliers_before - n_outliers_after
            
            print(f"   ✅ Outliers reduced: {n_outliers_before:,} → {n_outliers_after:,} (-{reduction:,})")
            
            self.topics = new_topics
            return new_topics
            
        except Exception as e:
            print(f"   ⚠️  Error during outlier reduction: {e}")
            return self.topics


# ============================================================================
# 6. ADVANCED METRICS CALCULATION - FIXED STABILITY BUG
# ============================================================================

class MetricsCalculator:
    """Calculate publication-ready quality metrics - WITH STABILITY FIX"""
    
    def __init__(self, model: BERTopic, texts: List[str], embeddings: np.ndarray):
        self.model = model
        self.texts = texts
        self.embeddings = embeddings  # Store embeddings directly
        self.metrics = {}
        
    def calculate_all_metrics(self) -> Dict:
        """Calculate diversity, coherence, and stability metrics"""
        
        print("\n" + "="*80)
        print("QUALITY METRICS CALCULATION")
        print("="*80)
        
        # Prepare data
        tokenized_docs = [doc.split() for doc in self.texts]
        dictionary = Dictionary(tokenized_docs)
        
        topic_info = self.model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        top_words_per_topic = [[w for w, _ in self.model.get_topic(tid)] 
                              for tid in valid_topics]
        
        # 1. Topic Diversity
        print("\n[1/3] Calculating Topic Diversity...")
        diversity = self._calculate_diversity(top_words_per_topic)
        self.metrics['diversity'] = diversity
        print(f"    🟣 Topic Diversity: {diversity:.4f}")
        
        # 2. Coherence C_v
        print("\n[2/3] Calculating Coherence C_v...")
        coherence = self._calculate_coherence(top_words_per_topic, tokenized_docs, dictionary)
        self.metrics['coherence_cv'] = coherence
        print(f"    🟢 Coherence C_v: {coherence:.4f}")
        
        # 3. Stability (Bootstrap) - FIXED
        print("\n[3/3] Calculating Stability (Bootstrap) - FIXED...")
        stability = self._calculate_stability()
        self.metrics['stability'] = stability
        print(f"    🟠 Topic Stability: {stability:.4f}")
        
        print("\n" + "="*80)
        print("METRICS SUMMARY")
        print("="*80)
        print(f"  Diversity:  {diversity:.4f} (higher = more unique words)")
        print(f"  Coherence:  {coherence:.4f} (higher = more semantically coherent)")
        print(f"  Stability:  {stability:.4f} (higher = more reproducible)")
        print("="*80 + "\n")
        
        return self.metrics
    
    def _calculate_diversity(self, top_words_per_topic: List[List[str]]) -> float:
        """Calculate topic diversity"""
        all_words = [w for topic in top_words_per_topic for w in topic]
        unique_words = set(all_words)
        total_slots = len(top_words_per_topic) * Config.TOP_N_WORDS
        diversity = len(unique_words) / total_slots if total_slots > 0 else 0
        return diversity
    
    def _calculate_coherence(self, 
                            top_words: List[List[str]], 
                            tokenized_docs: List[List[str]], 
                            dictionary: Dictionary) -> float:
        """Calculate coherence C_v score"""
        try:
            cm = CoherenceModel(
                topics=top_words,
                texts=tokenized_docs,
                dictionary=dictionary,
                coherence='c_v'
            )
            return cm.get_coherence()
        except Exception as e:
            print(f"    ⚠️  Error calculating coherence: {e}")
            return 0.0
    
    def _calculate_stability(self) -> float:
        """Calculate topic stability via bootstrap - FIXED BUG"""
        try:
            # Bootstrap sample
            n_samples = int(len(self.texts) * Config.STABILITY_N_SAMPLES)
            boot_indices = resample(
                range(len(self.texts)),
                n_samples=n_samples,
                replace=True,
                random_state=Config.SEED + Config.STABILITY_SEED_OFFSET
            )
            
            boot_docs = [self.texts[i] for i in boot_indices]
            boot_embeddings = self.embeddings[boot_indices]  # FIX: Use stored embeddings
            
            # Train bootstrap model
            print("    Training bootstrap model...")
            model_boot = BERTopic(
                embedding_model=self.model.embedding_model,
                umap_model=self.model.umap_model,
                hdbscan_model=self.model.hdbscan_model,
                vectorizer_model=self.model.vectorizer_model,
                ctfidf_model=self.model.ctfidf_model,
                representation_model=self.model.representation_model,
                verbose=False
            )
            
            model_boot.fit(boot_docs, embeddings=boot_embeddings)
            
            # Extract centroids - FIX: Use embedding model correctly
            orig_centroids = self._get_topic_centroids(self.model)
            boot_centroids = self._get_topic_centroids(model_boot)
            
            # Calculate similarities
            similarities = []
            if orig_centroids and boot_centroids:
                for tid, orig_vec in orig_centroids.items():
                    best_sim = 0
                    for boot_vec in boot_centroids.values():
                        sim = cosine_similarity(
                            orig_vec.reshape(1, -1), 
                            boot_vec.reshape(1, -1)
                        )[0][0]
                        if sim > best_sim:
                            best_sim = sim
                    similarities.append(best_sim)
                
                return np.mean(similarities)
            else:
                return 0.0
                
        except Exception as e:
            print(f"    ⚠️  Error calculating stability: {e}")
            import traceback
            traceback.print_exc()
            return 0.0
    
    def _get_topic_centroids(self, model: BERTopic) -> Dict[int, np.ndarray]:
        """Extract topic centroids as embedding vectors - FIXED"""
        centroids = {}
        
        # Get embedding model (handle both SentenceTransformer and backend wrapper)
        if hasattr(model.embedding_model, 'encode'):
            embed_fn = model.embedding_model.encode
        elif hasattr(model.embedding_model, 'embed'):
            embed_fn = model.embedding_model.embed
        else:
            print("    ⚠️  Cannot access embedding function")
            return {}
        
        for tid in model.get_topics().keys():
            if tid == -1:
                continue
            words = [w for w, _ in model.get_topic(tid)]
            if words:
                centroid_text = " ".join(words)
                try:
                    centroid_vec = embed_fn(centroid_text, show_progress_bar=False)
                    # Handle batch output
                    if isinstance(centroid_vec, np.ndarray) and len(centroid_vec.shape) > 1:
                        centroid_vec = centroid_vec[0]
                    centroids[tid] = centroid_vec
                except Exception as e:
                    print(f"    ⚠️  Error encoding topic {tid}: {e}")
                    continue
        
        return centroids


# [CONTINUED IN NEXT MESSAGE DUE TO LENGTH]

# ============================================================================
# 7. DATA EXPORT FOR QUALITATIVE ANALYSIS
# ============================================================================

class DataExporter:
    """Export results in publication-ready CSV formats"""
    
    def __init__(self, 
                 model: BERTopic, 
                 df_original: pd.DataFrame, 
                 valid_indices: List[int],
                 topics: List[int],
                 probs: Optional[np.ndarray],
                 metrics: Dict):
        self.model = model
        self.df_original = df_original
        self.valid_indices = valid_indices
        self.topics = topics
        self.probs = probs
        self.metrics = metrics
        
    def export_all(self):
        """Export all analysis files including quality metrics CSV"""
        
        print("\n" + "="*80)
        print("DATA EXPORT")
        print("="*80)
        
        # Create filtered dataframe aligned with topics
        df_aligned = self.df_original.iloc[self.valid_indices].reset_index(drop=True)
        df_aligned['Topic'] = self.topics
        
        # Add topic names
        df_aligned['Topic_Name'] = df_aligned['Topic'].apply(
            lambda x: self.model.get_topic_info(x)["Name"].values[0] if x != -1 else "Outlier"
        )
        
        # Add probabilities
        df_aligned['Probability'] = self._extract_probabilities()
        
        # 1. Full document assignments
        self._export_full_assignments(df_aligned)
        
        # 2. Top papers per topic
        self._export_top_papers(df_aligned)
        
        # 3. Topic metadata
        self._export_topic_metadata()
        
        # 4. Summary statistics
        self._export_summary_stats(df_aligned)
        
        # 5. Quality metrics CSV - NEW!
        self._export_quality_metrics()
        
        # 6. Outliers analysis - NEW!
        if -1 in self.topics:
            self._export_outliers_analysis(df_aligned)
        
        print("\n✅ All exports complete")
        print("="*80 + "\n")
    
    def _extract_probabilities(self) -> List[float]:
        """Extract topic probabilities robustly"""
        if self.probs is None:
            return [0.0] * len(self.topics)
        
        probs_list = []
        for i, topic in enumerate(self.topics):
            if topic == -1:
                probs_list.append(0.0)
            else:
                if len(self.probs.shape) == 1:
                    probs_list.append(self.probs[i])
                else:
                    if topic < self.probs.shape[1]:
                        probs_list.append(self.probs[i][topic])
                    else:
                        probs_list.append(0.0)
        
        return probs_list
    
    def _export_full_assignments(self, df: pd.DataFrame):
        """Export complete document-topic assignments"""
        output_path = Config.RESULTS_DIR / "01_Full_Document_Assignments.csv"
        
        export_cols = [
            'Title', 'Year', 'Authors', 'Abstract', 
            'Topic', 'Topic_Name', 'Probability',
            'DOI', 'Cited by', 'Author Keywords'
        ]
        
        export_cols = [col for col in export_cols if col in df.columns]
        
        df[export_cols].to_csv(output_path, index=False)
        print(f"\n[1/6] ✅ Saved: {output_path}")
        print(f"      {len(df):,} documents with topic assignments")
    
    def _export_top_papers(self, df: pd.DataFrame):
        """Export most relevant papers per topic"""
        output_path = Config.RESULTS_DIR / "02_Top_Papers_Per_Topic.csv"
        
        topic_info = self.model.get_topic_info()
        valid_topics = topic_info[topic_info["Topic"] != -1]["Topic"].tolist()
        
        top_docs_list = []
        for tid in valid_topics:
            topic_docs = df[df["Topic"] == tid].copy()
            
            topic_docs = topic_docs.sort_values(
                by="Probability", 
                ascending=False
            ).head(Config.TOP_N_PAPERS_PER_TOPIC)
            
            topic_docs['Topic_ID'] = tid
            topic_docs['Rank'] = range(1, len(topic_docs) + 1)
            
            top_docs_list.append(topic_docs)
        
        if top_docs_list:
            df_top = pd.concat(top_docs_list, ignore_index=True)
            
            review_cols = [
                'Topic_ID', 'Topic_Name', 'Rank',
                'Title', 'Year', 'Authors', 'Abstract',
                'Probability', 'Cited by', 'DOI'
            ]
            review_cols = [col for col in review_cols if col in df_top.columns]
            
            df_top[review_cols].to_csv(output_path, index=False)
            print(f"\n[2/6] ✅ Saved: {output_path}")
            print(f"      Top {Config.TOP_N_PAPERS_PER_TOPIC} papers × {len(valid_topics)} topics")
    
    def _export_topic_metadata(self):
        """Export topic keywords and metadata"""
        output_path = Config.RESULTS_DIR / "03_Topic_Metadata_Keywords.csv"
        
        topic_info = self.model.get_topic_info()
        topic_info.to_csv(output_path, index=False)
        
        print(f"\n[3/6] ✅ Saved: {output_path}")
        print(f"      Metadata for {len(topic_info)} topics")
    
    def _export_summary_stats(self, df: pd.DataFrame):
        """Export summary statistics"""
        output_path = Config.RESULTS_DIR / "04_Summary_Statistics.csv"
        
        stats_list = []
        for topic in df['Topic'].unique():
            if topic == -1:
                continue
            
            topic_docs = df[df['Topic'] == topic]
            
            stats = {
                'Topic_ID': topic,
                'Topic_Name': topic_docs['Topic_Name'].iloc[0],
                'N_Documents': len(topic_docs),
                'Percentage': (len(topic_docs) / len(df)) * 100,
                'Avg_Probability': topic_docs['Probability'].mean(),
                'Avg_Citations': topic_docs['Cited by'].mean() if 'Cited by' in topic_docs.columns else 0,
                'Year_Range': f"{topic_docs['Year'].min()}-{topic_docs['Year'].max()}",
                'Peak_Year': topic_docs['Year'].mode().values[0] if len(topic_docs) > 0 else None
            }
            
            stats_list.append(stats)
        
        df_stats = pd.DataFrame(stats_list)
        df_stats = df_stats.sort_values('N_Documents', ascending=False)
        df_stats.to_csv(output_path, index=False)
        
        print(f"\n[4/6] ✅ Saved: {output_path}")
        print(f"      Summary statistics for {len(df_stats)} topics")
    
    def _export_quality_metrics(self):
        """Export quality metrics CSV - NEW!"""
        output_path = Config.RESULTS_DIR / "05_Quality_Metrics.csv"
        
        metrics_df = pd.DataFrame([{
            'Metric': 'Topic Diversity',
            'Value': self.metrics.get('diversity', 0.0),
            'Interpretation': 'Proportion of unique words across topics',
            'Target': '> 0.80',
            'Status': '✅ Good' if self.metrics.get('diversity', 0) > 0.80 else '⚠️ Review'
        }, {
            'Metric': 'Coherence C_v',
            'Value': self.metrics.get('coherence_cv', 0.0),
            'Interpretation': 'Semantic coherence of topic keywords',
            'Target': '> 0.65',
            'Status': '✅ Good' if self.metrics.get('coherence_cv', 0) > 0.65 else '⚠️ Review'
        }, {
            'Metric': 'Topic Stability',
            'Value': self.metrics.get('stability', 0.0),
            'Interpretation': 'Consistency across bootstrap samples',
            'Target': '> 0.75',
            'Status': '✅ Good' if self.metrics.get('stability', 0) > 0.75 else '⚠️ Review'
        }])
        
        metrics_df.to_csv(output_path, index=False)
        
        print(f"\n[5/6] ✅ Saved: {output_path}")
        print(f"      Quality metrics with targets and status")
    
    def _export_outliers_analysis(self, df: pd.DataFrame):
        """Export outliers for manual review - NEW!"""
        output_path = Config.RESULTS_DIR / "06_Outliers_Analysis.csv"
        
        outliers = df[df['Topic'] == -1].copy()
        
        if len(outliers) > 0:
            # Sort by year to see temporal patterns
            outliers = outliers.sort_values('Year', ascending=False)
            
            export_cols = ['Title', 'Year', 'Authors', 'Abstract', 'DOI', 'Author Keywords']
            export_cols = [col for col in export_cols if col in outliers.columns]
            
            outliers[export_cols].to_csv(output_path, index=False)
            
            print(f"\n[6/6] ✅ Saved: {output_path}")
            print(f"      {len(outliers)} outliers for manual review")
            print(f"      → Consider: Are these noise or emerging topics?")
        else:
            print(f"\n[6/6] ℹ️  No outliers to export")


# ============================================================================
# 8. VISUALIZATION GENERATION
# ============================================================================

class VisualizationGenerator:
    """Generate BERTopic visualizations with extended time range"""
    
    def __init__(self, 
                 model: BERTopic, 
                 texts: List[str],
                 df_aligned: pd.DataFrame):
        self.model = model
        self.texts = texts
        self.df_aligned = df_aligned
        
    def generate_all(self):
        """Generate all visualizations"""
        
        print("\n" + "="*80)
        print("VISUALIZATION GENERATION")
        print("="*80)
        
        self._generate_intertopic_map()
        self._generate_barchart()
        self._generate_hierarchy()
        self._generate_heatmap()
        self._generate_topics_over_time()
        
        print("\n✅ All visualizations generated")
        print("="*80 + "\n")
    
    def _generate_intertopic_map(self):
        """Intertopic distance map"""
        try:
            output_path = Config.FIGURES_DIR / "Viz_01_Intertopic_Distance_Map.html"
            fig = self.model.visualize_topics()
            fig.write_html(str(output_path))
            print(f"\n[1/5] ✅ Saved: {output_path}")
        except Exception as e:
            print(f"\n[1/5] ⚠️  Error generating intertopic map: {e}")
    
    def _generate_barchart(self):
        """Topic word scores"""
        try:
            output_path = Config.FIGURES_DIR / "Viz_02_Topic_Word_Scores.html"
            fig = self.model.visualize_barchart(
                top_n_topics=Config.TOP_N_TOPICS_VISUALIZATION
            )
            fig.write_html(str(output_path))
            print(f"[2/5] ✅ Saved: {output_path}")
        except Exception as e:
            print(f"[2/5] ⚠️  Error generating barchart: {e}")
    
    def _generate_hierarchy(self):
        """Hierarchical clustering"""
        try:
            output_path = Config.FIGURES_DIR / "Viz_03_Hierarchical_Clustering.html"
            fig = self.model.visualize_hierarchy()
            fig.write_html(str(output_path))
            print(f"[3/5] ✅ Saved: {output_path}")
        except Exception as e:
            print(f"[3/5] ⚠️  Error generating hierarchy: {e}")
    
    def _generate_heatmap(self):
        """Topic similarity heatmap"""
        try:
            output_path = Config.FIGURES_DIR / "Viz_04_Topic_Similarity_Heatmap.html"
            fig = self.model.visualize_heatmap()
            fig.write_html(str(output_path))
            print(f"[4/5] ✅ Saved: {output_path}")
        except Exception as e:
            print(f"[4/5] ⚠️  Error generating heatmap: {e}")
    
    def _generate_topics_over_time(self):
        """Topics over time - EXTENDED TO 2025"""
        if 'Year' not in self.df_aligned.columns:
            print(f"[5/5] ⚠️  Skipping time series: Year column not found")
            return
        
        try:
            timestamps = pd.to_datetime(
                self.df_aligned['Year'].astype(str), 
                format='%Y', 
                errors='coerce'
            )
            
            print(f"    Calculating temporal dynamics...")
            topics_over_time = self.model.topics_over_time(
                self.texts,
                timestamps,
                global_tuning=True,
                evolution_tuning=True
            )
            
            # Apply year cutoff (now 2025!)
            topics_over_time_filtered = topics_over_time[
                topics_over_time["Timestamp"].dt.year <= Config.TIME_CUTOFF_YEAR
            ]
            
            output_path = Config.FIGURES_DIR / f"Viz_05_Topics_Over_Time_to_{Config.TIME_CUTOFF_YEAR}.html"
            fig = self.model.visualize_topics_over_time(
                topics_over_time_filtered,
                top_n_topics=12
            )
            fig.write_html(str(output_path))
            
            print(f"[5/5] ✅ Saved: {output_path}")
            print(f"      Time range: {topics_over_time_filtered['Timestamp'].dt.year.min()} - {Config.TIME_CUTOFF_YEAR}")
            
        except Exception as e:
            print(f"[5/5] ⚠️  Error generating time series: {e}")


# ============================================================================
# 9. MAIN PIPELINE
# ============================================================================

def main():
    """Main execution pipeline - OPTIMIZED v5.1"""
    
    print("\n" + "="*80)
    print("CIRP BIBLIOMETRIC ANALYSIS - BERTopic Pipeline v5.1 OPTIMIZED")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🎯 OPTIMIZATIONS:")
    print("   • Fewer, more interpretable topics (target: 15-25)")
    print("   • Fixed stability calculation bug")
    print("   • Research papers only (reviews excluded)")
    print("   • Outliers kept separate for manual review")
    print("   • Extended time analysis to 2025")
    print("   • Quality metrics CSV export")
    print("="*80 + "\n")
    
    # Setup
    set_seed()
    create_directories()
    
    # 1. Load data
    input_path = Config.DATA_DIR / Config.INPUT_FILE
    if not input_path.exists():
        input_path = Path("/mnt/user-data/uploads") / Config.INPUT_FILE
        if not input_path.exists():
            raise FileNotFoundError(f"Dataset not found: {Config.INPUT_FILE}")
    
    print(f"📂 Loading dataset: {input_path}")
    df = pd.read_csv(input_path)
    
    # 2. Filter reviews - NEW!
    if Config.EXCLUDE_REVIEWS and 'Document Type' in df.columns:
        n_before = len(df)
        df = df[df['Document Type'] != 'Review'].reset_index(drop=True)
        n_after = len(df)
        print(f"🔬 Filtered reviews: {n_before:,} → {n_after:,} (-{n_before - n_after} reviews)")
    
    # 3. Validate data
    validator = DataValidator(df)
    validation_report = validator.validate_and_report()
    
    # 4. Preprocess texts
    preprocessor = TextPreprocessor(synonym_mode="safe")
    text_embed, text_vec, valid_indices = preprocessor.process_corpus(
        df['Title'].fillna(""),
        df['Abstract'].fillna("")
    )
    
    df_filtered = df.iloc[valid_indices].reset_index(drop=True)
    
    # 5. Build model
    model_builder = BERTopicModelBuilder(preprocessor.stopwords_list)
    model = model_builder.build_model()
    
    # 6. Generate embeddings
    print("\n" + "="*80)
    print("EMBEDDING GENERATION")
    print("="*80)
    print(f"🔢 Generating embeddings for {len(text_embed):,} documents...")
    embeddings = model.embedding_model.encode(
        text_embed,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    print(f"✅ Embeddings shape: {embeddings.shape}")
    
    # 7. Train model
    trainer = ModelTrainer(model)
    topics, probs = trainer.train(text_vec, embeddings)
    
    # 8. Optionally reduce outliers
    topics = trainer.reduce_outliers(text_vec, embeddings)
    
    # 9. Calculate metrics (with fixed stability!)
    metrics_calc = MetricsCalculator(model, text_vec, embeddings)
    metrics = metrics_calc.calculate_all_metrics()
    
    # 10. Export data
    exporter = DataExporter(model, df, valid_indices, topics, probs, metrics)
    exporter.export_all()
    
    # 11. Generate visualizations
    df_aligned = df_filtered.copy()
    df_aligned['Topic'] = topics
    viz_gen = VisualizationGenerator(model, text_vec, df_aligned)
    viz_gen.generate_all()
    
    # 12. Save final report
    _save_final_report(validation_report, metrics, topics)
    
    print("\n" + "="*80)
    print("PIPELINE COMPLETE - v5.1 OPTIMIZED")
    print("="*80)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seed: {Config.SEED}")
    print("\n📊 Results ready for publication:")
    print(f"   • CSV files: {Config.RESULTS_DIR}")
    print(f"   • Visualizations: {Config.FIGURES_DIR}")
    print("\n🎯 Next steps:")
    print("   1. Review quality metrics in 05_Quality_Metrics.csv")
    print("   2. Validate topics in 02_Top_Papers_Per_Topic.csv")
    print("   3. Check outliers in 06_Outliers_Analysis.csv (if any)")
    print("   4. Consider manual topic merging if needed")
    print("="*80 + "\n")


def _save_final_report(validation_report: Dict, metrics: Dict, topics: List[int]):
    """Save comprehensive analysis report"""
    
    report_path = Config.RESULTS_DIR / "00_Analysis_Report.txt"
    
    n_topics = len(set(topics)) - (1 if -1 in topics else 0)
    n_outliers = sum(1 for t in topics if t == -1)
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("CIRP BIBLIOMETRIC ANALYSIS - FINAL REPORT v5.1 OPTIMIZED\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pipeline Version: v5.1 OPTIMIZED\n")
        f.write(f"Random Seed: {Config.SEED}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OPTIMIZATIONS APPLIED\n")
        f.write("="*80 + "\n")
        f.write(f"• UMAP n_neighbors: {Config.UMAP_N_NEIGHBORS} (↑ from 15)\n")
        f.write(f"• HDBSCAN min_cluster_size: {Config.HDBSCAN_MIN_CLUSTER_SIZE} (↑ from 10)\n")
        f.write(f"• HDBSCAN min_samples: {Config.HDBSCAN_MIN_SAMPLES} (↑ from 2)\n")
        f.write(f"• Vectorizer min_df: {Config.VECTORIZER_MIN_DF} (↑ from 5)\n")
        f.write(f"• Vectorizer max_df: {Config.VECTORIZER_MAX_DF} (NEW)\n")
        f.write(f"• Reviews excluded: {Config.EXCLUDE_REVIEWS}\n")
        f.write(f"• Outlier reduction: {Config.REDUCE_OUTLIERS}\n")
        f.write(f"• Time cutoff: {Config.TIME_CUTOFF_YEAR}\n\n")
        
        f.write("="*80 + "\n")
        f.write("DATA SUMMARY\n")
        f.write("="*80 + "\n")
        f.write(f"Total documents analyzed: {validation_report['total_documents']:,}\n")
        f.write(f"Valid documents (after preprocessing): {len(topics):,}\n")
        f.write(f"Time coverage: {validation_report['temporal_distribution']['min']} - ")
        f.write(f"{validation_report['temporal_distribution']['max']}\n\n")
        
        f.write("="*80 + "\n")
        f.write("TOPIC MODELING RESULTS\n")
        f.write("="*80 + "\n")
        f.write(f"Topics discovered: {n_topics}\n")
        f.write(f"Outlier documents: {n_outliers} ({(n_outliers/len(topics)*100):.1f}%)\n\n")
        
        f.write("="*80 + "\n")
        f.write("QUALITY METRICS\n")
        f.write("="*80 + "\n")
        f.write(f"Topic Diversity: {metrics['diversity']:.4f}")
        f.write(" (✅ Good)" if metrics['diversity'] > 0.80 else " (⚠️ Review)\n")
        f.write(f"Coherence C_v: {metrics['coherence_cv']:.4f}")
        f.write(" (✅ Good)" if metrics['coherence_cv'] > 0.65 else " (⚠️ Review)\n")
        f.write(f"Topic Stability: {metrics['stability']:.4f}")
        f.write(" (✅ Good)" if metrics['stability'] > 0.75 else " (⚠️ Review)\n\n")
        
        f.write("="*80 + "\n")
        f.write("OUTPUT FILES\n")
        f.write("="*80 + "\n")
        f.write("Results:\n")
        f.write("  • 00_Analysis_Report.txt - This file\n")
        f.write("  • 01_Full_Document_Assignments.csv - All documents with topics\n")
        f.write("  • 02_Top_Papers_Per_Topic.csv - Best papers for qualitative review\n")
        f.write("  • 03_Topic_Metadata_Keywords.csv - Topic keywords and stats\n")
        f.write("  • 04_Summary_Statistics.csv - Per-topic statistics\n")
        f.write("  • 05_Quality_Metrics.csv - Quality metrics with targets (NEW)\n")
        if n_outliers > 0:
            f.write("  • 06_Outliers_Analysis.csv - Outliers for manual review (NEW)\n")
        f.write("\nVisualizations:\n")
        f.write("  • Viz_01_Intertopic_Distance_Map.html\n")
        f.write("  • Viz_02_Topic_Word_Scores.html\n")
        f.write("  • Viz_03_Hierarchical_Clustering.html\n")
        f.write("  • Viz_04_Topic_Similarity_Heatmap.html\n")
        f.write(f"  • Viz_05_Topics_Over_Time_to_{Config.TIME_CUTOFF_YEAR}.html\n\n")
        
        f.write("="*80 + "\n")
    
    print(f"\n📄 Final report saved: {report_path}")


if __name__ == "__main__":
    main()
