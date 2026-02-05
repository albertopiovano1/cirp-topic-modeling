# CIRP Bibliometric Analysis - BERTopic Pipeline v5.0

Advanced Topic Modeling for Manufacturing Science & Technology Research

**Target Journal:** CIRP Journal of Manufacturing Science and Technology (JMST)

**Authors:** Alberto Piovano  
**Institution:** Politecnico di Torino - Department of Management and Production Engineering (DIMEAS)

---

## 📋 Overview

This pipeline performs advanced topic modeling on manufacturing research literature using BERTopic, designed for high-impact bibliometric studies. The system implements publication-ready quality metrics and comprehensive data exports for qualitative analysis.

### Key Features

- ✅ **Rigorous Data Validation**: Comprehensive quality reporting (coverage, duplicates, temporal distribution)
- ✅ **Dual Text Preprocessing**: Separate optimization for embeddings vs. vectorization
- ✅ **Controlled Synonym Normalization**: Safe mode (unambiguous) + optional aggressive mode
- ✅ **Publication-Ready Metrics**: Diversity, Coherence C_v, Stability (bootstrap)
- ✅ **Time-Series Analysis**: Temporal dynamics with configurable cutoff year
- ✅ **Qualitative Sampling**: Top-N papers per topic for manual review
- ✅ **Reproducibility**: Fixed random seeds across all stochastic components

---

## 📁 Project Structure

```
cirp-topic-modeling/
│
├── data/                          # Input data directory
│   └── CIRP_researchonly.csv      # Your dataset
│
├── notebooks/                     # Jupyter notebooks (optional)
│   └── exploratory_analysis.ipynb
│
├── results/                       # Generated CSV exports
│   ├── 00_Analysis_Report.txt
│   ├── 01_Full_Document_Assignments.csv
│   ├── 02_Top_Papers_Per_Topic.csv
│   ├── 03_Topic_Metadata_Keywords.csv
│   └── 04_Summary_Statistics.csv
│
├── figures/                       # Interactive visualizations
│   ├── Viz_01_Intertopic_Distance_Map.html
│   ├── Viz_02_Topic_Word_Scores.html
│   ├── Viz_03_Hierarchical_Clustering.html
│   ├── Viz_04_Topic_Similarity_Heatmap.html
│   └── Viz_05_Topics_Over_Time_to_2024.html
│
├── cirp_bertopic_v5_0.py         # Main pipeline script
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone repository
git clone https://github.com/albertopiovano1/cirp-topic-modeling.git
cd cirp-topic-modeling

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm
```

### 2. Prepare Data

Place your dataset in the `data/` directory:
- **Required file:** `CIRP_researchonly.csv`
- **Required columns:** `Title`, `Abstract`, `Year`, `Document Type`
- **Optional columns:** `Authors`, `DOI`, `Cited by`, `Author Keywords`, etc.

### 3. Run Pipeline

```bash
python cirp_bertopic_v5_0.py
```

**Expected Runtime:**
- Small corpus (~1K docs): ~5-10 minutes
- Medium corpus (~5K docs): ~20-30 minutes  
- Large corpus (~10K+ docs): ~45-60 minutes

*Note: First run downloads embedding model (~1.3GB)*

---

## ⚙️ Configuration

Edit the `Config` class in `cirp_bertopic_v5_0.py`:

### Key Parameters

```python
class Config:
    # Reproducibility
    SEED = 76
    
    # Data filtering
    MIN_WORDS_PER_DOC = 10
    MIN_ABSTRACT_LENGTH = 50
    
    # BERTopic model
    EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
    UMAP_N_NEIGHBORS = 15
    UMAP_N_COMPONENTS = 5
    HDBSCAN_MIN_CLUSTER_SIZE = 10
    HDBSCAN_MIN_SAMPLES = 2
    
    # Analysis
    TIME_CUTOFF_YEAR = 2024
    TOP_N_PAPERS_PER_TOPIC = 5
```

### Synonym Normalization Modes

Control synonym mapping via `TextPreprocessor(synonym_mode=...)`:

- **`"safe"`** (default): Only unambiguous acronyms (AI, IoT, RFID, etc.)
- **`"aggressive"`**: Includes broader synonyms (optimize, analyze, manufacturing)
- **`"off"`**: No synonym normalization

Example:
```python
preprocessor = TextPreprocessor(synonym_mode="safe")
```

---

## 📊 Output Files

### CSV Exports (results/)

1. **`00_Analysis_Report.txt`**  
   Comprehensive summary: data quality, metrics, configuration

2. **`01_Full_Document_Assignments.csv`**  
   All documents with topic assignments and probabilities

3. **`02_Top_Papers_Per_Topic.csv`**  
   Top-5 papers per topic for qualitative review (sorted by probability)

4. **`03_Topic_Metadata_Keywords.csv`**  
   Topic IDs, names, representative keywords, document counts

5. **`04_Summary_Statistics.csv`**  
   Per-topic statistics: size, average citations, year range, peak year

### Visualizations (figures/)

All visualizations are interactive HTML files (open in browser):

1. **Intertopic Distance Map**: 2D projection of topic similarities
2. **Topic Word Scores**: Bar charts of top words per topic
3. **Hierarchical Clustering**: Dendrogram of topic relationships
4. **Topic Similarity Heatmap**: Matrix of pairwise topic similarities
5. **Topics Over Time**: Temporal evolution (customizable cutoff year)

---

## 📈 Quality Metrics

The pipeline calculates three publication-ready metrics:

### 1. **Topic Diversity**
Proportion of unique words across all topics  
- Range: [0, 1]
- Interpretation: Higher = more distinct topics

### 2. **Coherence C_v**
Semantic coherence of topic keywords (Gensim implementation)  
- Range: [0, 1]
- Interpretation: Higher = more semantically coherent topics

### 3. **Topic Stability**
Consistency of topics across bootstrap samples  
- Range: [0, 1]
- Interpretation: Higher = more reproducible topics

---

## 🔬 Methodological Details

### Dual Text Preprocessing Strategy

The pipeline generates **two text versions** from each document:

1. **`text_embed`**: Natural language for embeddings  
   - Conservative cleaning (preserve structure)
   - Used for: Sentence-BERT embeddings, UMAP, HDBSCAN

2. **`text_vec`**: Optimized for vectorization  
   - Aggressive cleaning (compound terms with underscores)
   - Used for: c-TF-IDF, CountVectorizer, topic labeling

**Rationale:** Embedding models perform best with natural language, while TF-IDF benefits from explicit multi-word term detection.

### Outlier Reduction

Two-stage process:
1. Initial clustering with HDBSCAN
2. Embedding-based reassignment of outliers to nearest topics

### Stability Calculation

Bootstrap resampling method:
1. Resample corpus (100% size with replacement)
2. Train second model on bootstrap sample
3. Calculate topic centroid similarities
4. Average maximum similarity scores

---

## 🎯 Use Cases

This pipeline is designed for:

- 📚 **Systematic Literature Reviews**: Automated topic discovery in large corpora
- 📊 **Bibliometric Studies**: Quantitative analysis of research trends
- 🔍 **Research Gap Identification**: Finding under-explored areas
- 📈 **Trend Analysis**: Temporal evolution of research themes
- 🏆 **High-Impact Publications**: Nature/Science-style quantitative analyses

---

## 🛠️ Troubleshooting

### Common Issues

**Problem:** `ModuleNotFoundError: No module named 'bertopic'`  
**Solution:** Run `pip install -r requirements.txt`

**Problem:** `OSError: [E050] Can't find model 'en_core_web_sm'`  
**Solution:** Run `python -m spacy download en_core_web_sm`

**Problem:** Out of memory during embedding generation  
**Solution:** Reduce batch size in `encode()` call:
```python
embeddings = model.embedding_model.encode(
    text_embed,
    batch_size=32,  # Default: 64
    show_progress_bar=True
)
```

**Problem:** Too many outliers (>30%)  
**Solution:** Adjust HDBSCAN parameters:
- Decrease `min_cluster_size` (e.g., from 10 to 8)
- Decrease `min_samples` (e.g., from 2 to 1)

**Problem:** Topics too granular  
**Solution:** Increase `min_cluster_size` (e.g., from 10 to 15)

---

## 📝 Citation

If you use this pipeline in your research, please cite:

```
Piovano, A., Verna, E., Genta, G., & Galetto, M. (2026).
Trends in Advanced Manufacturing Technology Innovation: 
A BERTopic Analysis of CIRP Publications (2000-2025).
CIRP Journal of Manufacturing Science and Technology (in preparation).
```

---

## 📧 Contact

**Alberto Piovano**  
Department of Management and Production Engineering (DIMEAS)  
Politecnico di Torino  
Email: alberto.piovano@polito.it

---

## 📜 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 🙏 Acknowledgments

- **BERTopic**: Maarten Grootendorst (https://github.com/MaartenGr/BERTopic)
- **Sentence-BERT**: UKP Lab, TU Darmstadt
- **Inspiration**: Park et al. (2023) - "Trends in Advanced Manufacturing Technology Innovation"

---

## 🔄 Version History

### v5.0 (Current)
- ✨ Dual text preprocessing (embed + vec)
- ✨ Enhanced data validation with comprehensive reporting
- ✨ Controlled synonym normalization (safe/aggressive/off modes)
- ✨ Publication-ready metrics (Diversity, Coherence, Stability)
- ✨ Improved export structure for qualitative analysis
- ✨ Time-series with configurable cutoff year

### v4.0
- Initial working version
- Basic preprocessing and topic modeling
- Standard BERTopic visualizations

---

**Last Updated:** February 2026
