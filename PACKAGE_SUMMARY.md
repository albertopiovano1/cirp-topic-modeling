# 🚀 CIRP Bibliometric Analysis - Complete Package

**Version:** 5.0  
**Date:** February 5, 2026  
**Author:** Alberto Piovano (Politecnico di Torino - DIMEAS)

---

## 📦 Package Contents

This package contains everything needed to run a **publication-ready bibliometric analysis** using BERTopic on CIRP manufacturing research literature.

### Core Files

1. **`cirp_bertopic_v5_0.py`** (52 KB)
   - Main pipeline script
   - Modular, documented, production-ready
   - ~1200 lines of well-structured Python

2. **`requirements.txt`** (737 B)
   - All Python dependencies
   - Tested with Python 3.8+
   - Includes versions for reproducibility

3. **`setup_project.py`** (3.6 KB)
   - Automated project initialization
   - Creates directory structure
   - Generates placeholder files

### Documentation

4. **`README.md`** (9.2 KB)
   - Complete project overview
   - Quick start guide
   - Configuration reference
   - Troubleshooting section

5. **`COMPARISON_v4_v5.md`** (11 KB)
   - Detailed comparison with previous version
   - Methodological improvements
   - Migration guide
   - Literature citations

6. **`COLAB_GUIDE.md`** (9.9 KB)
   - Step-by-step Google Colab tutorial
   - GPU optimization tips
   - Colab-specific troubleshooting
   - Mobile workflow

7. **`ITERATION_PLAN.md`** (12 KB)
   - Iterative workflow methodology
   - 7-iteration refinement plan
   - Tracking templates
   - Success criteria

8. **`.gitignore`** (847 B)
   - Proper Git configuration
   - Excludes large files and outputs
   - Preserves directory structure

### Dataset

9. **`data/CIRP_researchonly.csv`** (Included)
   - 4,729 research papers
   - CIRP Annals + CIRP JMST (2000-2025)
   - Complete metadata (authors, citations, keywords)

### Directory Structure

```
cirp-topic-modeling/
├── data/                  ← Your input dataset
├── notebooks/             ← Jupyter notebooks (optional)
├── results/               ← CSV exports (generated)
├── figures/               ← Interactive visualizations (generated)
├── models/                ← Model checkpoints (optional)
└── [scripts & docs]
```

---

## ⚡ Quick Start (3 Steps)

### Local Machine

```bash
# 1. Install dependencies
pip install -r requirements.txt
python -m spacy download en_core_web_sm

# 2. Initialize project
python setup_project.py

# 3. Run pipeline
python cirp_bertopic_v5_0.py
```

### Google Colab

```python
# 1. Clone repository
!git clone https://github.com/albertopiovano1/cirp-topic-modeling.git
%cd cirp-topic-modeling

# 2. Install dependencies
!pip install -q bertopic sentence-transformers umap-learn hdbscan gensim spacy tqdm
!python -m spacy download en_core_web_sm

# 3. Run pipeline
!python cirp_bertopic_v5_0.py
```

**Expected Runtime:** 20-30 minutes for 4,729 documents

---

## 🎯 Key Features

### 1. Rigorous Data Validation
- Comprehensive quality report (coverage, duplicates, temporal distribution)
- Automatic detection of data issues
- Structured validation logging

### 2. Dual Text Preprocessing
- `text_embed`: Natural language → for embeddings (Sentence-BERT)
- `text_vec`: Optimized format → for c-TF-IDF (keyword extraction)
- Result: Better semantic embeddings + clearer topic labels

### 3. Controlled Synonym Normalization
- **Safe mode** (default): Only unambiguous acronyms (AI, IoT, RFID)
- **Aggressive mode**: Includes broader synonyms (optimize, manufacturing)
- **Off mode**: No normalization
- Result: Transparent, justifiable preprocessing choices

### 4. Publication-Ready Metrics
- **Diversity**: Proportion of unique words across topics
- **Coherence C_v**: Semantic coherence (Gensim implementation)
- **Stability**: Bootstrap resampling consistency
- Result: Quantitative quality assessment for methods section

### 5. Qualitative Sampling
- Top-N papers per topic (sorted by probability)
- Optimized columns for manual review
- Facilitates systematic qualitative coding
- Result: Human-in-the-loop validation

### 6. Time-Series Analysis
- Temporal evolution of topics
- Configurable cutoff year (default: 2024)
- Identifies emerging vs. declining themes
- Result: Longitudinal insights for discussion section

---

## 📊 Expected Outputs

### CSV Files (results/)

1. **00_Analysis_Report.txt**
   - Data quality summary
   - Model configuration
   - Quality metrics
   - Output file inventory

2. **01_Full_Document_Assignments.csv**
   - All 4,729 documents
   - Topic assignments + probabilities
   - Metadata (year, authors, citations)

3. **02_Top_Papers_Per_Topic.csv**
   - Top-5 papers × N topics
   - Ranked by relevance
   - Optimized for qualitative review

4. **03_Topic_Metadata_Keywords.csv**
   - Topic IDs and names
   - Representative keywords
   - Document counts

5. **04_Summary_Statistics.csv**
   - Per-topic statistics
   - Size, average citations, year range
   - Peak year of activity

### Visualizations (figures/)

All interactive HTML (open in browser):

1. **Viz_01_Intertopic_Distance_Map.html**
   - 2D projection of topic relationships
   - Based on t-SNE of topic embeddings

2. **Viz_02_Topic_Word_Scores.html**
   - Bar charts of top words per topic
   - c-TF-IDF scores

3. **Viz_03_Hierarchical_Clustering.html**
   - Dendrogram of topic similarities
   - Useful for identifying mergeable topics

4. **Viz_04_Topic_Similarity_Heatmap.html**
   - Pairwise topic similarity matrix
   - Based on cosine similarity

5. **Viz_05_Topics_Over_Time_to_2024.html**
   - Temporal evolution (2000-2024)
   - Smoothed trends with confidence intervals

---

## 🔬 Methodological Rigor

### Reproducibility
- Fixed random seeds (SEED = 76)
- Deterministic CUDNN settings
- All parameters logged in report
- Version control friendly

### Quality Assurance
- Multiple metrics (diversity, coherence, stability)
- Outlier handling (embedding-based reassignment)
- Comprehensive error handling
- Graceful degradation

### Transparency
- Modular, well-documented code
- Clear separation of concerns (OOP design)
- Configurable parameters (no magic numbers)
- Extensive inline comments

---

## 📚 Literature Support

Key methodological choices are supported by peer-reviewed literature:

1. **BERTopic Framework**
   - Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. *arXiv preprint*.

2. **Coherence Metric**
   - Röder, M., Both, A., & Hinneburg, A. (2015). Exploring the space of topic coherence measures. *WSDM*.

3. **Stability Analysis**
   - Hoyle, A. et al. (2021). Is Automated Topic Model Evaluation Broken? *TACL*.

4. **Evaluation Framework**
   - Dieng, A. et al. (2020). Topic modeling in embedding spaces. *TACL*.

---

## 🎓 Academic Context

### Target Journal
**CIRP Journal of Manufacturing Science and Technology (JMST)**
- Impact Factor: ~4.5
- Focus: Advanced manufacturing technologies
- Methodological rigor expected

### Study Type
**Bibliometric Analysis + Topic Modeling**
- Corpus: 4,729 papers (CIRP Annals + JMST, 2000-2025)
- Method: BERTopic (transformer-based topic modeling)
- Goal: Map evolution of manufacturing research
- Inspiration: Park et al. (2023) "Trends in Advanced Manufacturing Technology Innovation"

### Expected Contributions
1. **Empirical:** First comprehensive topic analysis of CIRP publications
2. **Methodological:** Rigorous pipeline for reproducible bibliometric studies
3. **Practical:** Identification of research gaps and emerging trends

---

## 🛠️ Customization Guide

### Common Adjustments

**More Topics (Finer Granularity):**
```python
Config.HDBSCAN_MIN_CLUSTER_SIZE = 8  # Decrease from 10
```

**Fewer Topics (Broader Themes):**
```python
Config.HDBSCAN_MIN_CLUSTER_SIZE = 15  # Increase from 10
```

**Reduce Outliers:**
```python
Config.HDBSCAN_MIN_SAMPLES = 1  # Decrease from 2
```

**Different Time Period:**
```python
Config.TIME_CUTOFF_YEAR = 2023  # Change from 2024
```

**More Papers per Topic:**
```python
Config.TOP_N_PAPERS_PER_TOPIC = 10  # Increase from 5
```

---

## 📈 Success Metrics

After running, check:

✅ **Data Quality**
- 100% of papers have abstracts
- <5% duplicate DOIs
- Temporal coverage: 2000-2024

✅ **Topic Quality**
- Coherence C_v > 0.65
- Diversity > 0.80
- Stability > 0.75
- <10% outliers

✅ **Interpretability**
- Topic keywords are specific (not generic)
- Top papers match topic theme
- Clear separation in intertopic map

---

## 🚦 Next Steps

### Immediate (First Run)

1. ✅ Review this summary
2. ⏳ Run Iteration 0 (baseline):
   ```bash
   python cirp_bertopic_v5_0.py
   ```
3. ⏳ Inspect outputs (20-30 min runtime)
4. ⏳ Upload results for review

### Short-Term (Refinement)

5. Iterate based on feedback (3-7 iterations typical)
6. Validate topics qualitatively (manual review)
7. Generate publication-ready figures

### Long-Term (Paper Writing)

8. Draft methods section (configuration → table)
9. Create results tables (top topics, statistics)
10. Prepare supplementary materials (full topic list)

---

## 📧 Support

### Technical Issues
- **GitHub:** https://github.com/albertopiovano1/cirp-topic-modeling/issues
- **Email:** alberto.piovano@polito.it

### Methodological Questions
- Review `COMPARISON_v4_v5.md` for rationale
- Check `ITERATION_PLAN.md` for troubleshooting
- Consult BERTopic documentation: https://maartengr.github.io/BERTopic/

---

## 📜 License

MIT License - See repository for details

---

## 🙏 Acknowledgments

- **BERTopic:** Maarten Grootendorst
- **Sentence-BERT:** UKP Lab, TU Darmstadt
- **CIRP Community:** For excellent open research

---

## ✅ Pre-Flight Checklist

Before running:

- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] spaCy model downloaded (`python -m spacy download en_core_web_sm`)
- [ ] Dataset in `data/CIRP_researchonly.csv`
- [ ] At least 4 GB free disk space
- [ ] ~30 minutes of computation time available

**All systems go? Run:** `python cirp_bertopic_v5_0.py`

---

**Good luck with your analysis!** 🎉

Last Updated: February 5, 2026
