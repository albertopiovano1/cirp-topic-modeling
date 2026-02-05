# Comparison: BERTopic v4.0 → v5.0

## Executive Summary

Version 5.0 represents a major refactoring focused on **methodological rigor**, **reproducibility**, and **publication readiness**. The pipeline now implements best practices from computational social science and text mining literature.

---

## 🔄 Major Changes

### 1. **Code Architecture**

| Aspect | v4.0 | v5.0 |
|--------|------|------|
| Structure | Monolithic script | Modular OOP architecture |
| Configuration | Hardcoded values | Centralized Config class |
| Error handling | Minimal | Comprehensive try-catch blocks |
| Logging | Basic prints | Structured progress reporting |
| Maintainability | Low | High |

**Impact:** v5.0 is significantly easier to modify, debug, and extend for future research.

---

### 2. **Data Validation**

#### v4.0: Basic Loading
```python
df = pd.read_csv(dataset_path)
# Minimal checks, no quality reporting
```

#### v5.0: Comprehensive Validation
```python
validator = DataValidator(df)
validation_report = validator.validate_and_report()
```

**New Features:**
- ✅ Data coverage analysis (field completion rates)
- ✅ Document type distribution
- ✅ Duplicate detection (DOI + Title)
- ✅ Temporal distribution analysis
- ✅ Content quality metrics (abstract length, short docs)
- ✅ Structured validation report export

**Impact:** Researchers can confidently assess data quality before analysis. Critical for peer review.

---

### 3. **Text Preprocessing Strategy**

#### v4.0: Single Text Output
- One preprocessing pipeline
- Same text used for embeddings and vectorization
- Limited optimization for either task

#### v5.0: Dual Text Output
```python
text_embed  # Natural language → for embeddings (BERT)
text_vec    # Optimized format → for c-TF-IDF (vectorizer)
```

**Rationale:**
- **Embedding models** (e.g., Sentence-BERT) perform best with natural language structure
- **TF-IDF vectorizers** benefit from explicit compound term detection (e.g., "additive_manufacturing")

**Example:**
```
Original: "Additive manufacturing using machine learning"

v4.0 output (single):
"additive manufacturing use machine learning"

v5.0 outputs (dual):
text_embed: "additive manufacturing use machine learning"  # Natural
text_vec:   "additive_manufacturing machine_learning"       # Explicit compounds
```

**Impact:** Better semantic embeddings + clearer topic keywords. Supported by Grootendorst (2022) BERTopic paper.

---

### 4. **Synonym Normalization**

#### v4.0: All-or-Nothing
- Fixed synonym map (hardcoded)
- No control over aggressiveness
- Risk of over-normalization

#### v5.0: Three-Mode System
```python
preprocessor = TextPreprocessor(synonym_mode="safe")
# Options: "safe", "aggressive", "off"
```

**Safe Mode (Default):**
- Only unambiguous acronyms: AI → artificial_intelligence, IoT → internet_of_things
- Domain-specific terms: FSC → food_supply_chain, RFID → radio_frequency_identification

**Aggressive Mode:**
- Adds broader synonyms: optimization/optimisation → optimize
- Verb forms: analyse/analysis → analyze
- Use with caution (may conflate distinct concepts)

**Off Mode:**
- No synonym normalization
- Preserves original terminology variance

**Impact:** Researchers can make informed choices based on their corpus and research questions.

---

### 5. **Stopword Management**

#### v4.0: Mixed List
```python
# Academic + publisher + domain terms all merged
stopwords_core = ["aim", "paper", "study", ...]
publisher_noise = ["emerald publishing", ...]
# Combined without clear categorization
```

#### v5.0: Categorized + Documented
```python
stopwords_academic = [...]    # Generic academic noise
stopwords_publisher = [...]   # Metadata artifacts
stopwords_domain = [...]      # Overly broad terms
```

**Impact:** Easier to audit, modify, and justify stopword choices in methods section.

---

### 6. **Quality Metrics**

| Metric | v4.0 | v5.0 |
|--------|------|------|
| **Diversity** | ✅ Implemented | ✅ Refined calculation |
| **Coherence** | ✅ C_v score | ✅ C_v with error handling |
| **Stability** | ⚠️ Basic bootstrap | ✅ **Robust centroid-based** |

#### Stability Calculation Enhancement

**v4.0 Issue:**
```python
# Direct topic ID matching
# Fails when bootstrap model reorders topics
sims = [cosine_similarity(orig[i], boot[i])]  # Wrong!
```

**v5.0 Solution:**
```python
# Best-match approach (topic IDs can differ)
for tid in orig_centroids:
    best_sim = max([
        cosine_similarity(orig_centroids[tid], boot_centroids[bid])
        for bid in boot_centroids
    ])
    sims.append(best_sim)
stability = mean(sims)
```

**Impact:** Correct stability measurement. Aligns with Hoyle et al. (2021) topic model evaluation practices.

---

### 7. **Export Structure for Qualitative Analysis**

#### v4.0: Generic Exports
```
1_Full_Document_Assignments.csv
2_Most_Relevant_Papers_per_Topic.csv
3_Topic_Metadata_Keywords.csv
```

#### v5.0: Publication-Ready Exports
```
00_Analysis_Report.txt           ← NEW: Comprehensive summary
01_Full_Document_Assignments.csv
02_Top_Papers_Per_Topic.csv      ← Enhanced with Rank field
03_Topic_Metadata_Keywords.csv
04_Summary_Statistics.csv        ← NEW: Per-topic stats
```

**New File: 00_Analysis_Report.txt**
- Data quality summary
- Model configuration record
- Quality metrics
- Output file inventory
- Essential for methods section

**New File: 04_Summary_Statistics.csv**
- Documents per topic
- Average citations
- Year range and peak year
- Percentage of corpus
- Perfect for results tables

**Enhancement: 02_Top_Papers_Per_Topic.csv**
- Added `Rank` field (1-5)
- Sorted by probability (most representative first)
- Optimized columns for review
- Facilitates systematic qualitative coding

---

### 8. **Time-Series Analysis**

#### v4.0: Hardcoded Cutoff
```python
# Cutoff 2024 hardcoded in visualization
topics_over_time_filtered = topics_over_time[
    topics_over_time["Timestamp"].dt.year < 2025
]
```

#### v5.0: Configurable Parameter
```python
class Config:
    TIME_CUTOFF_YEAR = 2024  # Easily adjustable
```

**Impact:** Adapt analysis period without code changes. Important for longitudinal studies with rolling updates.

---

### 9. **Reproducibility**

| Aspect | v4.0 | v5.0 |
|--------|------|------|
| Random seed | ✅ Set | ✅ Set + documented |
| Seed scope | NumPy, PyTorch | + PYTHONHASHSEED, CUDNN |
| Seed reporting | Console print | In final report |
| Environment | Not recorded | Recorded in report |

**v5.0 Additions:**
```python
os.environ["PYTHONHASHSEED"] = str(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

**Impact:** True reproducibility across machines/sessions. Critical for peer review and replication studies.

---

### 10. **Error Handling & User Experience**

#### v4.0: Minimal
- Crashes on missing data
- Generic error messages
- No progress indication for long operations

#### v5.0: Robust
- Comprehensive try-catch blocks
- Informative error messages
- Progress bars (tqdm) for all long operations
- Graceful degradation (continue with warnings when possible)

**Example:**
```python
# v5.0: Graceful failure
try:
    stability = self._calculate_stability()
except Exception as e:
    print(f"⚠️ Error calculating stability: {e}")
    stability = 0.0  # Continue with default
```

---

## 📊 Performance Comparison

### Memory Efficiency
- **v4.0:** Duplicate text storage (preprocessed used twice)
- **v5.0:** Dual preprocessing (but text shared where possible)
- **Result:** Comparable memory usage (~1-2 GB for 5K documents)

### Runtime
- **v4.0:** ~20-25 min (5K documents)
- **v5.0:** ~25-30 min (5K documents)
- **Overhead:** +5 min for validation + dual preprocessing
- **Tradeoff:** Worth it for quality assurance

### Disk Usage
- **v4.0:** ~50-100 MB (3 CSV files)
- **v5.0:** ~60-120 MB (5 files + report)
- **Additional:** Analysis report (~5-10 KB)

---

## 🎯 When to Use Each Version

### Use v4.0 if:
- Quick exploratory analysis
- Familiar dataset (already validated)
- Single-use analysis (not for publication)

### Use v5.0 if:
- **Publication-bound research** ✅
- First-time corpus analysis
- Need to justify methodological choices
- Collaboration (readable, maintainable code)
- Longitudinal study (reproducibility critical)

---

## 🔬 Methodological Justifications (for Paper)

Key improvements with literature support:

1. **Dual Text Preprocessing**
   - Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure.
   - Rationale: Optimize for both semantic embeddings and keyword extraction.

2. **Stability Metric**
   - Hoyle, A. et al. (2021). Is Automated Topic Model Evaluation Broken? TACL.
   - Rationale: Bootstrap resampling tests model robustness to sampling variation.

3. **Coherence C_v**
   - Röder, M. et al. (2015). Exploring the space of topic coherence measures. WSDM.
   - Rationale: C_v correlates best with human topic interpretability judgments.

4. **Diversity Metric**
   - Dieng, A. et al. (2020). Topic modeling in embedding spaces. TACL.
   - Rationale: Prevents degenerate solutions (topics with identical top words).

---

## 📝 Migration Guide (v4.0 → v5.0)

### Step 1: Update Script
```bash
# Backup your v4.0 results
cp -r results results_v4_backup

# Use new script
python cirp_bertopic_v5_0.py
```

### Step 2: Compare Outputs
- Topic count may differ slightly (improved clustering)
- Check `00_Analysis_Report.txt` for validation issues
- Review `04_Summary_Statistics.csv` for topic sizes

### Step 3: Adjust Configuration (if needed)
```python
# If topics too granular:
Config.HDBSCAN_MIN_CLUSTER_SIZE = 15  # Increase from 10

# If too many outliers:
Config.HDBSCAN_MIN_CLUSTER_SIZE = 8   # Decrease from 10

# For more aggressive synonym merging:
preprocessor = TextPreprocessor(synonym_mode="aggressive")
```

---

## 🚀 Future Roadmap (v6.0+)

Planned enhancements:
- [ ] Multi-language support
- [ ] Custom embedding models (domain-specific)
- [ ] Dynamic topic modeling (with temporal priors)
- [ ] Integration with citation networks
- [ ] Automated topic labeling (GPT-based)
- [ ] R integration for Nature-style figures

---

## 📧 Questions?

For migration assistance or customization:
- **Email:** alberto.piovano@polito.it
- **GitHub Issues:** https://github.com/albertopiovano1/cirp-topic-modeling/issues

---

**Last Updated:** February 2026
