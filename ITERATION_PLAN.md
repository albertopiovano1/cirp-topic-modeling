# CIRP Analysis: Iterative Workflow Plan

## 📋 Overview

This document outlines the **iterative development approach** for the CIRP bibliometric analysis. We'll work in cycles, testing, reviewing results, and refining the pipeline based on empirical feedback.

---

## 🔄 Workflow Methodology

### Philosophy: Test-Driven Topic Modeling

Unlike traditional coding where tests come first, in topic modeling:
1. **Run** the pipeline with initial parameters
2. **Inspect** the qualitative and quantitative outputs
3. **Diagnose** issues (too granular, too coarse, noisy topics)
4. **Adjust** parameters based on evidence
5. **Iterate** until satisfactory results

**Key Principle:** Topic quality is assessed **post-hoc** through human interpretation + metrics.

---

## 📂 GitHub Integration Strategy

### Repository Structure

```
cirp-topic-modeling/
├── data/
│   └── CIRP_researchonly.csv          # Your dataset (not tracked)
├── notebooks/
│   └── iteration_notes.md             # Track changes across iterations
├── results/
│   ├── iteration_01/                  # First run
│   ├── iteration_02/                  # After adjustments
│   └── ...
├── figures/
│   ├── iteration_01/
│   └── ...
├── cirp_bertopic_v5_0.py              # Main pipeline
├── requirements.txt
├── README.md
└── CHANGELOG.md                       # Version history
```

### Git Workflow

```bash
# Initial setup
git init
git add .
git commit -m "Initial pipeline v5.0"
git remote add origin https://github.com/albertopiovano1/cirp-topic-modeling.git
git push -u origin main

# After each iteration
mkdir results/iteration_XX
python cirp_bertopic_v5_0.py
# Results go to results/ and figures/

# Review locally
# Make notes in notebooks/iteration_notes.md

# Commit iteration
git add results/iteration_XX figures/iteration_XX notebooks/iteration_notes.md
git commit -m "Iteration XX: [brief description of changes]"
git push

# If you modified the code
git add cirp_bertopic_v5_0.py
git commit -m "Adjust HDBSCAN min_cluster_size: 10 → 15"
git push
```

---

## 🎯 Iteration Plan

### Iteration 0: Baseline Run

**Goal:** Establish baseline with default parameters

**Steps:**
1. ✅ Setup project structure
2. ✅ Copy dataset to `data/CIRP_researchonly.csv`
3. ✅ Install dependencies
4. Run pipeline with default config:
   ```bash
   python cirp_bertopic_v5_0.py
   ```

**Expected Output:**
- ~20-30 topics (typical for 4K+ corpus)
- 5-10% outliers
- First visualizations

**Review Checklist:**
- [ ] Check `00_Analysis_Report.txt` for data quality
- [ ] Open `Viz_01_Intertopic_Distance_Map.html` (topics should be distinct)
- [ ] Open `Viz_02_Topic_Word_Scores.html` (keywords should be interpretable)
- [ ] Review `02_Top_Papers_Per_Topic.csv` (do top papers match topic themes?)
- [ ] Check `04_Summary_Statistics.csv` (are topics balanced in size?)

**Decision Point:**
- **If topics look good:** Proceed to Iteration 1 (refinement)
- **If topics too granular:** Increase `HDBSCAN_MIN_CLUSTER_SIZE`
- **If topics too coarse:** Decrease `HDBSCAN_MIN_CLUSTER_SIZE`
- **If many outliers (>15%):** Adjust `HDBSCAN_MIN_SAMPLES`

---

### Iteration 1: Parameter Refinement

**Goal:** Optimize clustering for interpretability

**Common Adjustments:**

#### Issue 1: Too Many Small Topics (Fragmentation)
```python
# In cirp_bertopic_v5_0.py, Config class:
HDBSCAN_MIN_CLUSTER_SIZE = 15  # Increase from 10
```

#### Issue 2: Too Few Large Topics (Over-aggregation)
```python
HDBSCAN_MIN_CLUSTER_SIZE = 8   # Decrease from 10
```

#### Issue 3: High Outlier Rate (>15%)
```python
HDBSCAN_MIN_SAMPLES = 1        # Decrease from 2
# OR
HDBSCAN_MIN_CLUSTER_SIZE = 8   # More permissive clustering
```

#### Issue 4: Topics with Generic Keywords
```python
# Add terms to stopwords_domain in TextPreprocessor
stopwords_domain = [
    "food", "sector", "industry", "product", "system", 
    "solution", "process", "production", "company", "business",
    "manufacturing",  # If too generic in your corpus
    "technology",     # If appearing in every topic
]
```

**Run Updated Pipeline:**
```bash
# Rename previous iteration
mv results results_iteration_00
mv figures figures_iteration_00

# Run again
python cirp_bertopic_v5_0.py

# Compare
# Check if changes improved interpretability
```

---

### Iteration 2: Synonym Normalization Experiments

**Goal:** Test impact of aggressive synonym merging

**Test A: Conservative (Default)**
```python
preprocessor = TextPreprocessor(synonym_mode="safe")
```

**Test B: Aggressive**
```python
preprocessor = TextPreprocessor(synonym_mode="aggressive")
```

**Test C: Custom**
```python
# Add domain-specific synonyms
self.safe_synonyms = {
    # Existing...
    "additive": "additive_manufacturing",
    "subtractive": "subtractive_manufacturing",
    "hybrid_manufacturing": "additive_manufacturing",  # If treating as same
}
```

**Evaluation:**
Compare resulting topics:
- Are synonyms properly merged?
- Do topics become more coherent?
- Any over-merging of distinct concepts?

**Decision:** Choose mode that balances consolidation vs. specificity

---

### Iteration 3: Temporal Analysis Refinement

**Goal:** Ensure temporal trends are meaningful

**Check:**
1. Open `Viz_05_Topics_Over_Time_to_2024.html`
2. Look for:
   - ✅ Emerging topics (sharp rise in recent years)
   - ✅ Declining topics (research matured)
   - ✅ Stable topics (core research areas)
   - ❌ Erratic fluctuations (noise)

**If Noisy Trends:**
```python
# Adjust time series parameters
topics_over_time = self.model.topics_over_time(
    self.texts,
    timestamps,
    global_tuning=True,
    evolution_tuning=True,
    nr_bins=20  # Add: Smooth by binning years
)
```

**Alternative: Manual Binning**
```python
# In VisualizationGenerator._generate_topics_over_time()
# Group by 2-year periods instead of annual
df_aligned['Year_Bin'] = (df_aligned['Year'] // 2) * 2
```

---

### Iteration 4: Qualitative Validation

**Goal:** Manual review of topic coherence

**Process:**
1. Open `02_Top_Papers_Per_Topic.csv`
2. For each topic:
   - Read titles + abstracts of top 5 papers
   - Check: Do they genuinely share a theme?
   - Check: Does the topic name (keywords) match?

3. Create validation spreadsheet:
   ```
   Topic_ID | Topic_Name | Coherent? | Notes
   0        | additive_manufacturing machine_learning | Yes | Clear theme
   1        | optimization model system | No | Too generic - needs filtering
   ```

**Refinement Based on Review:**
- Topics that fail validation → add their generic keywords to stopwords
- Topics that are too similar → consider merging (see Iteration 5)

---

### Iteration 5: Topic Merging (If Needed)

**Goal:** Reduce redundancy in similar topics

**Use BERTopic's Built-in Merging:**
```python
# After initial fit
topics_to_merge = [[1, 5], [8, 12, 15]]  # Example: merge similar topics
model.merge_topics(text_vec, topics, topics_to_merge)
```

**OR: Hierarchical Reduction**
```python
# Reduce to target number of topics
model.reduce_topics(text_vec, topics, nr_topics=15)
```

**Evaluation:**
- Does merging improve interpretability?
- Check `Viz_03_Hierarchical_Clustering.html` to see natural groupings

---

### Iteration 6: Stability Analysis Deep Dive

**Goal:** Ensure robust topic structure

**Run Multiple Seeds:**
```python
seeds = [42, 76, 123, 456, 789]

for seed in seeds:
    Config.SEED = seed
    main()
    # Save results to results/stability_test/seed_{seed}/
```

**Analysis:**
1. Compare topic counts across seeds
2. Check if main themes (top 5-10 topics) are consistent
3. Calculate variance in metrics (diversity, coherence)

**Expected:**
- Topic **count** may vary slightly (±2-3)
- Top **themes** should be stable
- **Metrics** should have low variance (<0.05)

**If Unstable:**
- Increase `HDBSCAN_MIN_CLUSTER_SIZE` (larger clusters = more stable)
- Increase `UMAP_N_NEIGHBORS` (more global structure)

---

### Iteration 7: Finalization for Publication

**Goal:** Lock in configuration and generate publication-ready outputs

**Steps:**
1. Document final configuration in `METHODS.md`
2. Generate all visualizations at high resolution (if needed)
3. Create summary tables for paper:
   - Table 1: Top 10 topics by size
   - Table 2: Emerging vs. declining topics
   - Table 3: Most cited papers per topic

4. Prepare supplementary materials:
   - Full topic list (all topics)
   - Complete document assignments
   - Visualization gallery

**Commit Final Version:**
```bash
git tag -a v1.0 -m "Final version for CIRP JMST submission"
git push origin v1.0
```

---

## 📊 Iterative Tracking Template

Use this template in `notebooks/iteration_notes.md`:

```markdown
# Iteration Log

## Iteration 00 - Baseline (2026-02-05)
**Configuration:**
- HDBSCAN_MIN_CLUSTER_SIZE: 10
- HDBSCAN_MIN_SAMPLES: 2
- Synonym mode: safe

**Results:**
- Topics discovered: 23
- Outliers: 6.1%
- Diversity: 0.8423
- Coherence: 0.6891
- Stability: 0.7912

**Observations:**
- Topics generally coherent
- Topic 12 has very generic keywords ("model", "system")
- Topics 5 and 8 seem very similar (both about ML in manufacturing)

**Next Steps:**
- Add "model", "system" to domain stopwords
- Try merging topics 5 and 8
- → Iteration 01

---

## Iteration 01 - Stopword Refinement (2026-02-06)
**Changes:**
- Added "model", "system", "method" to stopwords_domain

**Results:**
- Topics discovered: 22 (merged one noisy topic)
- Outliers: 5.8%
- Diversity: 0.8612 (+0.0189)
- Coherence: 0.7123 (+0.0232)
- Stability: 0.7889 (-0.0023)

**Observations:**
- Topic keywords more specific
- Generic topic eliminated
- Slight decrease in stability (acceptable)

**Next Steps:**
- Validate with qualitative review
- → Iteration 02
```

---

## 🎯 Success Criteria (When to Stop Iterating)

Stop iterating when:
1. ✅ **Interpretability:** Topics have clear, specific keywords
2. ✅ **Coverage:** <10% outliers
3. ✅ **Balance:** No topics with <1% or >20% of corpus
4. ✅ **Coherence:** C_v score >0.65
5. ✅ **Stability:** Bootstrap stability >0.75
6. ✅ **Qualitative:** Top papers per topic consistently match theme

**Typical Range:** 3-7 iterations to reach publication quality

---

## 📝 Communication Plan

### After Each Iteration

**Quick Update:**
```
Subject: CIRP Analysis - Iteration X Complete

Topics discovered: XX
Key metric changes: [summary]
Main observation: [1-2 sentences]

Next iteration planned: [what you'll try]
```

### Before Final Run

**Detailed Report:**
- Configuration justification
- Iteration history summary
- Quality metrics comparison table
- Example topics with representative papers

---

## 🚀 Getting Started

### Your Next Steps:

1. **Copy dataset** to `data/CIRP_researchonly.csv`
   ```bash
   cp /mnt/user-data/uploads/CIRP_researchonly.csv ./data/
   ```

2. **Run Iteration 0** (baseline)
   ```bash
   python cirp_bertopic_v5_0.py
   ```

3. **Review outputs** (should take ~25 minutes)
   - Open HTML visualizations in browser
   - Read through CSV exports
   - Check metrics in `00_Analysis_Report.txt`

4. **Upload results** to our next conversation:
   - `00_Analysis_Report.txt`
   - Screenshots of 2-3 key visualizations
   - `04_Summary_Statistics.csv`
   - Your observations

5. **Discuss** and plan Iteration 1 together

---

## 📧 Questions During Iterations?

**For each iteration, share:**
1. Configuration changes made
2. Resulting metrics
3. Specific questions or concerns
4. Screenshots of interesting/problematic topics

**I'll help with:**
- Diagnosing issues
- Suggesting parameter adjustments
- Interpreting metrics
- Validating topic quality

---

**Let's start with Iteration 0 and iterate from there!**

Last Updated: February 2026
