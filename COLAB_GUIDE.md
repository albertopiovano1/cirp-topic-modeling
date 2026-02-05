# Running CIRP Analysis on Google Colab

Quick guide for running the BERTopic pipeline on Google Colab with GPU acceleration.

---

## 🚀 Quick Start (5 minutes)

### Step 1: Open Colab Notebook

Create a new notebook at [colab.research.google.com](https://colab.research.google.com)

### Step 2: Enable GPU (Recommended)

```
Runtime → Change runtime type → Hardware accelerator: GPU → Save
```

**Why GPU?** Speeds up embedding generation (~3x faster for large corpora)

### Step 3: Clone Repository

```python
!git clone https://github.com/albertopiovano1/cirp-topic-modeling.git
%cd cirp-topic-modeling
```

### Step 4: Install Dependencies

```python
!pip install -q bertopic sentence-transformers umap-learn hdbscan gensim spacy tqdm
!python -m spacy download en_core_web_sm
```

**Note:** `-q` flag suppresses verbose output. Remove for troubleshooting.

### Step 5: Upload Dataset

**Option A: From Google Drive**
```python
from google.colab import drive
drive.mount('/content/drive')

# Copy dataset
!cp "/content/drive/MyDrive/CIRP_researchonly.csv" ./data/
```

**Option B: Direct Upload**
```python
from google.colab import files
uploaded = files.upload()  # Select CIRP_researchonly.csv

# Move to data directory
!mv CIRP_researchonly.csv ./data/
```

### Step 6: Run Pipeline

```python
!python cirp_bertopic_v5_0.py
```

**Expected Runtime:**
- Small corpus (~1K docs): 5-8 minutes
- Medium corpus (~5K docs): 15-20 minutes
- Large corpus (~10K docs): 30-40 minutes

### Step 7: Download Results

```python
# Compress results
!zip -r results.zip results/ figures/

# Download
from google.colab import files
files.download('results.zip')
```

---

## 📊 Monitoring Progress

The pipeline prints structured progress:

```
================================================================================
DATA VALIDATION & QUALITY REPORT
================================================================================
...
✅ Dataset quality: EXCELLENT - Ready for analysis

================================================================================
TEXT PREPROCESSING
================================================================================
📄 Processing 4,729 documents...
  [1/4] Basic cleaning...
  [2/4] Filtering documents (min 10 words)...
    ✅ Valid documents: 4,687 / 4,729
...

================================================================================
MODEL TRAINING
================================================================================
📊 Training on 4,687 documents...
📈 Initial Results:
   • Topics discovered: 23
   • Outliers: 287 (6.1%)
...

================================================================================
QUALITY METRICS CALCULATION
================================================================================
🟣 Topic Diversity: 0.8423
🟢 Coherence C_v: 0.6891
🟠 Topic Stability: 0.7912
...
```

---

## ⚙️ Colab-Specific Optimizations

### Memory Management

If you encounter memory errors:

```python
# In cirp_bertopic_v5_0.py, modify embedding generation:

# Original:
embeddings = model.embedding_model.encode(
    text_embed,
    show_progress_bar=True,
    normalize_embeddings=True
)

# Optimized for Colab:
embeddings = model.embedding_model.encode(
    text_embed,
    batch_size=32,  # Reduce from default 64
    show_progress_bar=True,
    normalize_embeddings=True
)
```

### Session Management

Colab sessions timeout after ~12 hours of inactivity. For long runs:

```python
# Keep session alive
import time
from IPython.display import clear_output

def keep_alive():
    while True:
        clear_output(wait=True)
        print("Session active. Time:", time.strftime("%H:%M:%S"))
        time.sleep(300)  # Ping every 5 minutes

# Run in background (optional)
import threading
thread = threading.Thread(target=keep_alive)
thread.daemon = True
thread.start()
```

### GPU Utilization Check

```python
# Check if GPU is being used
import torch
print("GPU Available:", torch.cuda.is_available())
print("GPU Name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "None")

# Monitor GPU memory
!nvidia-smi
```

---

## 🔧 Customizing Configuration in Colab

### Modify Parameters Before Running

```python
# Option 1: Edit directly in the file
!nano cirp_bertopic_v5_0.py  # Or use Colab's built-in editor

# Option 2: Override via environment variables (if implemented)
import os
os.environ['BERTOPIC_SEED'] = '42'
os.environ['TIME_CUTOFF_YEAR'] = '2023'
```

### Quick Parameter Changes

Create a small wrapper script:

```python
# colab_config.py
import sys
sys.path.insert(0, '/content/cirp-topic-modeling')

from cirp_bertopic_v5_0 import Config

# Override parameters
Config.SEED = 42
Config.TIME_CUTOFF_YEAR = 2023
Config.HDBSCAN_MIN_CLUSTER_SIZE = 15
Config.TOP_N_PAPERS_PER_TOPIC = 10

# Run pipeline
from cirp_bertopic_v5_0 import main
main()
```

Then:
```python
!python colab_config.py
```

---

## 📁 Working with Google Drive Integration

### Full Integration Workflow

```python
from google.colab import drive
drive.mount('/content/drive')

# Setup project in Drive
project_path = "/content/drive/MyDrive/CIRP_Analysis"
!mkdir -p {project_path}

# Clone repo to Drive
%cd /content/drive/MyDrive
!git clone https://github.com/albertopiovano1/cirp-topic-modeling.git CIRP_Analysis
%cd CIRP_Analysis

# Dataset already in Drive
# No need to upload again!

# Run pipeline
!python cirp_bertopic_v5_0.py

# Results automatically saved to Drive
# Access from any device at: MyDrive/CIRP_Analysis/results/
```

**Advantages:**
- ✅ Results persist across sessions
- ✅ No need to re-upload data
- ✅ Easy sharing with collaborators

---

## 🎨 Viewing Visualizations in Colab

### Option 1: Direct Display

```python
from IPython.display import IFrame

# Display interactive HTML
IFrame(src='./figures/Viz_01_Intertopic_Distance_Map.html', 
       width=900, height=700)
```

### Option 2: Download and Open Locally

```python
# Download specific visualization
from google.colab import files
files.download('./figures/Viz_01_Intertopic_Distance_Map.html')
```

### Option 3: Host on Colab (Temporary)

```python
# Serve figures via HTTP
!pip install -q flask-ngrok

# create simple server
# (implementation details in separate notebook)
```

---

## 🐛 Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'bertopic'"

**Solution:**
```python
!pip install bertopic sentence-transformers umap-learn hdbscan
```

### Issue: Out of Memory

**Solution:**
```python
# Clear variables
%reset -f

# Use smaller batch size
# Edit cirp_bertopic_v5_0.py → embeddings = ... batch_size=16
```

### Issue: Runtime Disconnected

**Solution:**
- Colab free tier has usage limits
- Consider Colab Pro for longer runtimes
- Save intermediate results:

```python
# Add checkpointing to script
import pickle

# After embedding generation:
with open('embeddings_checkpoint.pkl', 'wb') as f:
    pickle.dump(embeddings, f)

# Resume if interrupted:
if os.path.exists('embeddings_checkpoint.pkl'):
    with open('embeddings_checkpoint.pkl', 'rb') as f:
        embeddings = pickle.load(f)
else:
    embeddings = model.encode(...)
```

### Issue: SpaCy Model Not Found

**Solution:**
```python
!python -m spacy download en_core_web_sm
!python -m spacy validate
```

---

## 📊 Expected Output Structure

After successful run, you should see:

```
cirp-topic-modeling/
├── data/
│   └── CIRP_researchonly.csv
├── results/
│   ├── 00_Analysis_Report.txt
│   ├── 01_Full_Document_Assignments.csv
│   ├── 02_Top_Papers_Per_Topic.csv
│   ├── 03_Topic_Metadata_Keywords.csv
│   └── 04_Summary_Statistics.csv
├── figures/
│   ├── Viz_01_Intertopic_Distance_Map.html
│   ├── Viz_02_Topic_Word_Scores.html
│   ├── Viz_03_Hierarchical_Clustering.html
│   ├── Viz_04_Topic_Similarity_Heatmap.html
│   └── Viz_05_Topics_Over_Time_to_2024.html
└── cirp_bertopic_v5_0.py
```

---

## 💡 Pro Tips

### 1. Version Control in Colab

```python
# Commit results to your GitHub fork
!git config user.email "your-email@example.com"
!git config user.name "Your Name"

!git add results/ figures/
!git commit -m "Analysis results - $(date +'%Y-%m-%d')"
!git push origin main
```

### 2. Parameter Sweep

Run multiple configurations:

```python
seeds = [42, 76, 123]
cluster_sizes = [8, 10, 15]

for seed in seeds:
    for size in cluster_sizes:
        print(f"\n{'='*80}")
        print(f"Running: seed={seed}, cluster_size={size}")
        print(f"{'='*80}\n")
        
        # Modify config
        Config.SEED = seed
        Config.HDBSCAN_MIN_CLUSTER_SIZE = size
        
        # Run pipeline
        main()
        
        # Rename results
        !mv results results_s{seed}_c{size}
```

### 3. Automated Reporting

```python
# Email results when complete
!pip install -q yagmail

import yagmail
yag = yagmail.SMTP('your-email@gmail.com', 'app-password')

yag.send(
    to='your-email@gmail.com',
    subject='CIRP Analysis Complete',
    contents='Your topic modeling analysis has finished!',
    attachments=['results.zip']
)
```

---

## 📱 Mobile Workflow (Colab App)

You can run the entire pipeline from your phone:

1. Install **Colab mobile app** (iOS/Android)
2. Open notebook with code cells from this guide
3. Connect to Google Drive
4. Tap "Runtime" → "Run all"
5. Check back in 20-30 minutes
6. Download results to phone

**Use case:** Start analysis during commute, review results at office!

---

## 🔗 Quick Links

- **Colab Notebook Template:** [Coming Soon]
- **Sample Results:** [Coming Soon]
- **Video Tutorial:** [Coming Soon]

---

## ❓ Need Help?

**Colab-specific issues:**
- [Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [Stack Overflow - google-colaboratory tag](https://stackoverflow.com/questions/tagged/google-colaboratory)

**Pipeline-specific issues:**
- GitHub Issues: https://github.com/albertopiovano1/cirp-topic-modeling/issues
- Email: alberto.piovano@polito.it

---

**Last Updated:** February 2026
