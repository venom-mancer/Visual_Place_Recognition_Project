# Google Colab Execution Guide

## Overview

This guide explains how to run the adaptive re-ranking pipeline in **Google Colab** instead of your local PC.

**Benefits of Colab:**
- ✅ Free GPU access (T4 GPU) - **No Colab Pro required!**
- ✅ No local setup required
- ✅ Easy to share and collaborate
- ✅ Automatic environment management

**Free Colab vs Colab Pro:**

| Feature | Free Colab | Colab Pro |
|---------|------------|-----------|
| GPU Access | ✅ Yes (T4) | ✅ Yes (T4/V100) |
| Session Timeout | ⚠️ ~12 hours | ✅ ~24 hours |
| GPU Hours/Day | ⚠️ ~12 hours | ✅ ~50+ hours |
| Background Execution | ❌ No | ✅ Yes |
| Faster GPUs | ❌ No | ✅ V100 available |
| Priority Access | ❌ No | ✅ Yes |

**Limitations (Free Colab):**
- ⚠️ Session timeout (~12 hours of inactivity)
- ⚠️ Limited GPU hours (~12 hours/day)
- ⚠️ Files deleted when session ends (unless saved to Drive)
- ⚠️ May disconnect during very long operations

---

## Quick Start

### Option 1: Use the Colab Notebook

1. **Open the notebook**: `COLAB_ADAPTIVE_RERANKING_PIPELINE.ipynb`
2. **Upload to Colab**: File → Upload notebook
3. **Enable GPU**: Runtime → Change runtime type → GPU (T4)
4. **Run cells**: Execute cells sequentially

### Option 2: Manual Setup

Follow the steps below manually in Colab.

---

## Step-by-Step Setup

### 1. Mount Google Drive (Optional)

```python
from google.colab import drive
drive.mount('/content/drive')
```

**Why**: Save results to Drive so they persist after session ends.

### 2. Clone Repository

```bash
!git clone --recursive https://github.com/FarInHeight/Visual-Place-Recognition-Project.git
%cd Visual-Place-Recognition-Project
```

### 3. Install Dependencies

```bash
# Install image matching models
%cd image-matching-models
!pip install -e .[all]
%cd ..

# Install other dependencies
!pip install faiss-cpu scikit-learn joblib matplotlib scipy tqdm
```

### 4. Download Datasets

**Option A**: Use download script
```bash
!python download_datasets.py
```

**Option B**: Download manually from Google Drive and upload to Colab
- Upload datasets to `/content/Visual-Place-Recognition-Project/data/`

---

## Running the Pipeline

### Step 1: VPR Evaluation

```bash
!python VPR-methods-evaluation/main.py \
  --num_workers 4 \
  --batch_size 32 \
  --log_dir log_sf_xs_test \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/database \
  --queries_folder data/sf_xs/test/queries \
  --num_preds_to_save 20 \
  --recall_values 1 5 10 20 \
  --save_for_uncertainty \
  --device cuda
```

**Time**: ~5-10 minutes per dataset

### Step 2: Extract Features

```bash
!python -m extension_6_1.stage_1_extract_features_no_inliers \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --z-data-path logs/log_sf_xs_test/[timestamp]/z_data.torch \
  --output-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --positive-dist-threshold 25
```

**Time**: ~1-2 minutes

### Step 3: Train Model

```bash
!python -m extension_6_1.stage_3_train_logreg_easy_queries \
  --train-features data/features_and_predictions/features_svox_train_improved.npz \
  --val-features data/features_and_predictions/features_sf_xs_val_improved.npz \
  --output-model logreg_easy_queries_optimal_C_tuned.pkl \
  --threshold-method f1
```

**Time**: ~1-2 minutes

### Step 4: Apply Model

```bash
!python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --output-path data/features_and_predictions/logreg_sf_xs_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_sf_xs_test.txt \
  --calibrate-threshold
```

**Time**: ~1 minute

### Step 5: Full Re-ranking (Ground Truth)

```bash
!python match_queries_preds.py \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20 \
  --out-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg
```

**Time**: ~2-4 hours (depends on number of queries)

### Step 6: Adaptive Image Matching

```bash
!python match_queries_preds_adaptive.py \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --hard-queries-list data/features_and_predictions/hard_queries_sf_xs_test.txt \
  --out-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20
```

**Time**: ~30-60 minutes (only hard queries)

### Step 7: Threshold Analysis

```bash
!python adaptive_reranking_threshold_analysis.py \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --datasets sf_xs_test tokyo_xs_test \
  --feature-paths \
    data/features_and_predictions/features_sf_xs_test_improved.npz \
    data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --preds-dirs \
    logs/log_sf_xs_test/[timestamp]/preds \
    log_tokyo_xs_test/[timestamp]/preds \
  --inliers-dirs \
    logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg \
    log_tokyo_xs_test/[timestamp]/preds_superpoint-lg \
  --output-dir output_stages/threshold_analysis_comprehensive \
  --threshold-range 0.1 0.99 \
  --threshold-step 0.05 \
  --num-preds 20 \
  --positive-dist-threshold 25
```

**Time**: ~10-20 minutes

### Step 8: Serialize to MATLAB

```bash
!python serialize_results_to_matlab.py \
  --results-dir output_stages/threshold_analysis_comprehensive \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --output-dir output_stages/matlab_files
```

**Time**: ~1 minute

---

## Saving Results

### Option 1: Download Files

```python
from google.colab import files

# Download individual files
files.download('output_stages/threshold_analysis_comprehensive/recall_at_1_vs_threshold.png')
files.download('output_stages/matlab_files/threshold_analysis_results.mat')
```

### Option 2: Save to Google Drive

```python
# Copy to Drive
!cp -r output_stages/threshold_analysis_comprehensive /content/drive/MyDrive/VPR_Results/
!cp -r output_stages/matlab_files /content/drive/MyDrive/VPR_Results/
```

### Option 3: Create Zip Archive

```python
import zipfile
import os

with zipfile.ZipFile('results.zip', 'w') as zipf:
    for root, dirs, files in os.walk('output_stages'):
        for file in files:
            if file.endswith(('.png', '.mat', '.md')):
                zipf.write(os.path.join(root, file))

from google.colab import files
files.download('results.zip')
```

---

## Time Estimates

| Step | Time (Colab GPU) | Notes |
|------|------------------|-------|
| VPR Evaluation | 5-10 min/dataset | Fast with GPU |
| Feature Extraction | 1-2 min/dataset | Very fast |
| Model Training | 1-2 min | Fast |
| Model Application | 1 min/dataset | Very fast |
| **Full Re-ranking** | **2-4 hours/dataset** | ⚠️ **Longest step** |
| Adaptive Matching | 30-60 min/dataset | Depends on hard queries |
| Threshold Analysis | 10-20 min | Fast |
| MATLAB Serialization | 1 min | Very fast |

**Total for one dataset**: ~3-5 hours (mostly full re-ranking)

---

## Tips for Colab

### 1. Enable GPU (Free Tier Works!)
- Runtime → Change runtime type → GPU (T4)
- **Free Colab includes T4 GPU - no Pro needed!**
- Check GPU: `!nvidia-smi`

### 2. Monitor Progress
- Use `tqdm` progress bars (already in code)
- Check Colab output for progress
- Keep browser tab active to prevent timeout

### 3. Handle Timeouts (Free Colab Strategy)
- **Full re-ranking takes 2-4 hours** - may timeout on free tier
- **Solutions for Free Colab**:
  - **Option A**: Run in smaller batches (split queries)
  - **Option B**: Save checkpoints to Drive, resume later
  - **Option C**: Run full re-ranking overnight (keep tab active)
  - **Option D**: Use hybrid approach (Colab for VPR, local for matching)

### 4. Save Checkpoints
- Save intermediate results to Drive
- Resume from checkpoints if session ends
- **Critical for free tier!**

### 5. Optimize for Colab
- Use smaller batch sizes if memory issues
- Reduce `num_workers` if needed
- Use `num-preds 20` (not 100) to save time
- **Keep browser tab active** during long operations

---

## Alternative: Hybrid Approach

**Run computationally expensive steps in Colab, others locally:**

1. **Colab**: VPR evaluation, full re-ranking
2. **Download**: Results from Colab
3. **Local**: Feature extraction, model training, threshold analysis
4. **Upload**: Results back to Colab for visualization

**Benefits**:
- ✅ Use Colab GPU for expensive operations
- ✅ Keep analysis local (faster iteration)
- ✅ Save Colab GPU hours

---

## Troubleshooting

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or `num-preds`

### Issue: "Session timeout"
**Solution**: 
- **Free Colab**: Keep browser tab active, save checkpoints to Drive
- **If timeout occurs**: Resume from saved checkpoints
- **Alternative**: Use Colab Pro for longer sessions (24h vs 12h)

### Issue: "Files not found"
**Solution**: Check paths, use absolute paths if needed

### Issue: "Module not found"
**Solution**: Re-run installation cells

---

## Example: Complete Workflow

```python
# 1. Setup
!git clone --recursive https://github.com/FarInHeight/Visual-Place-Recognition-Project.git
%cd Visual-Place-Recognition-Project
!pip install -e image-matching-models/.[all]
!pip install faiss-cpu scikit-learn joblib matplotlib scipy

# 2. VPR Evaluation
!python VPR-methods-evaluation/main.py --device cuda [args...]

# 3. Extract Features
!python -m extension_6_1.stage_1_extract_features_no_inliers [args...]

# 4. Train Model
!python -m extension_6_1.stage_3_train_logreg_easy_queries [args...]

# 5. Apply Model
!python -m extension_6_1.stage_4_apply_logreg_easy_queries [args...]

# 6. Full Re-ranking (long!)
!python match_queries_preds.py --device cuda [args...]

# 7. Adaptive Matching
!python match_queries_preds_adaptive.py --device cuda [args...]

# 8. Threshold Analysis
!python adaptive_reranking_threshold_analysis.py [args...]

# 9. Download Results
from google.colab import files
files.download('results.zip')
```

---

*See `COLAB_ADAPTIVE_RERANKING_PIPELINE.ipynb` for the complete notebook.*

