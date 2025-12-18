# SVOX Test - Execution Plan

## Dataset Information
- **Location**: `data/svox/images/test/`
- **Gallery**: 17,166 images
- **Queries**: 14,278 images
- **Status**: ⏳ In Progress

---

## Pipeline Steps

### Step 1: VPR Evaluation ⏳ (Running in background)
```bash
python VPR-methods-evaluation/main.py \
  --num_workers 4 \
  --batch_size 32 \
  --log_dir log_svox_test \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --image_size 512 512 \
  --database_folder data/svox/images/test/gallery \
  --queries_folder data/svox/images/test/queries \
  --num_preds_to_save 20 \
  --recall_values 1 5 10 20 \
  --save_for_uncertainty \
  --device cuda
```

**Expected Output**: 
- `log_svox_test/[timestamp]/preds/` (14,278 .txt and .jpg files)
- `log_svox_test/[timestamp]/z_data.torch`

**Expected Time**: ~2-3 hours (14,278 queries)

### Step 2: Extract Features (After VPR completes)
```bash
python extract_svox_features.py \
  --preds-dir log_svox_test/[timestamp]/preds \
  --z-data-path log_svox_test/[timestamp]/z_data.torch \
  --output-path features_svox_test_improved.npz \
  --positive-dist-threshold 25
```

**Expected Output**: `features_svox_test_improved.npz` (8 features)

### Step 3: Apply Model (LogReg Easy)
```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path features_svox_test_improved.npz \
  --output-path logreg_easy_svox_test.npz \
  --hard-queries-output hard_queries_svox_test_logreg_easy.txt
```

**Expected Output**: 
- `logreg_easy_svox_test.npz`
- `hard_queries_svox_test_logreg_easy.txt`

### Step 4: Image Matching (Adaptive - Only Hard Queries)
```bash
python match_queries_preds_adaptive.py \
  --preds-dir log_svox_test/[timestamp]/preds \
  --hard-queries-list hard_queries_svox_test_logreg_easy.txt \
  --out-dir log_svox_test/[timestamp]/preds_superpoint-lg_logreg_easy \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20
```

**Expected Output**: `log_svox_test/[timestamp]/preds_superpoint-lg_logreg_easy/` (14,278 .torch files)

### Step 5: Evaluate Adaptive Re-ranking
```bash
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir log_svox_test/[timestamp]/preds \
  --inliers-dir log_svox_test/[timestamp]/preds_superpoint-lg_logreg_easy \
  --logreg-output logreg_easy_svox_test.npz \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20
```

**Expected Output**: Recall@N metrics

### Step 6: Evaluate Baseline and Full Re-ranking (For Comparison)
```bash
# Baseline
python evaluate_baseline.py \
  --preds-dir log_svox_test/[timestamp]/preds \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20

# Full Re-ranking (need to run image matching for all queries first)
python match_queries_preds.py \
  --preds-dir log_svox_test/[timestamp]/preds \
  --out-dir log_svox_test/[timestamp]/preds_superpoint-lg \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20

python reranking.py \
  --preds-dir log_svox_test/[timestamp]/preds \
  --inliers-dir log_svox_test/[timestamp]/preds_superpoint-lg \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20
```

---

## Current Status

- ✅ **Step 1**: VPR Evaluation - Running in background
- ⏳ **Step 2**: Extract Features - Waiting for Step 1
- ⏳ **Step 3**: Apply Model - Waiting for Step 2
- ⏳ **Step 4**: Image Matching - Waiting for Step 3
- ⏳ **Step 5**: Evaluate - Waiting for Step 4
- ⏳ **Step 6**: Baseline/Full Comparison - Waiting for Step 5

---

## Expected Time

- **VPR Evaluation**: ~2-3 hours (14,278 queries)
- **Feature Extraction**: ~5-10 minutes
- **Model Application**: ~1 second
- **Image Matching (adaptive)**: ~3-4 hours (assuming ~25% hard queries = ~3,570 queries × ~9.5 sec)
- **Image Matching (full)**: ~14-15 hours (14,278 queries × ~9.5 sec)
- **Evaluation**: ~2-3 minutes

**Total**: ~20-22 hours (mostly image matching)

---

*Status: Step 1 running in background*

