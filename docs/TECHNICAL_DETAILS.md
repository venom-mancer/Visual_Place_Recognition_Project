# Technical Details - Implementation and Fixes

## Overview

This document covers implementation details, bug fixes, feature engineering, and technical decisions.

---

## Feature Engineering

### 8 Retrieval-Based Features

All features are available **before** image matching, making them suitable for adaptive re-ranking.

#### Basic Features (3)

1. **`top1_distance`**
   - Descriptor distance of Top-1 retrieved image
   - Higher = more uncertain
   - Source: `z_data['distances'][query_idx][0]`

2. **`peakiness`**
   - Ratio of Top-1 to Top-2 descriptor distances
   - Lower = more ambiguous (Top-1 and Top-2 are similar)
   - Formula: `dists[0] / (dists[1] + 1e-8)`
   - Source: `z_data['distances']`

3. **`sue_score`** (Spatial Uncertainty Estimate)
   - Spatial uncertainty from top-K neighbors
   - Higher = more spatially uncertain
   - **Fixed**: Normalized distances and adjusted slope to prevent numerical underflow
   - Formula: Weighted variance of top-K neighbor positions

#### Additional Features (5)

4. **`topk_distance_spread`**
   - Standard deviation of top-5 distances
   - Higher = more spread out (uncertain)
   - Formula: `np.std(dists[:5])`

5. **`top1_top2_similarity`**
   - Distance ratio (Top2/Top1)
   - Lower = Top-1 and Top-2 are very similar (ambiguous)
   - Formula: `dists[1] / (dists[0] + 1e-8)`

6. **`top1_top3_ratio`**
   - Distance ratio (Top1/Top3)
   - Lower = Top-1 is not much better than Top-3
   - Formula: `dists[0] / (dists[2] + 1e-8)`

7. **`top2_top3_ratio`**
   - Distance ratio (Top2/Top3)
   - Captures ambiguity in second-tier candidates
   - Formula: `dists[1] / (dists[2] + 1e-8)`

8. **`geographic_clustering`**
   - Average pairwise distance of top-K positions
   - Higher = candidates are spread out geographically (uncertain)
   - Formula: Mean of pairwise Euclidean distances between top-K UTM positions

---

## Bug Fixes

### Fix #1: sue_score Numerical Underflow

**Problem**: All `sue_score` values were zero (100% of queries).

**Root Cause**: Numerical underflow in weight computation
- Distances in L2 space (typically 0.5-2.0 range)
- Formula: `weight = e^((-1 * distance) * 350)`
- With distance=0.8 and slope=350: `e^(-280) ≈ 0` (underflow)
- Result: All weights become zero → variance = 0 → `sue_score = 0`

**Solution**:
1. **Normalize distances** to 0-1 range per query:
   ```python
   normalized_dists = (query_dists - dist_min) / (dist_max - dist_min)
   ```

2. **Adjust slope** for normalized distances:
   ```python
   adjusted_slope = slope / 50.0  # 350 / 50 = 7.0
   ```

3. **Use signed differences** (not absolute) as in original:
   ```python
   diff_lat = min(500, nn_poses[k, 0] - mean_pose[0])  # Signed!
   diff_lon = min(500, nn_poses[k, 1] - mean_pose[1])  # Signed!
   ```

**Result**: 
- ✅ All queries produce non-zero SUE values
- ✅ Weights are properly computed (non-zero)
- ✅ SUE values range from ~7 to ~600,000 (normalized during training)

**Files Modified**: `extension_6_1/stage_1_extract_features_no_inliers.py`

---

## Model Implementation

### Logistic Regression Configuration

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Model training
model = LogisticRegression(
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,
    random_state=42
)
model.fit(X_train_scaled, y_train)
```

### Threshold Selection

**Method**: Find threshold that maximizes F1-score on validation set

```python
def find_optimal_threshold(y_true, y_probs):
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_f1 = 0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
```

**Result**: Optimal threshold = 0.410 (learned from SF-XS validation set)

---

## Pipeline Implementation

### Stage 1: Feature Extraction

**Script**: `extension_6_1/stage_1_extract_features_no_inliers.py`

**Key Functions**:
- `compute_sue()`: Compute Spatial Uncertainty Estimate
- `get_list_distances_from_preds()`: Get geographic distances from predictions

**Output**: `.npz` file with labels and 8 features

### Stage 2: Feature I/O

**Script**: `extension_6_1/stage_2_feature_io.py`

**Key Functions**:
- `load_feature_file(path)`: Load .npz feature file
- Returns dictionary with all features and labels

### Stage 3: Model Training

**Script**: `extension_6_1/stage_3_train_logreg_easy_queries.py`

**Process**:
1. Load training and validation features
2. Build feature matrix (8 features)
3. Scale features (StandardScaler)
4. Train Logistic Regression
5. Find optimal threshold on validation set
6. Save model + threshold

**Output**: `.pkl` file with model and metadata

### Stage 4: Model Application

**Script**: `extension_6_1/stage_4_apply_logreg_easy_queries.py`

**Process**:
1. Load model and optimal threshold
2. Load test features
3. Scale features (using saved scaler)
4. Predict probabilities
5. Classify as easy/hard using optimal threshold
6. Save predictions and hard query indices

**Output**: `.npz` file with predictions, `hard_queries.txt` with indices

### Stage 5: Adaptive Image Matching

**Script**: `match_queries_preds_adaptive.py`

**Process**:
1. Load hard query indices from file
2. For each hard query: Run SuperPoint + LightGlue matching
3. For each easy query: Create empty .torch file (for compatibility)
4. Save inlier counts to .torch files

**Output**: Directory with .torch files (one per query)

### Stage 6: Evaluation

**Script**: `extension_6_1/stage_5_adaptive_reranking_eval.py`

**Process**:
1. Load predictions and inlier results
2. For easy queries: Use retrieval-only ranking
3. For hard queries: Re-rank by inlier count
4. Compute Recall@N metrics

**Output**: Prints Recall@1, R@5, R@10, R@20

---

## Data Flow

```
VPR Evaluation
    ↓
z_data.torch (descriptors, distances, poses)
    ↓
Feature Extraction (Stage 1)
    ↓
features.npz (labels + 8 features)
    ↓
Model Training (Stage 3)
    ↓
model.pkl (model + threshold)
    ↓
Model Application (Stage 4)
    ↓
hard_queries.txt (list of hard query indices)
    ↓
Adaptive Image Matching (Stage 5)
    ↓
inliers/ (directory with .torch files)
    ↓
Evaluation (Stage 6)
    ↓
Recall@N metrics
```

---

## Key Design Decisions

### 1. Predict "Easy" Queries (Not "Hard")

**Decision**: Predict probability of being "easy" (Top-1 correct)

**Rationale**:
- More natural: "Is this query easy?" vs "What's the wrong score?"
- Directly actionable: Skip re-ranking if easy
- Better alignment with the goal

### 2. Use Logistic Regression (Not Regressor)

**Decision**: Use classifier for binary classification

**Rationale**:
- Right tool for the right job
- Better interpretability (probability of being easy)
- Higher accuracy (92.5% vs 86.7% with regressor)

### 3. Learn Threshold from Validation Set

**Decision**: Find optimal threshold on validation set (not fixed 0.5)

**Rationale**:
- Prevents overfitting (threshold not seen during training)
- Better generalization to test data
- Data-driven and optimal

### 4. Use 8 Features (Not 3)

**Decision**: Add 5 new features to original 3

**Rationale**:
- Fixed `sue_score` (was broken)
- Capture different aspects of uncertainty
- Improve model accuracy

### 5. Class Balancing

**Decision**: Use `class_weight='balanced'` in Logistic Regression

**Rationale**:
- Handle class imbalance (more easy queries than hard)
- Model learns to handle both classes equally
- Better hard query detection

---

## Performance Optimizations

### 1. Feature Extraction
- Vectorized operations where possible
- Efficient NumPy array operations
- Compressed .npz file format

### 2. Model Inference
- Batch processing for feature scaling
- Efficient probability computation
- Fast threshold application

### 3. Image Matching
- Only process hard queries (saves 74.6% time)
- Parallel processing with CUDA
- Efficient .torch file I/O

---

## File Formats

### Feature Files (.npz)
- Compressed NumPy archive
- Contains: `labels`, `top1_distance`, `peakiness`, `sue_score`, `topk_distance_spread`, `top1_top2_similarity`, `top1_top3_ratio`, `top2_top3_ratio`, `geographic_clustering`

### Model Files (.pkl)
- Pickle format (joblib)
- Contains: `model`, `scaler`, `optimal_threshold`, `feature_names`

### Inlier Files (.torch)
- PyTorch format
- Contains: `num_inliers`, `matches`, `keypoints` (for each top-K prediction)

---

*See [Pipeline Guide](PIPELINE_GUIDE.md) for step-by-step instructions.*

