# Features Used During Validation

## Overview

During validation, your model uses **the same 8 features** that were used during training. These features are all available **before image matching**, which is why they can be used to predict hard queries.

### ⚠️ Important: No `num_inliers` During Validation

**You do NOT have `num_inliers` during validation** because:
- ❌ `num_inliers` is computed **during image matching** (SuperPoint + LightGlue)
- ✅ Validation happens **before image matching** (to predict which queries need matching)
- ✅ The whole point is to predict hard queries **before** doing expensive matching

**Why this matters:**
- The script is called `stage_1_extract_features_no_inliers.py` (note: "no_inliers")
- All 8 features are computed from VPR retrieval results only
- No image matching is required to compute these features
- This allows you to predict hard queries and save time by skipping matching for easy queries

## The 8 Features

### 1. **top1_distance** (Basic Feature)
- **Description**: Descriptor distance of the Top-1 retrieved image
- **Source**: VPR retrieval results
- **Meaning**: Lower distance = more similar to query

### 2. **peakiness** (Basic Feature)
- **Description**: Ratio of Top-1 to Top-2 descriptor distances
- **Formula**: `top1_distance / top2_distance`
- **Meaning**: Higher peakiness = Top-1 is more distinct from Top-2 (more confident)

### 3. **sue_score** (Basic Feature)
- **Description**: Spatial Uncertainty Estimate from top-K neighbors
- **Source**: Computed from geographic distribution of top-K retrieved images
- **Meaning**: Lower score = more spatially clustered (more confident)

### 4. **topk_distance_spread** (New Feature)
- **Description**: Variance of top-5 descriptor distances
- **Formula**: `np.var(top5_distances)`
- **Meaning**: Higher spread = more uncertainty in top-K results

### 5. **top1_top2_similarity** (New Feature)
- **Description**: Distance ratio of Top-2 to Top-1
- **Formula**: `top2_distance / top1_distance`
- **Meaning**: Lower ratio = Top-1 and Top-2 are more similar (ambiguous)

### 6. **top1_top3_ratio** (New Feature)
- **Description**: Distance ratio of Top-1 to Top-3
- **Formula**: `top1_distance / top3_distance`
- **Meaning**: Higher ratio = Top-1 is more distinct from Top-3

### 7. **top2_top3_ratio** (New Feature)
- **Description**: Distance ratio of Top-2 to Top-3
- **Formula**: `top2_distance / top3_distance`
- **Meaning**: Higher ratio = Top-2 is more distinct from Top-3

### 8. **geographic_clustering** (New Feature)
- **Description**: Average pairwise distance of top-K positions
- **Formula**: Average of all pairwise geographic distances between top-K retrieved images
- **Meaning**: Lower clustering = images are more spread out geographically (less confident)

## How Features Are Used in Validation

### Step 1: Load Validation Features
```python
val_features = load_feature_file(args.val_features)
X_val, y_val, _ = build_feature_matrix(val_features)
```

### Step 2: Scale Features
- Uses the **same scaler** that was fitted on training data
- This ensures validation features are scaled the same way as training features
```python
X_val_scaled = scaler.transform(X_val)  # scaler was fitted on X_train
```

### Step 3: Make Predictions
```python
y_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]  # Probability of being easy
```

### Step 4: Find Optimal Threshold
- Tests different thresholds (0.1 to 0.95) on validation set
- Selects threshold that maximizes F1-score (or recall, depending on method)
- This threshold is saved with the model

### Step 5: Evaluate Performance
```python
y_val_pred = (y_val_probs >= optimal_threshold).astype(int)
# Compute accuracy, F1, precision, recall
```

## Feature Matrix Structure

The feature matrix `X_val` has shape `(N, 8)` where:
- `N` = number of validation queries
- `8` = number of features

Each row represents one query with 8 feature values:
```
[top1_distance, peakiness, sue_score, topk_distance_spread, 
 top1_top2_similarity, top1_top3_ratio, top2_top3_ratio, geographic_clustering]
```

## Important Notes

1. **Same Features as Training**: Validation uses exactly the same 8 features as training
2. **Same Scaler**: Features are scaled using the scaler fitted on training data
3. **No Image Matching Required**: All features are computed from VPR retrieval results, before image matching
4. **No `num_inliers`**: `num_inliers` is NOT available because it requires image matching (which we're trying to avoid!)
5. **Feature Availability**: If new features are missing, the model falls back to 3 basic features (top1_distance, peakiness, sue_score)

## Why Not Use `num_inliers`?

**`num_inliers` would be the perfect feature** - it directly indicates how well the query matches the retrieved image. However:

| Aspect | `num_inliers` | Current 8 Features |
|--------|---------------|-------------------|
| **Availability** | ❌ Only after image matching | ✅ Available before matching |
| **Computation Cost** | ❌ Expensive (SuperPoint + LightGlue) | ✅ Free (from VPR results) |
| **Purpose** | ❌ Defeats the goal (we want to skip matching!) | ✅ Allows prediction before matching |
| **Use Case** | ❌ Can't use for prediction | ✅ Perfect for prediction |

**The Trade-off:**
- Using `num_inliers` would give perfect predictions, but requires running image matching on all queries first
- Using 8 retrieval features gives good predictions (92.5% accuracy) without any image matching
- This allows us to skip matching for ~75% of queries, saving significant time

## Code Reference

The features are defined in:
- **Extraction**: `extension_6_1/stage_1_extract_features_no_inliers.py`
- **Training/Validation**: `extension_6_1/stage_3_train_logreg_easy_queries.py`
  - Function: `build_feature_matrix()` (lines 40-80)
  - Function: `load_feature_file()` (lines 21-37)

## Validation Process Summary

```
1. Load validation features (.npz file)
   ↓
2. Extract 8 features for each query
   ↓
3. Scale features (using training scaler)
   ↓
4. Predict probabilities (using trained model)
   ↓
5. Find optimal threshold (maximize F1 on validation)
   ↓
6. Evaluate performance (accuracy, F1, precision, recall)
   ↓
7. Save optimal threshold with model
```

## Example Output

When you run validation, you'll see:
```
Training logistic regression on 1000 queries.
  Features: top1_distance, peakiness, sue_score, topk_distance_spread, 
            top1_top2_similarity, top1_top3_ratio, top2_top3_ratio, 
            geographic_clustering (8 features)
  Target: easy_score (1 = easy/correct, 0 = hard/wrong) - Top-1 correctness

Validation metrics (with optimal threshold 0.410):
  Accuracy: 92.5%
  F1-Score: 0.9234
  Precision: 0.9456
  Recall: 0.9012
  ROC-AUC: 0.9678
```

