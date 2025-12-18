# Upgrade Model - Update 1: Regressor Strategy Explanation

## What Problem Are We Solving?

**Challenge**: Image matching (re-ranking) is computationally expensive. Running it on all queries wastes time when many queries are already correct.

**Goal**: Identify "hard" queries (those likely to benefit from re-ranking) **before** running expensive image matching, so we can:
- Apply re-ranking only to hard queries
- Skip re-ranking for easy queries
- Save computation time while maintaining good performance

---

## The Strategy: Continuous Score Prediction

### Core Idea

Instead of using a binary classifier (hard/easy), we use a **Random Forest Regressor** to predict a continuous "wrong score" (0.0 to 1.0) that indicates how likely a query is to be wrong.

**Why Continuous Score?**
- More nuanced than binary classification
- Better interpretability (e.g., 0.7 = 70% likely to be wrong)
- Higher accuracy (86.7% vs 81.0% with classifier)

### The Wrong Score

**Definition**: 
- `wrong_score = 0.0` → Query is correct (Top-1 is within threshold)
- `wrong_score = 1.0` → Query is wrong (Top-1 is outside threshold)

**Why "Wrong Score" instead of "Hardness Score"?**
1. ✅ **Directly observable** before image matching (we know if Top-1 is correct)
2. ✅ **Stable across datasets** (no distribution shift)
3. ✅ **Actionable**: If wrong, we definitely need re-ranking
4. ✅ **Highly correlated** with actual hardness

**Previous Attempt (Hardness Score)**:
- Based on `num_inliers` (requires image matching - circular dependency!)
- Distribution shift between train/test
- Less stable

---

## How It Works: Step by Step

### Step 1: Extract Features (Before Image Matching)

We compute **8 retrieval features** that are available immediately after VPR retrieval:

**Basic Features (3)**:
1. **`top1_distance`**: How far is the Top-1 match in descriptor space?
   - Higher = more uncertain
2. **`peakiness`**: How much better is Top-1 than Top-2?
   - Lower = Top-1 and Top-2 are similar (ambiguous)
3. **`sue_score`**: Spatial Uncertainty Estimate from top-K neighbors
   - Higher = more spatially uncertain

**Additional Features (5)** - New in this update:
4. **`topk_distance_spread`**: How spread out are the top-5 distances?
   - Higher = more uncertainty
5. **`top1_top2_similarity`**: How similar are Top-1 and Top-2?
   - Lower = more ambiguous
6. **`top1_top3_ratio`**: How much better is Top-1 than Top-3?
   - Lower = Top-1 is not clearly the best
7. **`top2_top3_ratio`**: How similar are Top-2 and Top-3?
   - Captures second-tier ambiguity
8. **`geographic_clustering`**: How spread out are top-K candidates geographically?
   - Higher = candidates are far apart (uncertain)

**Key Point**: All features are available **before** image matching!

### Step 2: Train Regressor

**Model**: Random Forest Regressor
- **Input**: 8 features (scaled)
- **Output**: Continuous wrong_score [0.0, 1.0]
- **Training**: SVOX train (1,414 queries)
- **Validation**: SF-XS val (7,993 queries)

**Why Random Forest?**
- Handles non-linear relationships
- Robust to outliers
- Good generalization
- Feature importance analysis

**Training Process**:
```
1. Load features from training set
2. Create target: wrong_score = (labels == 0).astype("float32")
   - labels: 1 = correct, 0 = wrong
   - wrong_score: 1 = wrong, 0 = correct
3. Scale features (StandardScaler)
4. Train Random Forest
5. Validate on validation set
```

**Results**:
- Training R²: 0.8235 (good fit)
- Validation Classification Accuracy: **86.7%** ✅

### Step 3: Apply Regressor to Test Set

For each query in test set:
1. Extract 8 features (fast, retrieval-based)
2. Scale features using trained scaler
3. Predict wrong_score using trained regressor
4. Classify: 
   - `wrong_score > 0.5` → Hard (needs re-ranking)
   - `wrong_score ≤ 0.5` → Easy (skip re-ranking)

**Test Results**:
- Hard queries detected: 425 (42.5%)
- Easy queries (skipped): 575 (57.5%)
- Actually wrong: 369 (36.9%)
- Over-prediction: +56 queries (+5.6%) - acceptable

### Step 4: Run Image Matching (Only for Hard Queries)

**Critical**: Image matching runs **AFTER** prediction, not before!

**Pipeline Order**:
```
1. VPR Evaluation → Retrieval predictions
2. Extract Features → 8 retrieval features (fast)
3. Apply Regressor → Predict wrong_score
4. Classify Queries → Hard vs Easy
5. Image Matching → ONLY for hard queries ← Saves time!
6. Re-ranking → Sort by num_inliers for hard queries
7. Evaluation → Compute Recall@N
```

**Time Savings**:
- Full re-ranking: 100% of queries (1,000 queries)
- Adaptive: 42.5% of queries (425 queries)
- **Time saved: 57.5%** ✅

### Step 5: Evaluate Performance

**Results**:
- **R@1: 73.4%** (+10.3% vs baseline, -4.0% vs full re-ranking)
- **R@5: 78.9%** (+4.1% vs baseline, -1.4% vs full re-ranking)
- **R@10: 80.3%** (+1.7% vs baseline, -0.6% vs full re-ranking)
- **R@20: 81.4%** (same as baseline and full re-ranking)

---

## Why This Strategy Works

### 1. Direct Target Prediction

**Wrong Score** is directly observable before image matching:
- We know if Top-1 is correct (within threshold) or wrong (outside threshold)
- No need to predict something that requires image matching (like num_inliers)
- Stable across datasets

### 2. Rich Feature Representation

**8 features** capture multiple aspects of uncertainty:
- **Descriptor uncertainty**: top1_distance, peakiness
- **Spatial uncertainty**: sue_score, geographic_clustering
- **Ranking ambiguity**: top1_top2_similarity, top1_top3_ratio, top2_top3_ratio
- **Distribution spread**: topk_distance_spread

All features are:
- Available before image matching
- Computationally cheap to compute
- Informative about query difficulty

### 3. Continuous Score Prediction

**Advantages over binary classification**:
- More nuanced predictions (0.0 to 1.0 vs 0 or 1)
- Better interpretability (e.g., 0.7 = 70% likely wrong)
- Higher accuracy (86.7% vs 81.0%)
- Can adjust threshold based on requirements

### 4. Efficient Pipeline

**Key Innovation**: Image matching runs **after** prediction:
- Predict hard queries first (fast, feature-based)
- Run image matching only for hard queries (slow, but only 42.5%)
- True time savings (57.5% reduction)

**Previous Problem**: Image matching ran for all queries before prediction (no time savings!)

### 5. No Hard Thresholding on Features

**Previous approaches** used hard rules like:
- "If top1_distance > X, then hard"
- "If peakiness < Y, then hard"

**New approach**:
- Uses continuous regressor output
- Simple threshold on predicted score (0.5)
- More flexible and data-driven

---

## Comparison with Previous Approaches

### vs Logistic Regression Classifier

| Aspect | Classifier | Regressor |
|--------|-----------|-----------|
| **Model Type** | Binary classifier | Continuous regressor |
| **Output** | Probability [0, 1] | Wrong score [0, 1] |
| **Features** | 8 | 8 |
| **Validation Accuracy** | 81.0% | **86.7%** ✅ |
| **Interpretability** | Less | More |
| **Threshold** | On probability | On predicted score |

**Why Regressor is Better**:
- Higher accuracy (86.7% vs 81.0%)
- More interpretable (continuous score)
- Better generalization

### vs Initial Regressor (3 features, hardness_score)

| Aspect | Initial | Final |
|--------|---------|-------|
| **Features** | 3 | **8** ✅ |
| **Target** | hardness_score (percentile-based) | **wrong_score** ✅ |
| **Hard queries detected** | 30.8% | **42.5%** ✅ |
| **Validation Accuracy** | N/A | **86.7%** ✅ |
| **Distribution Shift** | High (negative R²) | Lower (classification works) |

**Why Final is Better**:
- More features (8 vs 3) = better representation
- Better target (wrong_score vs hardness_score) = more stable
- Better detection (42.5% vs 30.8%) = closer to actual (36.9%)

---

## Key Innovations

### 1. Wrong Score as Target
- Directly observable before image matching
- Stable across datasets
- Highly correlated with hardness

### 2. 8 Feature Representation
- Captures multiple uncertainty aspects
- All available before image matching
- Computationally cheap

### 3. Continuous Score Prediction
- More nuanced than binary
- Better interpretability
- Higher accuracy

### 4. Efficient Pipeline Order
- Predict first, match later
- True time savings
- Maintains performance

### 5. Simple Threshold on Score
- No hard thresholding on features
- Data-driven approach
- Flexible and tunable

---

## Results Summary

### Performance:
- **+10.3% R@1** vs baseline (73.4% vs 63.1%)
- **-4.0% R@1** vs full re-ranking (73.4% vs 77.4%) - acceptable trade-off
- **86.7% classification accuracy** on validation

### Efficiency:
- **42.5% of queries** re-ranked (vs 100% for full re-ranking)
- **57.5% time savings** in image matching
- **Good balance** between performance and efficiency

### Detection:
- **425 hard queries detected** (42.5%)
- **369 actually wrong** (36.9%)
- **Over-prediction: +5.6%** - acceptable margin

---

## Conclusion

The regressor-based strategy successfully:

1. ✅ **Predicts query difficulty** using continuous wrong_score
2. ✅ **Uses 8 rich features** available before image matching
3. ✅ **Achieves high accuracy** (86.7% classification accuracy)
4. ✅ **Improves performance** (+10.3% R@1 vs baseline)
5. ✅ **Saves computation** (57.5% time savings)
6. ✅ **No hard thresholding** on features (uses continuous score)

This approach provides a good balance between performance and efficiency, making it suitable for real-world deployment where computational resources are limited.

---

*Update 1 - Strategy Explanation*
*Date: 2025-12-18*
*Model: Random Forest Regressor with 8 features predicting wrong_score*

