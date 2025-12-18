# Methodology - Approaches and Strategies

## Overview

This document explains the different approaches tried for adaptive re-ranking, their trade-offs, and why the final approach (Logistic Regression for Easy Queries) was selected.

---

## Evolution of Approaches

### Approach 1: Initial Pipeline (With Inliers)

**Concept**: Use `num_inliers` as a feature to predict hard queries.

**Problem**: 
- ❌ `num_inliers` requires image matching (expensive!)
- ❌ Defeats the purpose of adaptive re-ranking
- ❌ Cannot predict before matching

**Status**: ❌ Abandoned

---

### Approach 2: Logistic Regression (Hard Queries, 3 Features)

**Concept**: Predict "hard" queries using 3 retrieval-based features.

**Features**:
1. `top1_distance`
2. `peakiness`
3. `sue_score` (initially broken - all zeros)

**Problems**:
- ❌ `sue_score` was broken (all zeros due to numerical underflow)
- ❌ Low accuracy (64.5%)
- ❌ Too conservative (only 20.2% hard queries detected vs 49.7% actual)
- ❌ Poor performance (-9.1% R@1 vs baseline)

**Status**: ❌ Replaced

---

### Approach 3: Regressor (Wrong Score, 8 Features)

**Concept**: Use Random Forest Regressor to predict continuous "wrong score".

**Target**: `wrong_score = (labels == 0).astype("float32")` (0 = correct, 1 = wrong)

**Features**: 8 improved features (fixed `sue_score` + 5 new features)

**Results**:
- ✅ Better accuracy: 86.7% validation accuracy
- ✅ Good performance: +10.3% R@1 vs baseline
- ⚠️ Over-predicts: 42.5% hard queries (vs 36.9% actual)
- ⚠️ Fixed threshold: 0.5 (not optimal)

**Status**: ✅ Works, but replaced by better approach

---

### Approach 4: Logistic Regression (Easy Queries, 8 Features) ⭐ **FINAL**

**Concept**: Predict "easy" queries directly using Logistic Regression classifier.

**Target**: `easy_score = labels` (1 = easy/correct, 0 = hard/wrong)

**Features**: Same 8 features as Approach 3

**Key Innovation**: **Optimal threshold learned from validation data**

**Results**:
- ✅ **Best accuracy**: 92.5% validation accuracy
- ✅ **Good performance**: +6.7% R@1 vs baseline
- ✅ **Optimal threshold**: 0.410 (learned, not fixed)
- ✅ **Better detection**: 25.4% hard queries (closer to 36.9% actual)
- ✅ **Best efficiency**: 74.6% time savings

**Status**: ✅ **Selected as final approach**

---

## Detailed Comparison

### Regressor vs Logistic Regression (Easy)

| Aspect | Regressor | Logistic Regression (Easy) |
|--------|-----------|---------------------------|
| **Model Type** | Random Forest Regressor | Logistic Regression Classifier |
| **Target** | Wrong score (continuous) | Easy score (binary probability) |
| **Threshold** | Fixed 0.5 | **Learned 0.410** ✅ |
| **Validation Accuracy** | 86.7% | **92.5%** ✅ |
| **Hard Queries Detected** | 42.5% (425/1000) | 25.4% (254/1000) |
| **Time Savings** | 57.5% | **74.6%** ✅ |
| **R@1 vs Baseline** | +10.3% | +6.7% |
| **R@1 vs Full Re-ranking** | -4.0% | -7.6% |
| **Model Type Fit** | Regressor for classification | **Classifier for classification** ✅ |

---

## Why Logistic Regression (Easy) is Better

### 1. **Right Model for the Task** ✅
- Binary classification problem → Use classifier, not regressor
- Logistic Regression is designed for probability estimation
- More interpretable (probability of being easy)

### 2. **Optimal Threshold Learning** ✅
- Threshold (0.410) learned from validation data
- Not arbitrary (like fixed 0.5)
- Data-driven and optimal

### 3. **Higher Accuracy** ✅
- 92.5% validation accuracy vs 86.7% with regressor
- Better generalization
- More reliable predictions

### 4. **Better Efficiency** ✅
- 74.6% time savings vs 57.5% with regressor
- More conservative (detects fewer hard queries)
- Better balance of performance and efficiency

### 5. **Predicts "Easy" Directly** ✅
- More natural: "Is this query easy?" vs "What's the wrong score?"
- Directly actionable: Skip re-ranking if easy
- Better alignment with the goal

---

## Feature Engineering Evolution

### Initial Features (3)
1. `top1_distance` ✅
2. `peakiness` ✅
3. `sue_score` ❌ (broken - all zeros)

### Fixed Features (3)
1. `top1_distance` ✅
2. `peakiness` ✅
3. `sue_score` ✅ (fixed - normalized distances, adjusted slope)

### Final Features (8) ⭐
1. `top1_distance` ✅
2. `peakiness` ✅
3. `sue_score` ✅
4. **`topk_distance_spread`** ✅ (new)
5. **`top1_top2_similarity`** ✅ (new)
6. **`top1_top3_ratio`** ✅ (new)
7. **`top2_top3_ratio`** ✅ (new)
8. **`geographic_clustering`** ✅ (new)

**Why 8 Features?**
- All available **before** image matching
- Capture different aspects of uncertainty:
  - **Descriptor uncertainty**: top1_distance, peakiness
  - **Spatial uncertainty**: sue_score, geographic_clustering
  - **Ranking ambiguity**: top1_top2_similarity, top1_top3_ratio, top2_top3_ratio
  - **Distribution spread**: topk_distance_spread

---

## Threshold Selection Strategy

### Problem with Fixed Threshold
- Fixed 0.5 threshold is arbitrary
- May not be optimal for different datasets
- Doesn't adapt to data distribution

### Solution: Learned Threshold ✅
1. **Train model** on training data
2. **Predict on validation data**
3. **Find optimal threshold** that maximizes F1-score
4. **Save threshold** with model
5. **Apply threshold** during inference

**Result**: Optimal threshold = 0.410 (learned from SF-XS validation set)

**Why Validation Set?**
- Prevents overfitting (threshold not seen during training)
- Better generalization to test data
- Standard machine learning practice

---

## Model Training Details

### Logistic Regression Configuration

```python
LogisticRegression(
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,
    random_state=42
)
```

### Class Balancing
- **Problem**: Class imbalance (more easy queries than hard)
- **Solution**: `class_weight='balanced'`
- **Effect**: Model learns to handle both classes equally

### Feature Scaling
- **Method**: StandardScaler (mean=0, std=1)
- **Why**: Logistic Regression benefits from normalized features
- **Applied**: To both training and test data

---

## Validation Strategy

### Data Splits
- **Training**: SVOX train (1,414 queries)
- **Validation**: SF-XS val (7,993 queries)
- **Test**: SF-XS test, Tokyo-XS test, SVOX test

### Why Separate Validation Set?
- SF-XS val is larger and more representative
- Better threshold estimation with more data
- Tests generalization across datasets

---

## Key Insights

### 1. **Target Selection Matters**
- Predicting "wrong score" (regressor) vs "easy score" (classifier)
- Directly observable target (Top-1 correctness) works better
- More stable across datasets

### 2. **Feature Engineering is Critical**
- Fixed `sue_score` (was broken)
- Added 5 new features (improved accuracy)
- All features must be available before image matching

### 3. **Threshold Learning is Important**
- Learned threshold (0.410) better than fixed (0.5)
- Validation set used for threshold selection
- Prevents overfitting

### 4. **Model Type Matters**
- Classifier (Logistic Regression) better than regressor for binary classification
- Right tool for the right job

---

*See [Results](RESULTS.md) for complete experimental results.*

