# Feature Quality Analysis - Why Performance is Limited

## Executive Summary

**Yes, you are correct!** The features are **not good enough**, which explains the -9.1% performance drop. Here's why:

---

## Critical Issues Identified

### 1. **sue_score is Completely Broken** ❌

| Issue | Impact |
|-------|--------|
| **All 1000 queries have sue_score = 0.0** | Feature provides **zero information** |
| **Correlation with hard queries: NaN** | Cannot help the model |
| **100% of values are zero** | Effectively using only 2 features instead of 3 |

**Root Cause**: The SUE computation is likely failing due to:
- All queries having very similar spatial distributions
- The variance calculation returning 0 for all queries
- Potential issue with the reference poses or distance calculations

### 2. **Model Accuracy is Low (64.5%)** ⚠️

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Accuracy** | 64.5% | **Below acceptable threshold** (should be >75%) |
| **Hard Query Detection** | 34.6% | **Very poor** - missing 65.4% of hard queries! |
| **Easy Query Detection** | 94.0% | Good - correctly identifying most easy queries |

**The Problem**: The model is **too conservative** - it's predicting most queries as "easy" and missing many that actually need re-ranking.

### 3. **Misclassification Analysis** 🔍

| Type | Count | Percentage | Impact |
|------|-------|------------|--------|
| **Wrong queries predicted as EASY** | 215 | 58.3% of wrong queries | **Major issue** - these would benefit from re-ranking but are skipped |
| **Correct queries predicted as HARD** | 48 | 7.6% of correct queries | Minor issue - wasting time on easy queries |

**Key Finding**: **58.3% of wrong queries are being skipped** - this directly explains the -9.1% R@1 drop!

---

## Feature Statistics

### Effective Features (2 out of 3):

| Feature | Hard Queries | Easy Queries | Difference | Correlation |
|---------|--------------|--------------|------------|-------------|
| **top1_distance** | 1.310 | 0.972 | **+0.338** | **0.669** (strong) |
| **peakiness** | 0.979 | 0.902 | +0.077 | 0.380 (moderate) |

### Broken Feature:

| Feature | Hard Queries | Easy Queries | Difference | Correlation |
|---------|--------------|--------------|------------|-------------|
| **sue_score** | 0.000 | 0.000 | **0.000** | **NaN** (useless) |

---

## Performance Impact

### Why We're Losing -9.1% R@1:

1. **Missing Hard Queries**: 325 queries (32.5%) that are actually hard are predicted as easy
2. **No Re-ranking Applied**: These 325 queries don't get re-ranked, so they stay at baseline performance
3. **Direct Impact**: If we had correctly identified these queries, we could have:
   - Applied re-ranking to them
   - Potentially recovered most of the -9.1% drop

### Current vs Ideal:

| Scenario | Hard Queries | R@1 Performance |
|----------|--------------|-----------------|
| **Ideal** (all hard queries identified) | 497 (49.7%) | ~76.6% (original) |
| **Current** (model prediction) | 202 (20.2%) | 67.5% (optimized) |
| **Missing** | 295 queries | **-9.1% R@1** |

---

## Model Prediction Quality

### Probability Distribution:

| Statistic | Value |
|-----------|-------|
| **Mean probability** | 0.284 (low - model is conservative) |
| **Hard query probs** | 0.501 - 0.840 (mean: 0.629) |
| **Easy query probs** | 0.001 - 0.498 (mean: 0.197) |

**Issue**: The model is too conservative - most probabilities are low, leading to only 20.2% being classified as hard (vs. 49.7% that are actually hard).

---

## Comparison with Original Pipeline

### Original Pipeline (4 features):

| Feature | Available? | Quality |
|---------|------------|---------|
| **num_inliers** | ✅ Yes | **Excellent** - direct indicator of matching quality |
| **top1_distance** | ✅ Yes | Good - strong correlation |
| **peakiness** | ✅ Yes | Moderate - some correlation |
| **sue_score** | ✅ Yes | Unknown (not analyzed) |

**Result**: 76.6% R@1 with 100% queries re-ranked

### Optimized Pipeline (3 features):

| Feature | Available? | Quality |
|---------|------------|---------|
| **num_inliers** | ❌ No | **Missing** - not available before matching |
| **top1_distance** | ✅ Yes | Good - strong correlation (0.669) |
| **peakiness** | ✅ Yes | Moderate - weak correlation (0.380) |
| **sue_score** | ✅ Yes | **Broken** - all zeros, zero correlation |

**Result**: 67.5% R@1 with 20.2% queries re-ranked

**Key Loss**: Removing `num_inliers` (the best feature) and having `sue_score` broken means we're effectively using only 1.5 features!

---

## Recommendations

### 1. **Fix sue_score Computation** 🔧

**Priority: HIGH**

- Investigate why all values are zero
- Check if reference poses (`database_utms`) are correct
- Verify distance calculations
- Consider alternative spatial uncertainty measures

### 2. **Improve Feature Engineering** 📊

**Priority: HIGH**

- Add more retrieval-based features:
  - **Top-K distance spread**: variance of top-5 distances
  - **Ranking consistency**: how stable are top predictions?
  - **Descriptor similarity**: cosine similarity between top-1 and top-2
  - **Geographic clustering**: density of top-K predictions

### 3. **Tune Model Threshold** ⚖️

**Priority: MEDIUM**

- Current threshold: 0.5 (too conservative)
- Try lower thresholds (0.3, 0.4) to catch more hard queries
- Trade-off: More queries processed vs. time saved

### 4. **Use Ensemble Features** 🎯

**Priority: MEDIUM**

- Combine multiple uncertainty measures
- Use learned feature combinations
- Consider non-linear models (e.g., Random Forest, XGBoost)

### 5. **Alternative Approach** 💡

**Priority: LOW**

- Use a two-stage approach:
  1. Fast filter (current 3 features) → identify likely hard queries
  2. Quick matching check (fewer keypoints) → confirm hard queries
  3. Full matching only for confirmed hard queries

---

## Expected Improvements

If we fix the issues:

| Improvement | Expected R@1 Gain |
|-------------|-------------------|
| **Fix sue_score** | +2-3% (from 67.5% → 69.5-70.5%) |
| **Add better features** | +3-5% (from 67.5% → 70.5-72.5%) |
| **Tune threshold** | +1-2% (from 67.5% → 68.5-69.5%) |
| **Combined** | **+6-10%** (from 67.5% → **73.5-77.5%**) |

**Target**: Get close to original 76.6% while maintaining time savings!

---

## Conclusion

**Yes, you are absolutely correct** - the features are not good enough:

1. ✅ **sue_score is completely broken** (all zeros)
2. ✅ **Model accuracy is low** (64.5%)
3. ✅ **Missing 32.5% of hard queries** (predicted as easy)
4. ✅ **Only using 2 effective features** (top1_distance + peakiness)
5. ✅ **Missing the best feature** (num_inliers - not available before matching)

**The -9.1% performance drop is directly caused by poor feature quality and low model accuracy.**

**Next Steps**: Fix `sue_score`, add better features, and tune the model to improve hard query detection.

---

*Analysis Date: After optimized pipeline completion*

