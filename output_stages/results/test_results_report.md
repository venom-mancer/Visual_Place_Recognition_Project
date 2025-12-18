# Extension 6.1 - Test Results Report

## Model Information
- **Training Dataset**: SVOX train (sun+night combined)
- **Validation Dataset**: SF-XS val
- **Model**: Logistic Regression with validation
- **Model File**: `logreg_svox_with_val.pkl`
- **Validation ROC-AUC**: 0.9700

## Test Results

### SF-XS Test ✅

| Metric | Value |
|--------|-------|
| **Total Queries** | 1,000 |
| **Easy Queries** (skip re-ranking) | 618 (61.8%) |
| **Hard Queries** (apply re-ranking) | 382 (38.2%) |

**Performance Comparison:**

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| **Baseline (Retrieval-only)** | 63.1% | 74.8% | 78.6% | 81.4% |
| **Always Re-rank** | 77.4% | 80.3% | 80.9% | 81.4% |
| **Adaptive Re-ranking** | **76.6%** | **79.9%** | **80.8%** | **81.4%** |
| **Gain vs Baseline** | **+13.5%** | **+5.1%** | **+2.2%** | **+0.0%** |

**Key Findings:**
- Adaptive re-ranking achieves **+13.5% improvement** in R@1 compared to baseline
- Adaptive re-ranking is **nearly as good** as always re-ranking (only -0.8% in R@1)
- Adaptive re-ranking **saves computation** by skipping re-ranking for 61.8% of queries
- At R@20, all methods converge (81.4%) - re-ranking helps most at lower N values

**Execution Time Comparison (Current Implementation):**

| Method | VPR Eval | Image Matching | Evaluation | **Total** |
|--------|----------|----------------|------------|-----------|
| **Baseline** | 9.19 min | **SKIPPED** | 0.1 sec | **9.19 min** |
| **Always Re-rank** | 9.19 min | 157.9 min | 6.1 sec | **167.2 min (2.79 hours)** |
| **Adaptive Re-rank** | 9.19 min | 157.9 min | 2.4 sec | **167.1 min (2.79 hours)** |

**⚠️ ISSUE IDENTIFIED:**

The current implementation has a **pipeline order problem**:
1. Image matching was run for **ALL 1000 queries** (157.9 min) BEFORE deciding which are hard/easy
2. Then logistic regression was applied to classify queries
3. Evaluation correctly skips re-ranking for easy queries, but we already paid the cost!

**Expected Time Savings (If Implemented Correctly):**
- Image matching should run **ONLY for hard queries** (382 queries = 38.2%)
- Expected time: 157.9 min × 0.382 = **60.3 min** (instead of 157.9 min)
- **Time saved**: 97.6 min (1.63 hours) - **61.8% reduction**

**Correct Pipeline Order Should Be:**
1. VPR Evaluation
2. Extract features (WITHOUT num_inliers) - fast, retrieval-based
3. Apply logistic regression → predict hard queries
4. Image matching **ONLY for hard queries** ← This step is missing!
5. Evaluate adaptive re-ranking

**Current Performance:**
- Adaptive (76.6%) is close to Always Re-rank (77.4%) because we're using inliers from ALL queries
- The evaluation script correctly skips re-ranking for easy queries, but inliers were already computed
- To get true cost savings, we need to run image matching AFTER prediction, not before

**Files:**
- Features: `features_sf_xs_test.npz`
- LogReg Outputs: `logreg_sf_xs_test_outputs.npz`
- CSV Exports: 
  - `output_stages/stage1_features_sf_xs_test.csv`
  - `output_stages/stage4_sf_xs_test_logreg_outputs.csv`

---

### Tokyo-XS Test ⏳

**Status**: Pending

---

### SVOX Test ⏳

**Status**: Pending

---

---

## Optimized Pipeline Results (3 Features, Selective Re-ranking) ✅

### SF-XS Test - Optimized Pipeline

| Metric | Value |
|--------|-------|
| **Total Queries** | 1,000 |
| **Easy Queries** (skip matching) | 798 (79.8%) |
| **Hard Queries** (apply matching) | 202 (20.2%) |

**Performance Results:**

| Method | R@1 | R@5 | R@10 | R@20 |
|--------|-----|-----|------|------|
| **Optimized Pipeline** | **67.5%** | **76.8%** | **79.4%** | **81.4%** |
| **Original Pipeline** | 76.6% | 79.9% | 80.8% | 81.4% |
| **Difference** | -9.1% | -3.1% | -1.4% | 0.0% |

**Execution Time (Optimized):**

| Method | VPR Eval | Image Matching | Evaluation | **Total** |
|--------|----------|----------------|------------|-----------|
| **Optimized Pipeline** | 9.19 min | **31.6 min** (202 queries) | 2.4 sec | **40.8 min** |
| **Original Pipeline** | 9.19 min | 157.9 min (1000 queries) | 2.4 sec | 167.2 min |
| **Time Saved** | 0 | **126.3 min** | 0 | **126.4 min (75.6%)** |

**Key Findings:**
- ✅ **Massive Time Savings**: 75.6% reduction in total time (126.4 minutes saved)
- ✅ **Correct Implementation**: Only 202 hard queries processed, 0 easy queries incorrectly processed
- ⚠️ **Performance Trade-off**: -9.1% R@1 compared to original (67.5% vs 76.6%)
- ✅ **Still Better than Baseline**: 67.5% vs 63.1% baseline (+4.4% improvement)

**Files:**
- Model: `logreg_no_inliers_with_val.pkl` (3 features)
- Features: `features_sf_xs_test_no_inliers.npz`
- Predictions: `logreg_sf_xs_test_before_matching.npz`
- Hard Queries: `hard_queries_sf_xs_test.txt` (202 queries)
- Image Matching: `logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg_optimized/`

---

## Summary

| Dataset | Pipeline | Status | R@1 | R@5 | R@10 | R@20 | Time Saved |
|---------|----------|--------|-----|-----|------|------|------------|
| SF-XS Test | Original (4 features) | ✅ | 76.6% | 79.9% | 80.8% | 81.4% | - |
| SF-XS Test | Optimized (3 features) | ✅ | 67.5% | 76.8% | 79.4% | 81.4% | **75.6%** |
| Tokyo-XS Test | - | ⏳ Pending | - | - | - | - | - |
| SVOX Test | - | ⏳ Pending | - | - | - | - | - |

---

*Last updated: After optimized pipeline completion*

