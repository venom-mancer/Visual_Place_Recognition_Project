# Extension 6.1 - Test Results Report

## Model Information
- **Training Dataset**: SVOX train (sun+night combined)
- **Validation Dataset**: SF-XS val
- **Model**: Logistic Regression with validation
- **Model File**: `logreg_svox_with_val.pkl`
- **Validation ROC-AUC**: 0.9700

## Test Results

---

## Adaptive vs Full Re-ranking (Recall@1) - Test Sets (Amir_V2 inliers_top1 gate)

Each row uses the model’s **saved validation threshold** (`optimal_threshold` from the `.pkl` bundle).

**Evaluation rule (true adaptive behavior):**
- **Step 1 (gate feature)**: run **top‑1 matching for all queries** to compute `inliers_top1`
- **Step 2 (gate)**: Logistic Regression predicts **hard vs easy**
- **Step 3 (expensive step)**: run **top‑K matching only for hard queries**, skip matching for easy queries
- **Step 4 (reranking)**: re-rank only the hard subset; easy subset stays retrieval-only

Note: the table numbers were computed using already-available top‑K inliers folders for speed, but the *logic matches* the true pipeline above (hard uses inliers; easy skips reranking).

| Dataset | Model | Hard% | Time Saving | Baseline R@1 | Adaptive R@1 | Full R@1 | dR1 vs Baseline | dR1 vs Full |
|---------|-------|-------|------------|-------------|-------------|--------|----------------|------------|
| SF-XS test | Night + Sun | 35.1% | 64.9% | 63.10% | 76.00% | 77.40% | +12.90% | -1.40% |
| SF-XS test | Night Only | 35.5% | 64.5% | 63.10% | 76.10% | 77.40% | +13.00% | -1.30% |
| SF-XS test | Sun Only | 35.6% | 64.4% | 63.10% | 76.10% | 77.40% | +13.00% | -1.30% |
| Tokyo-XS test | Night + Sun | 41.0% | 59.0% | 65.10% | 83.50% | 83.20% | +18.40% | +0.30% |
| Tokyo-XS test | Night Only | 41.3% | 58.7% | 65.10% | 83.20% | 83.20% | +18.10% | +0.00% |
| Tokyo-XS test | Sun Only | 41.3% | 58.7% | 65.10% | 83.20% | 83.20% | +18.10% | +0.00% |
| SVOX Night test | Night + Sun | 59.7% | 40.3% | 33.29% | 58.10% | 62.10% | +24.81% | -4.00% |
| SVOX Night test | Night Only | 60.0% | 40.0% | 33.29% | 58.10% | 62.10% | +24.81% | -4.00% |
| SVOX Night test | Sun Only | 60.6% | 39.4% | 33.29% | 58.30% | 62.10% | +25.01% | -3.80% |
| SVOX Sun test | Night + Sun | 28.6% | 71.4% | 62.30% | 77.50% | 84.50% | +15.20% | -7.00% |
| SVOX Sun test | Night Only | 29.2% | 70.8% | 62.30% | 78.10% | 84.50% | +15.80% | -6.40% |
| SVOX Sun test | Sun Only | 29.5% | 70.5% | 62.30% | 78.10% | 84.50% | +15.80% | -6.40% |

---

## Execution Time (True Amir_V2 Pipeline) — Re-evaluated

**Definition (true pipeline)**:
- Retrieval
- Top‑1 matching for all queries (gate feature)
- Top‑K matching **only** for predicted hard queries

**Online-time formula (per dataset+model)**:

\[
T_{\text{total}} \approx T_{\text{retrieval}} + T_{\text{top1}} + \left(\frac{\text{Hard\%}}{100}\right)\cdot T_{\text{topK(full)}}
\]

Where:
- \(T_{\text{retrieval}}\): compute the top‑K candidates (needed to know the top‑1/top‑K to match against)
- \(T_{\text{top1}}\): run image matching for **only the retrieved top‑1** for every query (to get `inliers_top1`)
- \(T_{\text{topK(full)}}\): time to run top‑K matching for **all** queries (full pipeline matching cost)
- \(\frac{\text{Hard\%}}{100}\cdot T_{\text{topK(full)}}\): estimated top‑K matching cost for **hard queries only**

**How times were computed**:
- **Retrieval**: from dataset `info.log` when available (SF‑XS, Tokyo‑XS), else N/A
- **Top‑1 match**: measured by running `match_queries_preds.py --num-preds 1` into fresh `temp/` folders
- **Full top‑K match**: approximated by mtime-span of an existing “full top‑20 inliers” directory
- **Top‑K hard-only**: estimated as `full_topK_time * hard_fraction`

> These are execution-time estimates for matching cost; evaluation time is negligible compared to matching.

| Dataset | Model | Hard% | Time Saving | Retrieval | Top-1 match | Top-K match (hard only, est.) | Total (est.) | Full pipeline (est.) |
|---------|-------|-------|------------|----------|------------|-------------------------------|-------------|----------------------|
| SF-XS test | Night + Sun | 35.1% | 64.9% | 9.2m | 8.7m | 54.7m | 1.21h | 2.75h |
| SF-XS test | Night Only | 35.5% | 64.5% | 9.2m | 8.7m | 55.3m | 1.22h | 2.75h |
| SF-XS test | Sun Only | 35.6% | 64.4% | 9.2m | 8.7m | 55.4m | 1.22h | 2.75h |
| Tokyo-XS test | Night + Sun | 41.0% | 59.0% | 3.0m | 2.7m | 20.0m | 25.8m | 52.0m |
| Tokyo-XS test | Night Only | 41.3% | 58.7% | 3.0m | 2.7m | 20.2m | 26.0m | 52.0m |
| Tokyo-XS test | Sun Only | 41.3% | 58.7% | 3.0m | 2.7m | 20.2m | 26.0m | 52.0m |
| SVOX Night test | Night + Sun | 59.7% | 40.3% | N/A | 6.9m | 3.33h | 3.44h | 5.58h |
| SVOX Night test | Night Only | 60.0% | 40.0% | N/A | 6.9m | 3.35h | 3.46h | 5.58h |
| SVOX Night test | Sun Only | 60.6% | 39.4% | N/A | 6.9m | 3.38h | 3.50h | 5.58h |
| SVOX Sun test | Night + Sun | 28.6% | 71.4% | N/A | 7.2m | 1.62h | 1.74h | 5.68h |
| SVOX Sun test | Night Only | 29.2% | 70.8% | N/A | 7.2m | 1.66h | 1.78h | 5.68h |
| SVOX Sun test | Sun Only | 29.5% | 70.5% | N/A | 7.2m | 1.68h | 1.80h | 5.68h |

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

