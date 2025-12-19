# Threshold Analysis Workflow - Complete Guide

## Overview

This workflow implements a **proper evaluation protocol**:
1. **Predict hard queries** using model with different thresholds (without knowing full re-ranking results)
2. **Run adaptive re-ranking** for each threshold
3. **Evaluate performance** for each threshold
4. **Compare with full re-ranking** (ground truth) at the end
5. **Generate comprehensive plots** and analysis

---

## Key Principle

**We do NOT use full re-ranking results during threshold selection!**

- Full re-ranking is only used as **ground truth for comparison**
- Threshold selection is based on **model predictions only**
- This ensures fair evaluation

---

## Scripts Created

### 1. `adaptive_reranking_threshold_analysis.py`

**Purpose**: Comprehensive threshold analysis across multiple datasets

**Features**:
- Tests multiple thresholds (0.1 to 0.99)
- For each threshold:
  - Predicts hard queries using model
  - Runs adaptive re-ranking
  - Evaluates performance
- Compares with full re-ranking (ground truth) at the end
- Generates 5 comprehensive plots
- Calculates cost savings

**Usage**:
```bash
python adaptive_reranking_threshold_analysis.py \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --datasets sf_xs_test tokyo_xs_test \
  --feature-paths \
    data/features_and_predictions/features_sf_xs_test_improved.npz \
    data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --preds-dirs \
    logs/log_sf_xs_test/2025-12-17_21-14-10/preds \
    log_tokyo_xs_test/2025-12-18_14-43-02/preds \
  --inliers-dirs \
    logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg \
    log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg \
  --output-dir output_stages/threshold_analysis_comprehensive \
  --threshold-range 0.1 0.99 \
  --threshold-step 0.05 \
  --num-preds 20 \
  --positive-dist-threshold 25
```

### 2. `serialize_results_to_matlab.py`

**Purpose**: Serialize analysis results to MATLAB .mat files

**Usage**:
```bash
python serialize_results_to_matlab.py \
  --results-dir output_stages/threshold_analysis_comprehensive \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --feature-path data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --output-dir output_stages/matlab_files
```

---

## Generated Plots

### 1. `recall_at_1_vs_threshold.png`
- Shows R@1 for each threshold
- Different lines for different datasets
- Horizontal dashed lines show full re-ranking performance (ground truth)

### 2. `performance_ratio_vs_threshold.png`
- Shows performance ratio (Adaptive / Full Re-ranking) vs threshold
- 90% target line shown
- Helps identify thresholds that achieve target performance

### 3. `cost_savings_vs_threshold.png`
- Shows time savings (%) for each threshold
- Higher threshold = more hard queries = less time savings

### 4. `recall_vs_cost_savings.png`
- Trade-off curve: Performance vs Cost
- Shows Pareto frontier
- Full re-ranking point marked (0% savings, full performance)

### 5. `dataset_comparison_optimal.png`
- Bar chart comparing optimal thresholds across datasets
- Shows how dataset choice influences threshold

---

## Output Files

### Plots (PNG):
- `recall_at_1_vs_threshold.png`
- `performance_ratio_vs_threshold.png`
- `cost_savings_vs_threshold.png`
- `recall_vs_cost_savings.png`
- `dataset_comparison_optimal.png`

### Summary Report:
- `threshold_analysis_summary.md` - Comprehensive markdown report

### MATLAB Files:
- `threshold_analysis_results.mat` - Analysis results
- `model_predictions.mat` - Model predictions

---

## Cost Savings Calculation

**Formula**: `Cost Savings = (Easy Queries / Total Queries) × 100%`

**Where**:
- **Easy queries**: Queries predicted as easy (skip image matching)
- **Time per query**: ~9.5 seconds (SuperPoint + LightGlue)
- **Total time saved**: Easy queries × 9.5 seconds

**Example** (SF-XS test, threshold=0.41):
- 1,000 queries total
- 746 easy queries (74.6%)
- **Cost savings**: 74.6%
- **Time saved**: 746 × 9.5 seconds = 7,087 seconds ≈ 118 minutes

---

## Workflow Summary

### Step 1: Run Threshold Analysis
```bash
python adaptive_reranking_threshold_analysis.py [arguments]
```

**What it does**:
- Tests different thresholds
- For each threshold, predicts hard queries
- Runs adaptive re-ranking
- Evaluates performance
- Compares with full re-ranking at the end

### Step 2: Serialize to MATLAB
```bash
python serialize_results_to_matlab.py [arguments]
```

**What it does**:
- Converts results to MATLAB .mat format
- Saves model predictions
- Enables further analysis in MATLAB

### Step 3: Review Results
- Check plots in `output_stages/threshold_analysis_comprehensive/`
- Read summary report
- Identify optimal thresholds for each dataset

---

## Key Findings from Analysis

### SF-XS Test:
- **Optimal threshold**: ~0.40-0.45
- **Optimal R@1**: ~69-70%
- **Full re-ranking R@1**: 77.4%
- **Performance ratio**: ~90-91%
- **Cost savings**: ~70-75%

### Tokyo-XS Test:
- **Optimal threshold**: ~1.00 (all queries hard)
- **Optimal R@1**: ~83.2% (same as full re-ranking)
- **Full re-ranking R@1**: 83.2%
- **Performance ratio**: 100%
- **Cost savings**: 0% (model too confident)

**Issue**: Model is overconfident on Tokyo-XS (all probabilities >= 0.726)

---

## Interpretation Guide

### Reading R@1 vs Threshold Plot:
- **X-axis**: Threshold value
- **Y-axis**: Recall@1 (%)
- **Lines**: One per dataset
- **Dashed lines**: Full re-ranking performance (ground truth)
- **Optimal point**: Highest R@1 for each dataset

### Reading Performance Ratio Plot:
- **Y-axis**: Adaptive R@1 / Full Re-ranking R@1
- **90% line**: Target performance (red dashed)
- **Above 90%**: Good performance
- **Below 90%**: Needs improvement

### Reading Trade-off Plot:
- **X-axis**: Cost savings (%)
- **Y-axis**: Recall@1 (%)
- **Top-right**: Best (high performance, high savings)
- **Full re-ranking point**: (0% savings, full performance)

---

## Best Practices

1. **Don't use full re-ranking during threshold selection**
   - Full re-ranking is only for comparison
   - Threshold should be selected based on model predictions

2. **Test wide threshold range**
   - Start with 0.1 to 0.99
   - Use step size 0.02-0.05 for detailed analysis

3. **Compare across datasets**
   - Shows how dataset choice influences threshold
   - Helps identify generalization issues

4. **Calculate cost savings**
   - Important for practical deployment
   - Balance performance vs efficiency

---

*See [Threshold Analysis](THRESHOLD_ANALYSIS.md) for detailed documentation.*

