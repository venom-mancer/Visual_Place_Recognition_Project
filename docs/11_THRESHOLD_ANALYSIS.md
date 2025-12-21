# Threshold Analysis - R@1 vs Threshold and Cost Savings

## Overview

This document explains how to generate comprehensive threshold analysis plots showing:
1. **R@1 vs Threshold** for different datasets
2. **How dataset choice influences threshold** computation and final performance
3. **Cost savings** of the adaptive re-ranking strategy

---

## Running the Analysis

### Script: `analyze_threshold_impact.py`

This script performs comprehensive threshold analysis across multiple datasets.

### Basic Usage

```bash
python analyze_threshold_impact.py \
  --model-path logreg_easy_queries_optimal.pkl \
  --datasets sf_xs_test tokyo_xs_test svox_test \
  --feature-paths \
    data/features_and_predictions/features_sf_xs_test_improved.npz \
    data/features_and_predictions/features_tokyo_xs_test_improved.npz \
    data/features_and_predictions/features_svox_test_improved.npz \
  --preds-dirs \
    logs/log_sf_xs_test/[timestamp]/preds \
    log_tokyo_xs_test/[timestamp]/preds \
    log_svox_test/[timestamp]/preds \
  --inliers-dirs \
    logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg \
    log_tokyo_xs_test/[timestamp]/preds_superpoint-lg \
    log_svox_test/[timestamp]/preds_superpoint-lg \
  --output-dir output_stages/threshold_analysis \
  --threshold-range 0.1 0.95 \
  --threshold-step 0.02 \
  --num-preds 20 \
  --positive-dist-threshold 25
```

### Parameters

- `--model-path`: Trained logistic regression model
- `--datasets`: List of dataset names
- `--feature-paths`: Feature files for each dataset
- `--preds-dirs`: Prediction directories (with .txt files)
- `--inliers-dirs`: Inliers directories (with .torch files from full re-ranking)
- `--output-dir`: Where to save plots and summary
- `--threshold-range`: Range of thresholds to test (default: 0.1 0.95)
- `--threshold-step`: Step size for threshold testing (default: 0.02)
- `--num-preds`: Number of predictions to consider (default: 20)
- `--positive-dist-threshold`: Distance threshold in meters (default: 25)

---

## Generated Plots

### 1. Recall@1 vs Threshold

**File**: `recall_vs_threshold.png`

Shows how R@1 varies as a function of threshold for each dataset.

**Key insights**:
- Optimal threshold for each dataset (marked with star)
- How R@1 changes with threshold
- Dataset-specific threshold requirements

### 2. Hard Query Rate vs Threshold

**File**: `hard_query_rate_vs_threshold.png`

Shows percentage of queries predicted as hard for each threshold.

**Key insights**:
- How threshold affects hard query detection
- Dataset-specific behavior

### 3. Cost Savings vs Threshold

**File**: `cost_savings_vs_threshold.png`

Shows time savings (percentage of queries that skip image matching) for each threshold.

**Key insights**:
- Trade-off between threshold and computational cost
- Maximum achievable cost savings

### 4. Performance vs Cost Trade-off

**File**: `recall_vs_cost_savings.png`

Shows R@1 vs cost savings (Pareto curve).

**Key insights**:
- Optimal operating points
- Performance-cost trade-offs
- Efficiency of adaptive strategy

### 5. Dataset Comparison - Optimal Thresholds

**File**: `dataset_comparison_optimal.png`

Bar chart comparing optimal thresholds and max R@1 across datasets.

**Key insights**:
- Threshold variation across datasets
- Performance differences

---

## Summary Report

**File**: `threshold_analysis_summary.md`

Contains:
- Comparison table with all metrics
- Key findings and conclusions
- Dataset-specific characteristics

---

## Expected Results

### For SF-XS Test:
- **Optimal threshold**: ~0.40-0.45
- **Max R@1**: ~69-70%
- **Cost savings**: ~70-75%
- **Hard query rate**: ~25-30%

### For Tokyo-XS Test:
- **Optimal threshold**: ~0.65-0.70
- **Max R@1**: ~80-83%
- **Cost savings**: ~30-40%
- **Hard query rate**: ~60-70%

### For SVOX Test:
- **Optimal threshold**: ~0.70-0.80
- **Max R@1**: ~96-97%
- **Cost savings**: ~95-97%
- **Hard query rate**: ~3-5%

---

## Cost Savings Calculation

**Cost Savings = (Number of Easy Queries / Total Queries) × 100%**

Where:
- **Easy queries**: Queries predicted as easy (skip image matching)
- **Time saved**: Percentage of image matching operations skipped
- **Average time per query**: ~9.5 seconds (SuperPoint + LightGlue)

**Example**:
- 1,000 queries total
- 254 hard queries (25.4%)
- 746 easy queries (74.6%)
- **Cost savings**: 74.6% of image matching time
- **Time saved**: 746 × 9.5 seconds = 7,087 seconds ≈ 118 minutes

---

## Dataset Influence Analysis

The analysis reveals:

1. **Threshold Variation**:
   - Different datasets require different optimal thresholds
   - SF-XS: Lower threshold (~0.41)
   - Tokyo-XS: Higher threshold (~0.65-0.70)
   - SVOX: Very high threshold (~0.70-0.80)

2. **Feature Distribution Shift**:
   - Mean probabilities vary across datasets
   - SF-XS: Lower mean probability (more uncertainty)
   - SVOX: Very high mean probability (high confidence)

3. **Performance Impact**:
   - Using wrong threshold can significantly reduce R@1
   - Dataset-specific calibration is essential

---

## Interpretation Guide

### Reading the R@1 vs Threshold Plot

- **Steep curve**: Threshold is critical (small changes have large impact)
- **Flat curve**: Threshold is less critical (robust to changes)
- **Optimal point**: Maximum R@1 (marked with star)

### Reading the Trade-off Plot

- **Top-right corner**: Best performance and cost savings
- **Pareto frontier**: Optimal operating points
- **Curve shape**: Efficiency of adaptive strategy

---

## Example Output

After running the analysis, you'll get:

```
output_stages/threshold_analysis/
├── recall_vs_threshold.png
├── hard_query_rate_vs_threshold.png
├── cost_savings_vs_threshold.png
├── recall_vs_cost_savings.png
├── dataset_comparison_optimal.png
└── threshold_analysis_summary.md
```

---

*See [Threshold Calibration Guide](THRESHOLD_CALIBRATION_GUIDE.md) for how to use calibrated thresholds.*

