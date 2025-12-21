# Threshold Analysis - Quick Start Guide

## What This Analysis Provides

1. **R@1 vs Threshold plots** for each dataset
2. **Dataset comparison** showing how threshold varies across datasets
3. **Cost savings calculation** (time saved by skipping easy queries)
4. **Performance vs cost trade-off** curves
5. **Optimal threshold identification** for each dataset

---

## Quick Run (Example)

### For SF-XS Test Only:

```bash
python analyze_threshold_impact.py \
  --model-path logreg_easy_queries_optimal.pkl \
  --datasets sf_xs_test \
  --feature-paths data/features_and_predictions/features_sf_xs_test_improved.npz \
  --preds-dirs logs/log_sf_xs_test/2025-12-17_21-14-10/preds \
  --inliers-dirs logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg \
  --output-dir output_stages/threshold_analysis_sf_xs \
  --threshold-range 0.2 0.6 \
  --threshold-step 0.01
```

### For All Test Datasets:

See `run_threshold_analysis_example.bat` (Windows) or `run_threshold_analysis_example.sh` (Linux/Mac)

---

## What You'll Get

### Plots Generated:

1. **`recall_vs_threshold.png`**
   - Shows R@1 for each threshold value
   - Optimal threshold marked with star
   - Different datasets as different lines

2. **`recall_vs_cost_savings.png`**
   - Performance vs cost trade-off
   - Shows efficiency of adaptive strategy

3. **`dataset_comparison_optimal.png`**
   - Bar chart comparing optimal thresholds across datasets
   - Shows dataset influence on threshold

4. **`hard_query_rate_vs_threshold.png`**
   - How threshold affects hard query detection

5. **`cost_savings_vs_threshold.png`**
   - Time savings for each threshold

### Summary Report:

**`threshold_analysis_summary.md`** with:
- Comparison table
- Key findings
- Dataset-specific analysis

---

## Cost Savings Calculation

The script calculates:

**Cost Savings = (Easy Queries / Total Queries) × 100%**

Where:
- **Easy queries**: Queries predicted as easy (skip image matching)
- **Time per query**: ~9.5 seconds (SuperPoint + LightGlue)
- **Total time saved**: Easy queries × 9.5 seconds

**Example** (SF-XS test, threshold=0.410):
- 1,000 queries
- 746 easy queries (74.6%)
- **Cost savings**: 74.6%
- **Time saved**: 746 × 9.5 = 7,087 seconds ≈ 118 minutes

---

## Expected Results

### Threshold Range Analysis:

| Dataset | Optimal Threshold | Max R@1 | Cost Savings |
|---------|------------------|---------|--------------|
| SF-XS test | ~0.40-0.45 | ~69-70% | ~70-75% |
| Tokyo-XS test | ~0.65-0.70 | ~80-83% | ~30-40% |
| SVOX test | ~0.70-0.80 | ~96-97% | ~95-97% |

---

## Interpretation

### R@1 vs Threshold Plot:
- **X-axis**: Threshold value (0.1 to 0.95)
- **Y-axis**: Recall@1 (%)
- **Lines**: One per dataset
- **Stars**: Optimal threshold for each dataset

### Key Observations:
1. **Steep curves**: Threshold is critical (small changes matter)
2. **Flat regions**: Threshold is robust (less sensitive)
3. **Optimal points**: Maximum R@1 achieved

### Dataset Influence:
- **Different datasets → Different optimal thresholds**
- **Feature distribution shift** causes threshold variation
- **Dataset-specific calibration** is essential

---

*See [Threshold Analysis](THRESHOLD_ANALYSIS.md) for detailed documentation.*

