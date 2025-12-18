# Logistic Regression Easy Queries - Test Summary

## Test Configuration

### Model:
- **Type**: Logistic Regression (predicts easy queries)
- **Features**: 8 improved features
- **Optimal Threshold**: 0.410 (learned from validation)
- **Model File**: `logreg_easy_queries_optimal.pkl`

### Test Set (SF-XS test, 1,000 queries):
- **Hard queries detected**: 254 (25.4%)
- **Easy queries (skipped)**: 746 (74.6%)
- **Hard queries file**: `hard_queries_test_logreg_easy.txt`

---

## Current Status

### ✅ Completed:
1. **Model Training**: ✅
   - Training Accuracy: 78.9%
   - Validation Accuracy: **92.5%** (with optimal threshold 0.410)
   - Validation F1: 0.9582

2. **Query Detection**: ✅
   - Applied model to test set
   - Detected 254 hard queries (25.4%)
   - Saved predictions: `logreg_easy_queries_test.npz`

3. **Image Matching**: ⏳ **Running in background**
   - Processing 254 hard queries
   - Output directory: `logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg_logreg_easy/`

### ⏳ Pending:
4. **Evaluation**: Waiting for image matching to complete
   - Will evaluate adaptive re-ranking performance
   - Compare with baseline and full re-ranking

---

## Expected Results

### Performance Comparison (Expected):

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings |
|--------|-----|-----|------|------|-------------|--------------|
| **Baseline (Retrieval-only)** | 63.1% | 74.8% | 78.6% | 81.4% | 0% | 100% |
| **Full Re-ranking** | 77.4% | 80.3% | 80.9% | 81.4% | 100% | 0% |
| **Adaptive (LogReg Easy)** | **TBD** | **TBD** | **TBD** | **TBD** | **25.4%** | **74.6%** |

### Expected Improvements:
- **Better than baseline**: Should achieve >63.1% R@1
- **Close to full re-ranking**: Expected 70-75% R@1 (vs 77.4% full)
- **Time savings**: 74.6% reduction in image matching time

---

## Why This Approach Should Work Better

### Advantages over Regressor:
1. **Higher validation accuracy**: 92.5% vs 86.7%
2. **Better detection**: 25.4% hard queries (closer to 36.9% actual) vs 42.5%
3. **Optimal threshold**: Learned 0.410 (not fixed 0.5)
4. **Right model**: Logistic Regression for binary classification
5. **More time savings**: 74.6% vs 57.5%

### Potential Concerns:
- **Under-prediction**: 25.4% detected vs 36.9% actual (-11.5%)
  - This is conservative (better than over-prediction)
  - May miss some hard queries that need re-ranking
  - But should still improve over baseline

---

## Next Steps

1. ⏳ **Wait for image matching to complete** (254 queries)
2. ✅ **Re-run evaluation** once matching files are ready
3. ✅ **Compare results** with baseline and full re-ranking
4. ✅ **Analyze performance** and time savings

---

## Evaluation Command (Once Matching Completes)

```bash
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir logs/log_sf_xs_test/2025-12-17_21-14-10/preds \
  --inliers-dir logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg_logreg_easy \
  --logreg-output logreg_easy_queries_test.npz \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20
```

---

*Status: Image matching in progress, evaluation pending*

