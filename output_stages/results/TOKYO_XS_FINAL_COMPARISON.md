# Tokyo-XS Test - Final Comparison

## Performance Comparison Table

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline (Retrieval-only)** | 65.1% | 79.7% | 86.0% | 89.5% | 0% | 100% | - |
| **Full Re-ranking** | **83.2%** | **87.0%** | **88.3%** | **89.5%** | 100% | 0% | **+18.1%** |
| **Adaptive (LogReg Easy)** | **65.1%** | **79.7%** | **86.0%** | **89.5%** | **0.0%** | **100%** | **0.0%** |

---

## Detailed Results

### Baseline (Retrieval-only)
- **R@1**: 65.1%
- **R@5**: 79.7%
- **R@10**: 86.0%
- **R@20**: 89.5%
- **Queries**: 315

### Adaptive Re-ranking (LogReg Easy)
- **R@1**: 65.1% (same as baseline)
- **R@5**: 79.7% (same as baseline)
- **R@10**: 86.0% (same as baseline)
- **R@20**: 89.5% (same as baseline)
- **Hard queries detected**: 0 (0.0%)
- **Easy queries (skipped)**: 315 (100.0%)
- **Time savings**: 100% (no image matching performed)

### Full Re-ranking
- **R@1**: 83.2% (+18.1% vs baseline)
- **R@5**: 87.0% (+7.3% vs baseline)
- **R@10**: 88.3% (+2.3% vs baseline)
- **R@20**: 89.5% (+0.0% vs baseline)
- **Queries processed**: 315 (100%)
- **Performance gain**: Significant improvement in R@1 (+18.1%)

---

## Key Observations

### 1. Model Prediction Behavior
- **All queries predicted as easy**: 315/315 (100%)
- **Probability range**: 0.670 - 1.000 (mean: 0.999)
- **Threshold used**: 0.410 (learned from SF-XS validation)
- **Result**: Model is extremely conservative on Tokyo-XS test

### 2. Performance Impact
- **Full Re-ranking**: Significant improvement (+18.1% R@1 vs baseline)
- **Adaptive**: No improvement (65.1% R@1 = baseline)
- **Missed opportunity**: Full re-ranking achieves 83.2% R@1, but adaptive detected 0 hard queries
- **Trade-off**: Perfect efficiency (100% time savings), but missed 18.1% performance gain

### 3. Comparison with SF-XS Test

| Aspect | SF-XS Test | Tokyo-XS Test |
|--------|------------|---------------|
| **Baseline R@1** | 63.1% | 65.1% |
| **Adaptive R@1** | 69.8% | 65.1% |
| **R@1 Improvement** | +6.7% | 0.0% |
| **Hard Queries Detected** | 25.4% (254/1000) | 0.0% (0/315) |
| **Time Savings** | 74.6% | 100% |
| **Model Behavior** | Moderate (detects some hard queries) | Very conservative (detects none) |

---

## Analysis

### Why All Queries Predicted as Easy?

1. **Dataset Difficulty**: Tokyo-XS test appears easier than SF-XS test
   - Baseline R@1: 65.1% (Tokyo) vs 63.1% (SF-XS)
   - Higher baseline suggests fewer truly "hard" queries

2. **Model Generalization**: Model trained on SVOX train, validated on SF-XS val
   - May not generalize well to Tokyo-XS (different dataset)
   - Threshold (0.410) optimized for SF-XS val, may be too low for Tokyo-XS

3. **Feature Distribution**: Features on Tokyo-XS may be different
   - All queries have high "easy" probabilities (≥0.670)
   - Model sees all queries as confident/easy

### Implications

**Positive:**
- ✅ Maximum computational efficiency (100% time savings)
- ✅ No performance degradation (same as baseline)

**Negative:**
- ⚠️ No performance improvement (missed opportunity)
- ⚠️ Model too conservative (should detect some hard queries)
- ⚠️ 110 queries are actually wrong (34.9%), but model detected 0

---

## Recommendations

1. **Dataset-Specific Threshold**: Consider tuning threshold per dataset
   - Current: 0.410 (from SF-XS val)
   - Tokyo-XS: May need higher threshold (e.g., 0.6-0.7)

2. **Model Retraining**: Consider training on more diverse datasets
   - Include Tokyo-XS in validation set
   - Or train separate model for Tokyo-XS

3. **Feature Analysis**: Investigate why features are so "easy-looking"
   - Check feature distributions on Tokyo-XS vs SF-XS
   - May need dataset-specific feature normalization

---

## Files Generated

- **Features**: `features_tokyo_xs_test_improved.npz` (315 queries, 8 features)
- **Model predictions**: `logreg_easy_tokyo_xs_test.npz`
- **Hard queries list**: `hard_queries_tokyo_xs_test_logreg_easy.txt` (empty)
- **VPR logs**: `log_tokyo_xs_test/2025-12-18_14-43-02/`
- **Image matching (adaptive)**: `log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg_logreg_easy/` (empty - no hard queries)
- **Image matching (full)**: `log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg/` (315 queries completed)

---

## Performance Gains Summary

| Metric | Full Re-ranking vs Baseline | Adaptive vs Baseline | Adaptive vs Full Re-ranking |
|--------|----------------------------|---------------------|----------------------------|
| **R@1** | **+18.1%** ✅ | 0.0% | **-18.1%** ⚠️ |
| **R@5** | **+7.3%** ✅ | 0.0% | **-7.3%** ⚠️ |
| **R@10** | **+2.3%** ✅ | 0.0% | **-2.3%** ⚠️ |
| **R@20** | 0.0% | 0.0% | 0.0% |

### Key Insight:
- **Full re-ranking** shows that re-ranking is highly beneficial on Tokyo-XS (+18.1% R@1)
- **Adaptive approach** missed this opportunity by predicting all queries as easy
- **Potential improvement**: If model correctly detected hard queries, could achieve close to full re-ranking performance with time savings

---

*Evaluation completed: 2025-12-18*
*All results finalized*

