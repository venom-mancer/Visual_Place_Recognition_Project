# Tokyo-XS Test - Final Comparison

## Performance Comparison Table

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline (Retrieval-only)** | 65.1% | 79.7% | 86.0% | 89.5% | 0% | 100% | - |
| **Full Re-ranking** | **83.2%** | **87.0%** | **88.3%** | **89.5%** | 100% | 0% | **+18.1%** |
| **Adaptive (LogReg, Temp-Scaled)** | **75.2%** | **83.8%** | **87.0%** | **89.5%** | **24.4%** | **75.6%** | **+10.1%** |

---

## Detailed Results

### Baseline (Retrieval-only)
- **R@1**: 65.1%
- **R@5**: 79.7%
- **R@10**: 86.0%
- **R@20**: 89.5%
- **Queries**: 315

### Adaptive Re-ranking (LogReg Easy)
### Adaptive Re-ranking (LogReg, Temp-Scaled)
- **R@1**: 75.2% (**+10.1%** vs baseline)
- **R@5**: 83.8% (**+4.1%** vs baseline)
- **R@10**: 87.0% (**+1.0%** vs baseline)
- **R@20**: 89.5% (+0.0% vs baseline)
- **Hard queries detected**: 77 (24.4%)
- **Easy queries (skipped)**: 238 (75.6%)
- **Time savings**: 75.6% (image matching only for the hard subset)

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
- **Hard/easy split**: 77 hard (24.4%), 238 easy (75.6%)
- **Probability range**: 0.527 - 0.930 (mean: 0.914, median: 0.930)
- **Threshold used**: 0.930 (temp-scaled model output)
- **Result**: Model now detects a meaningful hard subset on Tokyo-XS

### 2. Performance Impact
- **Full Re-ranking**: Significant improvement (+18.1% R@1 vs baseline)
- **Adaptive (Temp-Scaled)**: Strong improvement (+10.1% R@1 vs baseline)
- **Remaining gap**: Full re-ranking achieves 83.2% R@1, adaptive achieves 75.2% R@1 (gap: -8.0%)
- **Trade-off**: 75.6% time savings, while recovering much of the full re-ranking gain

### 3. Comparison with SF-XS Test

| Aspect | SF-XS Test | Tokyo-XS Test |
|--------|------------|---------------|
| **Baseline R@1** | 63.1% | 65.1% |
| **Adaptive R@1** | 69.8% | 75.2% |
| **R@1 Improvement** | +6.7% | +10.1% |
| **Hard Queries Detected** | 25.4% (254/1000) | 24.4% (77/315) |
| **Time Savings** | 74.6% | 75.6% |
| **Model Behavior** | Moderate (detects some hard queries) | Balanced (detects meaningful hard subset) |

---

## Analysis

### Why it used to predict (almost) everything as easy?

Previously, the adaptive pipeline could end up with **0 detected hard queries** on Tokyo because the hard-query list was generated with **index misalignment** (dropping NaN rows changed indices).  
On `Amir_V2`, this is fixed by preserving original query indexing and using query file IDs consistently end-to-end.

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
- **Model predictions**: `data/features_and_predictions/logreg_tokyo_xs_temperature_scaled.npz`
- **Hard queries list**: `data/features_and_predictions/hard_queries_tokyo_xs_temperature_scaled.txt`
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
- **Adaptive approach (Temp-Scaled)** recovers much of that gain (+10.1% R@1) while saving 75.6% matching compute
- **Remaining opportunity**: further reduce the -8.0% gap to full reranking without collapsing time savings

---

*Evaluation completed: 2025-12-18*
*All results finalized*

