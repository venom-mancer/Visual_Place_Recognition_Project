# Tokyo-XS Test Results

## Dataset Information
- **Location**: `data/tokyo_xs/test/`
- **Database**: 12,771 images
- **Queries**: 315 images
- **Status**: ✅ Completed

---

## Performance Comparison

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings |
|--------|-----|-----|------|------|-------------|--------------|
| **Baseline (Retrieval-only)** | **65.1%** | **79.7%** | **86.0%** | **89.5%** | 0% | 100% |
| **Full Re-ranking** | ⏳ Processing | ⏳ Processing | ⏳ Processing | ⏳ Processing | 100% | 0% |
| **Adaptive (LogReg Easy)** | **65.1%** | **79.7%** | **86.0%** | **89.5%** | **0.0%** | **100%** |

---

## Key Findings

### Model Prediction:
- **Easy queries (predicted)**: 315 (100.0%)
- **Hard queries (predicted)**: 0 (0.0%)
- **Actually wrong queries**: 110 (34.9%)
- **Model threshold**: 0.410 (learned from validation)

### Prediction Statistics:
- **Min probability**: 0.670
- **Max probability**: 1.000
- **Mean probability**: 0.999
- **Median probability**: 1.000

### Analysis:
The model predicted **ALL queries as easy** (100%), which means:
- ✅ **Maximum time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline (65.1% R@1)
- ⚠️ **Model is too conservative**: All probabilities >= 0.410 threshold

**Possible reasons:**
1. Tokyo-XS test is easier than SF-XS test (65.1% vs 63.1% baseline)
2. Model trained on SF-XS val may not generalize well to Tokyo-XS
3. Threshold (0.410) may be too low for this dataset

---

## Comparison with SF-XS Test

| Dataset | Baseline R@1 | Adaptive R@1 | Hard Queries Detected | Time Savings |
|---------|--------------|--------------|----------------------|--------------|
| **SF-XS test** | 63.1% | 69.8% | 25.4% (254/1000) | 74.6% |
| **Tokyo-XS test** | 65.1% | 65.1% | 0.0% (0/315) | 100% |

**Key Difference:**
- SF-XS: Model detected 25.4% hard queries → +6.7% R@1 improvement
- Tokyo-XS: Model detected 0% hard queries → No improvement (but 100% time savings)

---

## Files Generated:
- **Features**: `features_tokyo_xs_test_improved.npz`
- **Model predictions**: `logreg_easy_tokyo_xs_test.npz`
- **Hard queries list**: `hard_queries_tokyo_xs_test_logreg_easy.txt` (empty - 0 queries)
- **VPR logs**: `log_tokyo_xs_test/2025-12-18_14-43-02/`

---

*Evaluation completed: 2025-12-18*
*Full re-ranking results: Pending (image matching in progress)*

