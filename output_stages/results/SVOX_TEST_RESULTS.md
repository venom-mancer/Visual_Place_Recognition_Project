# SVOX Test - Results

## Dataset Information
- **Location**: `data/svox/images/test/`
- **Gallery**: 17,166 images
- **Queries**: 14,278 images
- **Status**: ✅ Completed

---

## Performance Comparison

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline (Retrieval-only)** | **96.3%** | **97.9%** | **98.3%** | **98.7%** | 0% | 100% | - |
| **Full Re-ranking** | ⏳ *Processing* | ⏳ *Processing* | ⏳ *Processing* | ⏳ *Processing* | 100% | 0% | ⏳ *Pending* |
| **Adaptive (LogReg Easy)** | **96.3%** | **97.9%** | **98.3%** | **98.7%** | **0.0%** | **100%** | **0.0%** |

---

## Key Findings

### Model Prediction:
- **Easy queries (predicted)**: 14,278 (100.0%)
- **Hard queries (predicted)**: 0 (0.0%)
- **Actually wrong queries**: 524 (3.7%)
- **Actually correct queries**: 13,754 (96.3%)
- **Model threshold**: 0.410 (learned from validation)

### Prediction Statistics:
- **Min probability**: 0.700
- **Max probability**: 1.000
- **Mean probability**: 1.000
- **Median probability**: 1.000

### Analysis:
The model predicted **ALL queries as easy** (100%), which means:
- ✅ **Maximum time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline (96.3% R@1)
- ⚠️ **Model is too conservative**: All probabilities >= 0.410 threshold

**Possible reasons:**
1. SVOX test is very easy (96.3% baseline R@1 - very high!)
2. Model trained on SVOX train, validated on SF-XS val may not generalize well
3. Threshold (0.410) may be too low for this dataset

---

## Comparison with Other Datasets

| Dataset | Baseline R@1 | Adaptive R@1 | Hard Queries Detected | Time Savings | Actually Wrong |
|---------|--------------|--------------|----------------------|--------------|----------------|
| **SF-XS test** | 63.1% | 69.8% | 25.4% (254/1000) | 74.6% | 36.9% (369/1000) |
| **Tokyo-XS test** | 65.1% | 65.1% | 0.0% (0/315) | 100% | 34.9% (110/315) |
| **SVOX test** | **96.3%** | **96.3%** | **0.0% (0/14278)** | **100%** | **3.7% (524/14278)** |

**Key Observations:**
- **SVOX test is the easiest dataset** (96.3% baseline R@1)
- **Very few wrong queries** (only 3.7% vs 36.9% for SF-XS)
- **Model behavior**: Same as Tokyo-XS (all queries predicted as easy)
- **High baseline performance** suggests re-ranking may have limited benefit

---

## Files Generated:
- **Features**: `features_svox_test_improved.npz` (14,278 queries, 8 features)
- **Model predictions**: `logreg_easy_svox_test.npz`
- **Hard queries list**: `hard_queries_svox_test_logreg_easy.txt` (empty - 0 queries)
- **VPR logs**: `log_svox_test/2025-12-18_16-01-59/`

---

*Evaluation completed: 2025-12-18*
*Full re-ranking results: Pending (would take ~37 hours for 14,278 queries)*

