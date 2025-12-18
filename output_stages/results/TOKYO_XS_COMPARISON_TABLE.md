# Tokyo-XS Test - Complete Results Comparison

## Performance Comparison Table

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline (Retrieval-only)** | 65.1% | 79.7% | 86.0% | 89.5% | 0% | 100% | - |
| **Full Re-ranking** | **83.2%** | **87.0%** | **88.3%** | **89.5%** | 100% | 0% | **+18.1%** |
| **Adaptive (LogReg Easy)** | **65.1%** | **79.7%** | **86.0%** | **89.5%** | **0.0%** | **100%** | **0.0%** |

---

## Performance Gains

| Metric | Full Re-ranking vs Baseline | Adaptive vs Baseline | Adaptive vs Full Re-ranking |
|--------|----------------------------|---------------------|----------------------------|
| **R@1** | **+18.1%** ✅ | 0.0% | **-18.1%** ⚠️ |
| **R@5** | **+7.3%** ✅ | 0.0% | **-7.3%** ⚠️ |
| **R@10** | **+2.3%** ✅ | 0.0% | **-2.3%** ⚠️ |
| **R@20** | 0.0% | 0.0% | 0.0% |

---

## Comparison with SF-XS Test

| Dataset | Baseline R@1 | Full Re-ranking R@1 | Adaptive R@1 | Hard Queries Detected | Time Savings |
|---------|--------------|---------------------|--------------|----------------------|-------------|
| **SF-XS test** | 63.1% | 77.4% | 69.8% | 25.4% (254/1000) | 74.6% |
| **Tokyo-XS test** | 65.1% | 83.2% | 65.1% | 0.0% (0/315) | 100% |

### Key Differences:
- **SF-XS**: Model detected 25.4% hard queries → +6.7% R@1 improvement, 74.6% time savings
- **Tokyo-XS**: Model detected 0% hard queries → 0% improvement, 100% time savings (but missed +18.1% potential gain)

---

## Analysis

### Model Behavior:
- **All queries predicted as easy**: 315/315 (100%)
- **Probability range**: 0.670 - 1.000 (mean: 0.999)
- **Threshold**: 0.410 (learned from SF-XS validation)
- **Result**: Model is extremely conservative on Tokyo-XS

### Performance Impact:
- **Full re-ranking**: Shows re-ranking is highly beneficial (+18.1% R@1)
- **Adaptive**: Missed opportunity - no queries detected as hard
- **Trade-off**: 100% time savings but missed 18.1% performance gain

### Recommendations:
1. **Dataset-specific threshold tuning** for Tokyo-XS
2. **Model retraining** with Tokyo-XS in validation set
3. **Feature analysis** to understand why all queries appear "easy"

---

*Evaluation completed: 2025-12-18*

