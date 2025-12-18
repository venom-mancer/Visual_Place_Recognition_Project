# Complete Results Comparison - All Test Datasets

## Performance Comparison Table

| Dataset | Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|---------|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **SF-XS test** | Baseline | 63.1% | 74.8% | 78.6% | 81.4% | 0% | 100% | - |
| **SF-XS test** | Full Re-ranking | **77.4%** | **80.3%** | **80.9%** | **81.4%** | 100% | 0% | **+14.3%** |
| **SF-XS test** | Adaptive (LogReg Easy) | **69.8%** | **77.7%** | **79.5%** | **81.4%** | **25.4%** | **74.6%** | **+6.7%** |
| | | | | | | | | |
| **Tokyo-XS test** | Baseline | 65.1% | 79.7% | 86.0% | 89.5% | 0% | 100% | - |
| **Tokyo-XS test** | Full Re-ranking | **83.2%** | **87.0%** | **88.3%** | **89.5%** | 100% | 0% | **+18.1%** |
| **Tokyo-XS test** | Adaptive (LogReg Easy) | **65.1%** | **79.7%** | **86.0%** | **89.5%** | **0.0%** | **100%** | **0.0%** |
| | | | | | | | | |
| **SVOX test** | Baseline | **96.3%** | **97.9%** | **98.3%** | **98.7%** | 0% | 100% | - |
| **SVOX test** | Full Re-ranking | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | 100% | 0% | ⏳ *Pending* |
| **SVOX test** | Adaptive (LogReg Easy) | **96.3%** | **97.9%** | **98.3%** | **98.7%** | **0.0%** | **100%** | **0.0%** |

---

## Summary Statistics

### Dataset Characteristics

| Dataset | Queries | Baseline R@1 | Actually Wrong | Hard Queries Detected | Time Savings |
|---------|---------|--------------|----------------|----------------------|--------------|
| **SF-XS test** | 1,000 | 63.1% | 36.9% (369) | 25.4% (254) | 74.6% |
| **Tokyo-XS test** | 315 | 65.1% | 34.9% (110) | 0.0% (0) | 100% |
| **SVOX test** | 14,278 | **96.3%** | **3.7% (524)** | **0.0% (0)** | **100%** |

### Performance Gains

| Dataset | Full Re-ranking Gain | Adaptive Gain | Adaptive vs Full Re-ranking |
|--------|---------------------|---------------|----------------------------|
| **SF-XS test** | +14.3% R@1 | +6.7% R@1 | -7.6% R@1 |
| **Tokyo-XS test** | +18.1% R@1 | 0.0% R@1 | **-18.1% R@1** ⚠️ |
| **SVOX test** | ⏳ *Pending* | 0.0% R@1 | ⏳ *Pending* |

---

## Key Findings

### 1. SF-XS Test (Best Adaptive Performance)
- ✅ **Model works well**: Detects 25.4% hard queries
- ✅ **Good balance**: +6.7% R@1 improvement with 74.6% time savings
- ✅ **Reasonable trade-off**: -7.6% vs full re-ranking, but saves 74.6% time

### 2. Tokyo-XS Test (Model Too Conservative)
- ⚠️ **Model too conservative**: Detects 0% hard queries (all predicted as easy)
- ⚠️ **Missed opportunity**: Full re-ranking shows +18.1% potential gain
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline

### 3. SVOX Test (Easiest Dataset)
- ✅ **Very high baseline**: 96.3% R@1 (easiest dataset)
- ⚠️ **Model too conservative**: Detects 0% hard queries (all predicted as easy)
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline
- ℹ️ **Few wrong queries**: Only 3.7% (524/14,278) are actually wrong

---

## Model Behavior Analysis

### Prediction Patterns

| Dataset | Min Prob | Max Prob | Mean Prob | Median Prob | Threshold | Hard Queries |
|---------|----------|----------|-----------|-------------|-----------|--------------|
| **SF-XS test** | - | - | - | - | 0.410 | 25.4% (254) |
| **Tokyo-XS test** | 0.670 | 1.000 | 0.999 | 1.000 | 0.410 | 0.0% (0) |
| **SVOX test** | 0.700 | 1.000 | 1.000 | 1.000 | 0.410 | 0.0% (0) |

### Why Model Fails on Tokyo-XS and SVOX?

1. **Threshold too low**: 0.410 optimized for SF-XS val, may not work for other datasets
2. **Feature distribution shift**: Features on different datasets may have different distributions
3. **Dataset difficulty**: Tokyo-XS and SVOX may have different characteristics than SF-XS

---

## Recommendations

### 1. Dataset-Specific Thresholds
- **SF-XS**: 0.410 (current, works well)
- **Tokyo-XS**: May need 0.6-0.7 (higher threshold)
- **SVOX**: May need 0.7-0.8 (higher threshold)

### 2. Model Retraining
- Include multiple datasets in validation set
- Or train separate models per dataset
- Use cross-dataset validation

### 3. Feature Normalization
- Dataset-specific feature normalization
- Or use more robust features that generalize better

---

## Efficiency Analysis

| Dataset | Adaptive Time Savings | Performance vs Baseline | Efficiency Score |
|---------|---------------------|------------------------|-------------------|
| **SF-XS test** | 74.6% | +6.7% R@1 | 0.26 |
| **Tokyo-XS test** | 100% | 0.0% R@1 | 0.00 |
| **SVOX test** | 100% | 0.0% R@1 | 0.00 |

**Efficiency Score = (R@1 - Baseline) / (1 - Time Savings)**

---

## Conclusion

The adaptive re-ranking approach works well on **SF-XS test** but struggles on **Tokyo-XS** and **SVOX** due to:
1. Model being too conservative (predicting all queries as easy)
2. Threshold not optimal for these datasets
3. Potential feature distribution shift

**Best performance**: SF-XS test (+6.7% R@1, 74.6% time savings)  
**Most efficient**: Tokyo-XS and SVOX (100% time savings, but no performance gain)

---

*Evaluation completed: 2025-12-18*
*Full re-ranking for SVOX test: Pending (would take ~37 hours)*

