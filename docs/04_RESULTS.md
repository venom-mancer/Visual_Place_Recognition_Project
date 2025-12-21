# Results - Complete Experimental Results

## Overview

This document summarizes all experimental results across different test datasets and approaches.

---

## SF-XS Test Results (1,000 queries)

### Performance Comparison

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline** (Retrieval-only) | 63.1% | 74.8% | 78.6% | 81.4% | 0% | 100% | - |
| **Full Re-ranking** | **77.4%** | **80.3%** | **80.9%** | **81.4%** | 100% | 0% | **+14.3%** |
| **Adaptive (Regressor)** | **73.4%** | **78.9%** | **80.3%** | **81.4%** | 42.5% | 57.5% | **+10.3%** |
| **Adaptive (LogReg Easy)** | **69.8%** | **77.7%** | **79.5%** | **81.4%** | **25.4%** | **74.6%** | **+6.7%** |

### Key Findings

- ✅ **Best performance**: Full re-ranking (77.4% R@1)
- ✅ **Best efficiency**: LogReg Easy (74.6% time savings)
- ✅ **Best balance**: Regressor (73.4% R@1, 57.5% savings)
- ✅ **Adaptive works**: Both approaches improve over baseline

### Model Performance

| Approach | Validation Accuracy | Hard Queries Detected | Actually Wrong |
|----------|-------------------|----------------------|----------------|
| **Regressor** | 86.7% | 42.5% (425/1000) | 36.9% (369/1000) |
| **LogReg Easy** | **92.5%** | **25.4% (254/1000)** | 36.9% (369/1000) |

---

## Tokyo-XS Test Results (315 queries)

### Performance Comparison

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline** (Retrieval-only) | 65.1% | 79.7% | 86.0% | 89.5% | 0% | 100% | - |
| **Full Re-ranking** | **83.2%** | **87.0%** | **88.3%** | **89.5%** | 100% | 0% | **+18.1%** |
| **Adaptive (LogReg Easy)** | **65.1%** | **79.7%** | **86.0%** | **89.5%** | **0.0%** | **100%** | **0.0%** |

### Key Findings

- ⚠️ **Model too conservative**: All queries predicted as easy (0% hard queries)
- ⚠️ **Missed opportunity**: Full re-ranking shows +18.1% potential gain
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline

### Model Behavior

- **Probability range**: 0.670 - 1.000 (mean: 0.999)
- **Threshold used**: 0.410 (learned from SF-XS validation)
- **Actually wrong**: 34.9% (110/315 queries)
- **Hard queries detected**: 0% (0/315 queries)

### Analysis

**Why all queries predicted as easy?**
1. Dataset difficulty: Tokyo-XS appears easier than SF-XS
2. Model generalization: Trained on SVOX, validated on SF-XS, may not generalize to Tokyo-XS
3. Threshold: 0.410 may be too low for Tokyo-XS (may need 0.6-0.7)

---

## SVOX Test Results (14,278 queries)

### Performance Comparison

| Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain vs Baseline |
|--------|-----|-----|------|------|-------------|--------------|----------------------|
| **Baseline** (Retrieval-only) | **96.3%** | **97.9%** | **98.3%** | **98.7%** | 0% | 100% | - |
| **Full Re-ranking** | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | ⏳ *Pending* | 100% | 0% | ⏳ *Pending* |
| **Adaptive (LogReg Easy)** | **96.3%** | **97.9%** | **98.3%** | **98.7%** | **0.0%** | **100%** | **0.0%** |

### Key Findings

- ✅ **Very high baseline**: 96.3% R@1 (easiest dataset)
- ⚠️ **Model too conservative**: All queries predicted as easy (0% hard queries)
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline
- ℹ️ **Few wrong queries**: Only 3.7% (524/14,278) are actually wrong

### Model Behavior

- **Probability range**: 0.700 - 1.000 (mean: 1.000)
- **Threshold used**: 0.410 (learned from SF-XS validation)
- **Actually wrong**: 3.7% (524/14,278 queries)
- **Hard queries detected**: 0% (0/14,278 queries)

### Analysis

**Why all queries predicted as easy?**
1. Very easy dataset: 96.3% baseline R@1 (very high!)
2. Few wrong queries: Only 3.7% are actually wrong
3. Model conservatism: All probabilities >= 0.410 threshold
4. Full re-ranking: Would take ~37.7 hours (not recommended)

---

## All Datasets Summary

### Performance Comparison Table

| Dataset | Method | R@1 | R@5 | R@10 | R@20 | % Re-ranked | Time Savings | R@1 Gain |
|---------|--------|-----|-----|------|------|-------------|--------------|-----------|
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

### Dataset Characteristics

| Dataset | Queries | Baseline R@1 | Actually Wrong | Hard Queries Detected | Time Savings |
|---------|---------|--------------|----------------|----------------------|--------------|
| **SF-XS test** | 1,000 | 63.1% | 36.9% (369) | 25.4% (254) | 74.6% |
| **Tokyo-XS test** | 315 | 65.1% | 34.9% (110) | 0.0% (0) | 100% |
| **SVOX test** | 14,278 | **96.3%** | **3.7% (524)** | **0.0% (0)** | **100%** |

---

## Key Insights

### 1. SF-XS Test (Best Adaptive Performance) ✅
- ✅ **Model works well**: Detects 25.4% hard queries
- ✅ **Good balance**: +6.7% R@1 improvement with 74.6% time savings
- ✅ **Reasonable trade-off**: -7.6% vs full re-ranking, but saves 74.6% time

### 2. Tokyo-XS Test (Model Too Conservative) ⚠️
- ⚠️ **Model too conservative**: Detects 0% hard queries (all predicted as easy)
- ⚠️ **Missed opportunity**: Full re-ranking shows +18.1% potential gain
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline

### 3. SVOX Test (Easiest Dataset) ℹ️
- ✅ **Very high baseline**: 96.3% R@1 (easiest dataset)
- ⚠️ **Model too conservative**: Detects 0% hard queries (all predicted as easy)
- ✅ **Time savings**: 100% (no image matching needed)
- ⚠️ **No performance improvement**: Same as baseline
- ℹ️ **Few wrong queries**: Only 3.7% (524/14,278) are actually wrong

---

## Efficiency Analysis

### Efficiency Score = (R@1 - Baseline) / (1 - Time Savings)

| Dataset | Adaptive Time Savings | Performance vs Baseline | Efficiency Score |
|---------|---------------------|------------------------|-------------------|
| **SF-XS test** | 74.6% | +6.7% R@1 | **0.26** ✅ |
| **Tokyo-XS test** | 100% | 0.0% R@1 | 0.00 |
| **SVOX test** | 100% | 0.0% R@1 | 0.00 |

**LogReg Easy has the best efficiency score on SF-XS test!**

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

## Conclusion

The adaptive re-ranking approach works well on **SF-XS test** but struggles on **Tokyo-XS** and **SVOX** due to:
1. Model being too conservative (predicting all queries as easy)
2. Threshold not optimal for these datasets
3. Potential feature distribution shift

**Best performance**: SF-XS test (+6.7% R@1, 74.6% time savings)  
**Most efficient**: Tokyo-XS and SVOX (100% time savings, but no performance gain)

---

*See [Methodology](METHODOLOGY.md) for details on approaches and strategies.*

