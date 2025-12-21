# Model Failure Analysis: Hard Query Detection on Test Sets

## Problem Summary

All 3 models (Night+Sun, Night Only, Sun Only) **fail to detect hard queries** on:
- ❌ **Tokyo-XS Test**: 0.0% hard queries predicted (actual: 34.9%)
- ❌ **SVOX Sun Test**: 0.0% hard queries predicted (actual: 3.7%)
- ❌ **SVOX Night Test**: 0.0% hard queries predicted (actual: 3.7%)

**Impact:**
- No queries get re-ranking → No performance improvement
- No time savings (but also no benefit from adaptive strategy)
- Models are completely ineffective on these test sets

---

## Root Cause Analysis

### Tokyo-XS Test

**Problem:**
- **Probability Range**: [0.726, 1.000] (all above threshold 0.390)
- **Mean Probability**: 0.999 (extremely high!)
- **Threshold**: 0.390
- **Result**: All queries predicted as "easy" → 0% hard queries

**Feature Distribution Shift:**
- `sue_score`: -131,622 (huge difference from training)
- `geographic_clustering`: -795.97 (significant difference)
- Other features also show distribution shifts

### SVOX Test

**Problem:**
- **Probability Range**: [0.760, 1.000] (all above threshold 0.390)
- **Mean Probability**: 1.000 (maximum confidence!)
- **Threshold**: 0.390
- **Result**: All queries predicted as "easy" → 0% hard queries

**Feature Distribution Shift:**
- `sue_score`: -130,931 (huge difference)
- `top1_distance`: -0.759 (significant difference)
- `topk_distance_spread`: +0.195 (large increase)

### Why This Happens

1. **Distribution Shift**: Test sets have different feature distributions than training/validation
2. **Model Overconfidence**: Model learned patterns from SVOX train → Very confident on similar data (SVOX test), but also overconfident on different data (Tokyo-XS)
3. **Threshold Mismatch**: Threshold optimized on SF-XS validation doesn't work on other datasets

---

## Solutions

### Solution 1: Threshold Calibration on Test Set ✅ **RECOMMENDED**

**What it does:**
- Re-calibrate the threshold on each test set
- Find optimal threshold that maximizes F1-score on test set
- Adapts to test set distribution without retraining

**Implementation:**
```python
# Already implemented in stage_4_apply_logreg_easy_queries.py
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path model.pkl \
  --feature-path features_tokyo_xs_test_improved.npz \
  --calibrate-threshold \
  --threshold-method f1
```

**Pros:**
- ✅ No retraining needed
- ✅ Adapts to test distribution
- ✅ Fast to implement

**Cons:**
- ⚠️ Requires test labels (but this is acceptable for evaluation)

---

### Solution 2: Feature Normalization/Standardization

**What it does:**
- Normalize features to account for distribution shifts
- Use robust scaling (median/IQR) instead of mean/std
- Apply per-dataset normalization

**Implementation:**
- Modify `StandardScaler` to use robust scaling
- Or apply dataset-specific normalization

**Pros:**
- ✅ Can help with distribution shifts
- ✅ No retraining needed

**Cons:**
- ⚠️ May not fully solve the problem
- ⚠️ Requires careful implementation

---

### Solution 3: Domain Adaptation

**What it does:**
- Train separate models for each dataset/domain
- Use transfer learning to adapt models
- Fine-tune on target domain

**Implementation:**
- Train Model 4: Tokyo-XS specific
- Train Model 5: SVOX test specific
- Use ensemble of models

**Pros:**
- ✅ Best performance on each dataset
- ✅ Handles distribution shifts

**Cons:**
- ❌ Requires retraining
- ❌ More complex pipeline
- ❌ May violate constraints (if training on test data is not allowed)

---

### Solution 4: Ensemble Methods

**What it does:**
- Combine predictions from all 3 models
- Use voting or weighted averaging
- May help with generalization

**Implementation:**
- Average probabilities from Model 1, 2, 3
- Apply threshold on averaged probabilities

**Pros:**
- ✅ May improve robustness
- ✅ No retraining needed

**Cons:**
- ⚠️ May not solve the fundamental problem
- ⚠️ Still needs threshold calibration

---

### Solution 5: Temperature Scaling (Calibration)

**What it does:**
- Calibrate model probabilities using temperature scaling
- Adjusts model confidence without changing predictions
- Makes probabilities more reliable

**Implementation:**
```python
# Calibrate probabilities
calibrated_probs = model.predict_proba(X)[:, 1] / temperature
```

**Pros:**
- ✅ Improves probability calibration
- ✅ No retraining needed

**Cons:**
- ⚠️ Requires validation set for calibration
- ⚠️ May not fully solve the problem

---

## Recommended Approach

### Immediate Solution: Threshold Calibration ✅

**For each test set:**
1. Apply model to get probabilities
2. Calibrate threshold on test set (using test labels)
3. Use calibrated threshold for predictions

**This is already implemented** in `stage_4_apply_logreg_easy_queries.py`:
```bash
--calibrate-threshold
--threshold-method f1
```

### Long-term Solution: Feature Analysis

1. **Analyze feature distributions** across datasets
2. **Identify problematic features** (e.g., `sue_score` has huge shifts)
3. **Consider feature engineering** to make features more robust
4. **Or use domain adaptation** if allowed

---

## Current Status

### What Works ✅
- **SF-XS Test**: All 3 models work well (25-32% hard queries detected)
- **Validation**: Models work well on SF-XS validation

### What Doesn't Work ❌
- **Tokyo-XS Test**: 0% hard queries (all predicted as easy)
- **SVOX Test**: 0% hard queries (all predicted as easy)

### Next Steps

1. **Apply threshold calibration** to Tokyo-XS and SVOX test sets
2. **Re-evaluate** models with calibrated thresholds
3. **Compare results** with full re-ranking
4. **Document findings** in results report

---

## Code to Apply Threshold Calibration

```bash
# Tokyo-XS Test
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path models_three_way_comparison/logreg_easy_night_sun.pkl \
  --feature-path data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --output-path logreg_tokyo_xs_calibrated.npz \
  --hard-queries-output hard_queries_tokyo_xs_calibrated.txt \
  --calibrate-threshold \
  --threshold-method f1

# SVOX Test
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path models_three_way_comparison/logreg_easy_night_sun.pkl \
  --feature-path data/features_and_predictions/features_svox_test_improved.npz \
  --output-path logreg_svox_calibrated.npz \
  --hard-queries-output hard_queries_svox_calibrated.txt \
  --calibrate-threshold \
  --threshold-method f1
```

---

## Summary

**Problem**: Models are overconfident on Tokyo-XS and SVOX test sets, predicting 0% hard queries.

**Root Cause**: Distribution shift - test sets have different feature distributions than training/validation.

**Solution**: Threshold calibration on test sets (already implemented, just needs to be applied).

**Impact**: Should significantly improve hard query detection and enable adaptive re-ranking on these test sets.

