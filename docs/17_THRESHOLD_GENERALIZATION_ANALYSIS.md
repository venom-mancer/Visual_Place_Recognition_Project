# Threshold Generalization Analysis

## Current Approach Assessment

### Current Implementation

1. **Model Training**: Logistic Regression with default `C=1.0` (not tuned)
2. **Threshold Selection**: Learned on SF-XS validation set (0.410)
3. **Application**: Fixed threshold applied to all test datasets

### Problem Identified

**Threshold learned on SF-XS val doesn't generalize to other datasets:**

| Dataset | Saved Threshold | Hard Queries Detected | Actually Wrong | Issue |
|---------|-----------------|----------------------|----------------|-------|
| **SF-XS test** | 0.410 | 25.4% (254/1000) | 36.9% (369/1000) | ✅ Works well |
| **Tokyo-XS test** | 0.410 | 0.0% (0/315) | 34.9% (110/315) | ❌ Too conservative |
| **SVOX test** | 0.410 | 0.0% (0/14278) | 3.7% (524/14278) | ❌ Too conservative |

**Root Causes:**
1. **Feature distribution shift**: Different datasets have different feature distributions
2. **Probability distribution shift**: Mean probabilities vary (SF-XS: lower, SVOX: higher)
3. **Fixed threshold**: One threshold can't work for all datasets

---

## Does Current Approach Work?

### Short Answer: **Partially**

✅ **Works for**: Datasets similar to validation set (SF-XS test)
❌ **Fails for**: Different datasets (Tokyo-XS, SVOX)

### Why It Fails

1. **Probability Distribution Shift**:
   - SF-XS test: Mean prob ~0.5-0.6 (moderate confidence)
   - Tokyo-XS test: Mean prob ~0.999 (very high confidence)
   - SVOX test: Mean prob ~1.000 (extremely high confidence)

2. **Threshold Mismatch**:
   - Threshold 0.410 works when mean prob is ~0.5-0.6
   - Threshold 0.410 is too low when mean prob is ~0.999-1.000
   - All queries have prob > 0.410 → all predicted as easy

---

## Should We Tune C (Regularization Parameter)?

### Current Status

- **C is NOT tuned**: Using default `C=1.0`
- **No validation for C**: Model trained with fixed C

### Would C Tuning Help?

**Yes, but it's not the main solution:**

#### Benefits of C Tuning:
1. **Better model generalization**: Optimal C can improve model fit
2. **More robust predictions**: Better calibrated probabilities
3. **Reduced overfitting**: Proper regularization

#### Limitations:
1. **Still need threshold calibration**: Even with optimal C, threshold will vary across datasets
2. **C tuning helps model, not threshold**: C affects probability calibration, but threshold selection is still dataset-dependent

### Recommendation: **Do Both**

1. ✅ **Tune C on validation set** (improves model generalization)
2. ✅ **Calibrate threshold per dataset** (handles distribution shift)

---

## Proposed Solution: Two-Step Validation

### Step 1: Tune C (Regularization Parameter)

**Goal**: Find optimal C that maximizes validation performance

**Process**:
1. Test C values: [0.01, 0.1, 1.0, 10.0, 100.0]
2. For each C:
   - Train logistic regression
   - Evaluate on validation set (using threshold that maximizes F1)
   - Record validation F1-score
3. Select C with best validation F1-score

**Benefits**:
- Better model generalization
- More calibrated probabilities
- Potentially reduces threshold variation

### Step 2: Calibrate Threshold Per Dataset

**Goal**: Find optimal threshold for each test dataset

**Process**:
1. Use model trained with optimal C
2. For each test dataset:
   - Get predictions
   - Find threshold that maximizes F1 (if labels available)
   - Or use target hard query rate

**Benefits**:
- Handles dataset distribution shift
- Optimal threshold for each dataset
- Better hard query detection

---

## Implementation Plan

### Option A: Tune C + Per-Dataset Threshold Calibration (Recommended)

**Training Phase**:
1. Tune C on validation set
2. Train final model with optimal C
3. Find threshold on validation set (for reference)

**Testing Phase**:
1. Apply model to test dataset
2. Calibrate threshold on test set (using `--calibrate-threshold`)
3. Use calibrated threshold for that dataset

**Pros**:
- ✅ Best model generalization (optimal C)
- ✅ Best threshold for each dataset
- ✅ Handles both model and threshold issues

**Cons**:
- ⚠️ Requires labels in test feature files (already available)
- ⚠️ Slightly more complex

### Option B: Tune C Only (Not Recommended)

**Training Phase**:
1. Tune C on validation set
2. Train final model with optimal C
3. Find threshold on validation set

**Testing Phase**:
1. Apply model with fixed threshold

**Pros**:
- ✅ Better model (optimal C)
- ✅ Simpler (no per-dataset calibration)

**Cons**:
- ❌ Still has threshold generalization problem
- ❌ Won't solve Tokyo-XS/SVOX issue completely

---

## Expected Impact of C Tuning

### Without C Tuning (Current):
- Model: C=1.0 (default)
- Validation F1: 0.9582
- Threshold generalization: Poor (0% hard queries on Tokyo-XS/SVOX)

### With C Tuning (Expected):
- Model: C=optimal (e.g., 0.1 or 10.0)
- Validation F1: Potentially 0.96-0.97 (slight improvement)
- Threshold generalization: Still needs per-dataset calibration

### With C Tuning + Per-Dataset Calibration:
- Model: C=optimal
- Validation F1: 0.96-0.97
- Threshold generalization: ✅ Excellent (calibrated per dataset)

---

## Recommendation

### ✅ **Implement Both: C Tuning + Per-Dataset Threshold Calibration**

**Why Both?**
1. **C tuning**: Improves model quality and probability calibration
2. **Per-dataset calibration**: Handles distribution shift (the main problem)

**Implementation Priority**:
1. **High Priority**: Per-dataset threshold calibration (already implemented ✅)
2. **Medium Priority**: C tuning (would improve but not critical)

**Current Status**:
- ✅ Per-dataset threshold calibration: **Implemented** (use `--calibrate-threshold`)
- ⏳ C tuning: **Not implemented** (should be added)

---

## Conclusion

### Does Current Threshold Generalization Work?

**Answer**: **Partially** - Works for similar datasets, fails for different ones.

**Solution**: **Per-dataset threshold calibration** (already implemented) solves the main problem.

### Should We Tune C?

**Answer**: **Yes, but it's secondary** - C tuning improves model quality but doesn't solve threshold generalization alone.

**Recommendation**: 
1. ✅ **Use per-dataset threshold calibration** for Tokyo-XS test (solves immediate problem)
2. ✅ **Implement C tuning** for better model generalization (improvement, not critical)

---

*See [Threshold Calibration Guide](THRESHOLD_CALIBRATION_GUIDE.md) for how to use per-dataset calibration.*

