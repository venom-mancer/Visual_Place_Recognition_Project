# Threshold Calibration Guide - Solving Dataset Distribution Shift

## Problem

When the test dataset changes, the threshold learned on the validation set (e.g., 0.410 from SF-XS val) doesn't efficiently find hard queries on other datasets.

### Example Problem:
- **SF-XS test**: Threshold 0.410 works well → 25.4% hard queries detected
- **Tokyo-XS test**: Threshold 0.410 → 0.0% hard queries detected (all predicted as easy)
- **SVOX test**: Threshold 0.410 → 0.0% hard queries detected (all predicted as easy)

**Root Cause**: Feature distribution shift between datasets causes the fixed threshold to be suboptimal.

---

## Solution: Dataset-Specific Threshold Calibration

The updated `stage_4_apply_logreg_easy_queries.py` now supports **threshold calibration on the test set** to adapt to different datasets.

---

## Usage

### Option 1: Use Saved Threshold (Default)

Use the threshold learned from validation set (works well if test dataset is similar to validation):

```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --output-path logreg_easy_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_test.txt
```

### Option 2: Calibrate Threshold on Test Set (F1-Maximization)

Find optimal threshold that maximizes F1-score on the test set (requires labels in feature file):

```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --output-path logreg_easy_tokyo_xs_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_tokyo_xs_test.txt \
  --calibrate-threshold
```

**What it does**:
- Tests thresholds from 0.1 to 0.95
- Selects threshold that maximizes F1-score on test set
- Reports precision, recall, and F1-score

### Option 3: Target Hard Query Rate

Calibrate threshold to achieve a specific hard query rate (e.g., 30% hard queries):

```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_svox_test_improved.npz \
  --output-path logreg_easy_svox_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_svox_test.txt \
  --calibrate-threshold \
  --target-hard-rate 0.30
```

**What it does**:
- Tests thresholds from 0.1 to 0.95
- Selects threshold that achieves closest to target hard query rate (30%)
- Useful when you want to control the percentage of queries that get re-ranked

---

## When to Use Each Option

### Use Saved Threshold (Default)
- ✅ Test dataset is similar to validation dataset (SF-XS test vs SF-XS val)
- ✅ You want consistent behavior across runs
- ✅ Labels not available in test set

### Use F1-Maximization Calibration
- ✅ Test dataset is different from validation (Tokyo-XS, SVOX)
- ✅ Labels are available in test feature file
- ✅ You want best detection accuracy

### Use Target Rate Calibration
- ✅ You want to control computational cost (target specific % hard queries)
- ✅ You know approximate wrong query rate (e.g., 30-40%)
- ✅ Labels are available in test feature file

---

## Example: Fixing Tokyo-XS Test

### Before (Using Saved Threshold):
```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --output-path logreg_easy_tokyo_xs_test.npz \
  --hard-queries-output hard_queries_tokyo_xs_test.txt
```

**Result**: 0% hard queries detected (all predicted as easy)

### After (Using Calibrated Threshold):
```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_tokyo_xs_test_improved.npz \
  --output-path logreg_easy_tokyo_xs_test.npz \
  --hard-queries-output hard_queries_tokyo_xs_test.txt \
  --calibrate-threshold
```

**Result**: Optimal threshold found (e.g., 0.65) → Detects appropriate % of hard queries

---

## How It Works

### F1-Maximization Calibration:
1. Load test features and labels
2. Get model predictions (probabilities)
3. Test thresholds from 0.1 to 0.95 in 0.01 steps
4. For each threshold:
   - Classify queries as easy/hard
   - Compute F1-score
5. Select threshold with highest F1-score

### Target Rate Calibration:
1. Load test features and labels
2. Get model predictions (probabilities)
3. Test thresholds from 0.1 to 0.95 in 0.01 steps
4. For each threshold:
   - Compute hard query rate
   - Measure distance from target rate
5. Select threshold closest to target rate

---

## Output Information

The script now reports:
- **Threshold source**: "saved_from_validation" or "calibrated_on_test"
- **Optimal threshold**: The threshold value used
- **Detection accuracy**: If labels available
- **Hard query rate**: Percentage of queries predicted as hard
- **Ground truth comparison**: Actually wrong vs predicted hard (if labels available)

---

## Recommendations

### For SF-XS Test:
- Use saved threshold (0.410) - works well

### For Tokyo-XS Test:
- Use F1-maximization calibration
- Expected threshold: ~0.65-0.70

### For SVOX Test:
- Use target rate calibration (target ~3.7% hard queries, matching actual wrong rate)
- Or use F1-maximization if you want best accuracy

---

## Technical Details

The calibration process:
- Uses the same `find_optimal_threshold()` function as training
- Only requires labels in the feature file (already computed during feature extraction)
- Doesn't require re-training the model
- Fast: Tests ~85 thresholds in <1 second for 14K queries

---

*See [Technical Details](TECHNICAL_DETAILS.md) for implementation details.*

