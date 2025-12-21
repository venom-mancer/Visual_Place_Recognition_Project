# Model Change: From Easy to Hard Probability

## Summary

The logistic regression model has been updated to predict **hard query probability** instead of easy query probability.

---

## What Changed

### Before (Easy Probability)
- **Target**: `easy_score` (1 = easy/correct, 0 = hard/wrong)
- **Model Output**: `P(easy)` - Probability of being easy
- **Decision**: `if P(easy) >= threshold → Easy → Skip re-ranking`

### After (Hard Probability)
- **Target**: `hard_score` (1 = hard/wrong, 0 = easy/correct)
- **Model Output**: `P(hard)` - Probability of being hard
- **Decision**: `if P(hard) >= threshold → Hard → Apply re-ranking`

---

## Code Changes

### Training Script (`stage_3_train_logreg_easy_queries.py`)

**Before:**
```python
labels = features_dict["labels"]  # 1 = correct, 0 = wrong
easy_score = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
return X, easy_score, feature_names
```

**After:**
```python
labels = features_dict["labels"]  # 1 = correct, 0 = wrong
hard_score = (1 - labels).astype("float32")  # 1 = hard/wrong, 0 = easy/correct
return X, hard_score, feature_names
```

### Application Script (`stage_4_apply_logreg_easy_queries.py`)

**Before:**
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
is_easy = probs >= optimal_threshold
is_hard = ~is_easy
```

**After:**
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being hard
is_hard = probs >= optimal_threshold  # If P(hard) >= threshold → Hard
is_easy = ~is_hard                    # If P(hard) < threshold → Easy
```

---

## Model Files

### New Model
- **File**: `models/logreg_hard_with_inliers.pkl`
- **Target**: `hard_score` (1 = hard/wrong, 0 = easy/correct)
- **Features**: 9 features (8 retrieval + `num_inliers_top1`)
- **Optimal Threshold**: 0.590 (learned from validation)

### Validation Results
- **Accuracy**: 95.1%
- **F1-Score**: 0.7852
- **Hard Query Detection**: 12.0% (960/7993 queries)
- **Time Savings**: 88.0% (skip re-ranking for easy queries)

---

## Example Usage

```python
# Load model
bundle = joblib.load("models/logreg_hard_with_inliers.pkl")
model = bundle["model"]
threshold = bundle["optimal_threshold"]  # 0.590

# Predict
probs = model.predict_proba(X)[:, 1]  # P(hard) for each query

# Decision
is_hard = probs >= threshold  # Queries to re-rank
is_easy = probs < threshold   # Queries to skip

# Example:
# Query A: P(hard) = 0.80 → Hard → Apply re-ranking
# Query B: P(hard) = 0.20 → Easy → Skip re-ranking
```

---

## Interpretation

| Probability | Meaning | Action |
|-------------|---------|--------|
| `P(hard) = 0.90` | Very likely hard | Apply re-ranking |
| `P(hard) = 0.50` | Moderately hard | Apply re-ranking (if threshold < 0.50) |
| `P(hard) = 0.10` | Likely easy | Skip re-ranking |

---

## Benefits

1. ✅ **More Intuitive**: Directly predicts what we care about (hard queries)
2. ✅ **Clearer Logic**: High probability = hard = re-rank
3. ✅ **Same Performance**: Mathematically equivalent to easy prediction
4. ✅ **Better Alignment**: Matches the original task definition (identify hard queries)

---

## Notes

- The model file name still contains "easy_queries" in the script name, but the functionality now predicts hard queries.
- The threshold (0.590) is learned from the validation set to maximize F1-score.
- The model uses 9 features including `num_inliers_top1` (if available, otherwise zeros).


