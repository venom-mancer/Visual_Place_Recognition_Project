# How the Optimal Threshold is Chosen

## Overview

The optimal threshold is chosen **automatically during training** based on **F1-score maximization** on the validation set.

---

## Selection Criteria: F1-Score Maximization

### What is F1-Score?

F1-score is the **harmonic mean of Precision and Recall**:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

- **Precision**: Of all queries predicted as "easy", how many are actually easy?
- **Recall**: Of all queries that are actually easy, how many did we correctly predict as easy?
- **F1-Score**: Balances both precision and recall (single metric)

### Why F1-Score?

1. **Balanced metric**: Considers both false positives and false negatives
2. **Handles class imbalance**: Works well when easy queries (89.4%) >> hard queries (10.6%)
3. **Standard practice**: Common metric for binary classification
4. **Actionable**: Directly relates to model performance

---

## How It Works: Step-by-Step

### Step 1: Model Training
```python
# Train logistic regression on training data (SVOX train)
logreg.fit(X_train_scaled, y_train)
```

### Step 2: Validation Predictions
```python
# Get probabilities on validation set (SF-XS val)
y_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]  # Probability of being easy
```

### Step 3: Test Different Thresholds
```python
thresholds = np.arange(0.1, 0.95, 0.01)  # Test thresholds from 0.1 to 0.95

for threshold in thresholds:
    y_pred = (y_val_probs >= threshold).astype(int)  # Predict easy/hard
    f1 = f1_score(y_val, y_pred)  # Compute F1-score
    
    if f1 > best_f1:
        best_f1 = f1
        optimal_threshold = threshold
```

### Step 4: Select Best Threshold
```python
# The threshold with highest F1-score is chosen
optimal_threshold = 0.390  # Example: This threshold maximizes F1
```

---

## Code Location

**File**: `extension_6_1/stage_3_train_logreg_easy_queries.py`

**Function**: `find_optimal_threshold()` (lines 83-115)

```python
def find_optimal_threshold(y_true, y_probs, method="f1"):
    """
    Find optimal threshold on validation set.
    
    Args:
        y_true: True labels (1 = easy, 0 = hard)
        y_probs: Predicted probabilities (probability of being easy)
        method: "f1" (maximize F1) or "recall" (target recall rate)
    
    Returns:
        optimal_threshold: Best threshold value
        best_score: Best F1 or recall score
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score
```

**Called from**: `main()` function (lines 247-250)

```python
# Find optimal threshold on VALIDATION set (prevents overfitting)
optimal_threshold, best_score = find_optimal_threshold(
    y_val, y_val_probs, method=args.threshold_method  # Default: "f1"
)
```

---

## Why Validation Set (Not Training Set)?

**Critical**: The threshold is chosen on the **validation set**, NOT the training set.

### Reasons:
1. **Prevents overfitting**: Training set would give optimistic results
2. **Better generalization**: Validation set represents unseen data
3. **Standard practice**: Hyperparameters (including thresholds) are tuned on validation set
4. **Fair evaluation**: Test set remains completely untouched

### Data Split:
- **Training**: SVOX train → Train the model
- **Validation**: SF-XS val → Choose optimal threshold
- **Test**: SF-XS test, Tokyo-XS test, SVOX test → Final evaluation

---

## Threshold Selection Process

### What Happens:

1. **Model is trained** on training data (SVOX train)
2. **Model predicts probabilities** on validation data (SF-XS val)
3. **Different thresholds are tested** (0.1, 0.11, 0.12, ..., 0.95)
4. **For each threshold**:
   - Classify queries as easy/hard
   - Compute F1-score
5. **Threshold with highest F1-score** is selected
6. **Selected threshold is saved** with the model

### Example:

| Threshold | F1-Score | Classification Accuracy | Hard Queries |
|-----------|----------|------------------------|--------------|
| 0.30 | 0.9450 | 91.2% | 12.5% |
| 0.35 | 0.9520 | 91.8% | 10.2% |
| **0.390** | **0.9585** | **92.5%** | **9.0%** |
| 0.40 | 0.9570 | 92.3% | 8.5% |
| 0.45 | 0.9500 | 91.5% | 7.2% |

**Result**: Threshold 0.390 is chosen because it has the **highest F1-score (0.9585)**.

---

## Alternative Methods

The code supports two methods (controlled by `--threshold-method`):

### Method 1: F1-Maximization (Default) ✅
```python
--threshold-method f1
```
- **Goal**: Maximize F1-score
- **Use case**: Balance precision and recall
- **Result**: Threshold = 0.390 (F1 = 0.9585)

### Method 2: Recall-Maximization
```python
--threshold-method recall
```
- **Goal**: Maximize recall (catch all easy queries)
- **Use case**: Prioritize not missing easy queries
- **Result**: Lower threshold (more queries predicted as easy)

---

## Visual Representation

The validation threshold analysis plot shows:

1. **Panel 1**: F1-Score vs Threshold
   - Shows the F1-score curve
   - **Red star marks the peak** (optimal threshold)
   - This is why the threshold was chosen!

2. **Panel 2**: Classification Accuracy vs Threshold
   - Shows accuracy at each threshold
   - Optimal threshold marked

3. **Panel 3**: Adaptive VPR R@1 vs Threshold
   - Shows actual VPR performance
   - Compared with full re-ranking (ground-truth)

4. **Panel 4**: Combined view
   - F1-score and Adaptive R@1 together
   - Shows relationship between F1-max and performance

---

## Key Insight

**The threshold is NOT chosen to maximize VPR R@1 directly!**

Instead:
- ✅ **Chosen to maximize F1-score** (classification performance)
- ✅ **F1-score correlates with good VPR performance**
- ✅ **But F1 is computed on validation set** (faster, no need for image matching)

**Why this works:**
- Good classification (high F1) → Good hard query detection → Good adaptive performance
- We can't optimize VPR R@1 directly during training (requires expensive image matching)
- F1-score is a proxy that works well in practice

---

## Summary

| Aspect | Details |
|--------|---------|
| **Selection Criteria** | F1-score maximization |
| **Dataset Used** | Validation set (SF-XS val) |
| **Range Tested** | 0.1 to 0.95 (step 0.01) |
| **Method** | Grid search (try all thresholds) |
| **Saved With** | Model bundle (`.pkl` file) |
| **Applied To** | Test sets (automatically) |

**The optimal threshold (0.390) was chosen because it maximizes F1-score (0.9585) on the validation set.**

