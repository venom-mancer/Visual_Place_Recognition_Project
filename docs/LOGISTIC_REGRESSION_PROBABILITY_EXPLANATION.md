# Logistic Regression Probability Explanation

## Question
**What probability does our Logistic Regression return?**
- The probability of **soft/easy queries**?
- Or the probability of **hard queries**?

## Answer: **Probability of EASY (Soft) Queries**

The logistic regression model returns the **probability that a query is EASY** (soft/correct).

---

## Model Definition

### Target Variable: `easy_score`
- **1 = Easy/Correct**: Top-1 retrieval is correct (within distance threshold)
- **0 = Hard/Wrong**: Top-1 retrieval is wrong (outside distance threshold)

### Model Output
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
```

**`probs`** = Probability that the query is **EASY** (class 1)

---

## How It Works

### Probability Interpretation:
- **High probability (e.g., 0.9)**: Query is likely **EASY** → Skip re-ranking
- **Low probability (e.g., 0.2)**: Query is likely **HARD** → Apply re-ranking

### Classification Rule:
```python
is_easy = probs >= optimal_threshold  # If prob >= threshold → Easy
is_hard = ~is_easy                    # If prob < threshold → Hard
```

### Example:
- **Threshold**: 0.170 (from model with inliers)
- **Query A**: `probs = 0.95` → `is_easy = True` → **Skip re-ranking** ✅
- **Query B**: `probs = 0.10` → `is_easy = False` → **Apply re-ranking** ✅

---

## Why This Design?

### Original Task Approach:
The original task suggests using inliers to identify **hard queries** (low inliers = hard).

### Our Implementation:
We predict **easy queries** instead because:
1. ✅ **More intuitive**: High confidence = easy = skip matching
2. ✅ **Better for time savings**: Identify queries we can skip
3. ✅ **Same result**: `is_hard = ~is_easy` gives us hard queries

### Mathematical Equivalence:
```
P(easy) = 1 - P(hard)
P(hard) = 1 - P(easy)

If P(easy) >= threshold → Easy → Skip
If P(easy) < threshold → Hard → Re-rank
```

---

## Code Reference

### Training (`stage_3_train_logreg_easy_queries.py`):
```python
# Target: Easy score (1 = easy/correct, 0 = hard/wrong)
labels = features_dict["labels"]  # 1 = correct, 0 = wrong
easy_score = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
```

### Application (`stage_4_apply_logreg_easy_queries.py`):
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
is_easy = probs >= optimal_threshold
is_hard = ~is_easy
```

---

## Summary

| Aspect | Value |
|--------|-------|
| **Model Predicts** | Probability of **EASY** queries |
| **Class 1** | Easy/Correct (skip re-ranking) |
| **Class 0** | Hard/Wrong (apply re-ranking) |
| **High Probability** | Query is easy → Skip matching |
| **Low Probability** | Query is hard → Apply matching |
| **Threshold** | Learned from validation (e.g., 0.170) |

---

## Visual Example

```
Probability Distribution:
┌─────────────────────────────────────┐
│  P(easy) = 0.95  →  Easy  →  Skip   │
│  P(easy) = 0.80  →  Easy  →  Skip   │
│  P(easy) = 0.50  →  Easy  →  Skip   │
│  P(easy) = 0.15  →  Hard  →  Re-rank│ ← Threshold = 0.170
│  P(easy) = 0.05  →  Hard  →  Re-rank│
└─────────────────────────────────────┘
```

**Key Point**: The model returns `P(easy)`, not `P(hard)`.


## Question
**What probability does our Logistic Regression return?**
- The probability of **soft/easy queries**?
- Or the probability of **hard queries**?

## Answer: **Probability of EASY (Soft) Queries**

The logistic regression model returns the **probability that a query is EASY** (soft/correct).

---

## Model Definition

### Target Variable: `easy_score`
- **1 = Easy/Correct**: Top-1 retrieval is correct (within distance threshold)
- **0 = Hard/Wrong**: Top-1 retrieval is wrong (outside distance threshold)

### Model Output
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
```

**`probs`** = Probability that the query is **EASY** (class 1)

---

## How It Works

### Probability Interpretation:
- **High probability (e.g., 0.9)**: Query is likely **EASY** → Skip re-ranking
- **Low probability (e.g., 0.2)**: Query is likely **HARD** → Apply re-ranking

### Classification Rule:
```python
is_easy = probs >= optimal_threshold  # If prob >= threshold → Easy
is_hard = ~is_easy                    # If prob < threshold → Hard
```

### Example:
- **Threshold**: 0.170 (from model with inliers)
- **Query A**: `probs = 0.95` → `is_easy = True` → **Skip re-ranking** ✅
- **Query B**: `probs = 0.10` → `is_easy = False` → **Apply re-ranking** ✅

---

## Why This Design?

### Original Task Approach:
The original task suggests using inliers to identify **hard queries** (low inliers = hard).

### Our Implementation:
We predict **easy queries** instead because:
1. ✅ **More intuitive**: High confidence = easy = skip matching
2. ✅ **Better for time savings**: Identify queries we can skip
3. ✅ **Same result**: `is_hard = ~is_easy` gives us hard queries

### Mathematical Equivalence:
```
P(easy) = 1 - P(hard)
P(hard) = 1 - P(easy)

If P(easy) >= threshold → Easy → Skip
If P(easy) < threshold → Hard → Re-rank
```

---

## Code Reference

### Training (`stage_3_train_logreg_easy_queries.py`):
```python
# Target: Easy score (1 = easy/correct, 0 = hard/wrong)
labels = features_dict["labels"]  # 1 = correct, 0 = wrong
easy_score = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
```

### Application (`stage_4_apply_logreg_easy_queries.py`):
```python
probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
is_easy = probs >= optimal_threshold
is_hard = ~is_easy
```

---

## Summary

| Aspect | Value |
|--------|-------|
| **Model Predicts** | Probability of **EASY** queries |
| **Class 1** | Easy/Correct (skip re-ranking) |
| **Class 0** | Hard/Wrong (apply re-ranking) |
| **High Probability** | Query is easy → Skip matching |
| **Low Probability** | Query is hard → Apply matching |
| **Threshold** | Learned from validation (e.g., 0.170) |

---

## Visual Example

```
Probability Distribution:
┌─────────────────────────────────────┐
│  P(easy) = 0.95  →  Easy  →  Skip   │
│  P(easy) = 0.80  →  Easy  →  Skip   │
│  P(easy) = 0.50  →  Easy  →  Skip   │
│  P(easy) = 0.15  →  Hard  →  Re-rank│ ← Threshold = 0.170
│  P(easy) = 0.05  →  Hard  →  Re-rank│
└─────────────────────────────────────┘
```

**Key Point**: The model returns `P(easy)`, not `P(hard)`.


