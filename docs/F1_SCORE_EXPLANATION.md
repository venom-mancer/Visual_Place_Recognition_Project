# What is F1-Score?

## Simple Definition

**F1-score** is a single number (0 to 1) that tells you how well your model performs at **both**:
- **Correctly identifying easy queries** (not missing them)
- **Not falsely labeling hard queries as easy** (avoiding mistakes)

**Higher F1-score = Better model performance**

---

## The Problem: Why Not Just Use Accuracy?

### Example Scenario

Imagine you have 1000 queries:
- **900 are easy** (89.4%)
- **100 are hard** (10.6%)

### Bad Model (High Accuracy, Poor Performance)

A model that predicts **everything as easy**:
- ✅ Correctly identifies: 900 easy queries
- ❌ Misses: 100 hard queries (all predicted as easy)
- **Accuracy**: 900/1000 = **90%** (looks good!)
- **But**: This model is **useless** - it never finds hard queries!

### Good Model (Balanced Performance)

A model that correctly identifies:
- ✅ 850 easy queries (correctly predicted as easy)
- ✅ 80 hard queries (correctly predicted as hard)
- ❌ 50 easy queries (incorrectly predicted as hard)
- ❌ 20 hard queries (incorrectly predicted as easy)
- **Accuracy**: 930/1000 = **93%**
- **But**: This model is **useful** - it finds most hard queries!

**Problem**: Accuracy alone doesn't tell the full story when classes are imbalanced!

---

## The Solution: Precision, Recall, and F1-Score

### Precision: "When I say easy, am I right?"

**Precision** = Of all queries I predicted as "easy", how many are actually easy?

```
Precision = True Positives / (True Positives + False Positives)
```

**Example:**
- Model predicts 800 queries as "easy"
- 750 are actually easy (True Positives)
- 50 are actually hard (False Positives - I was wrong!)
- **Precision** = 750 / (750 + 50) = **93.75%**

**High Precision** = When I predict "easy", I'm usually right ✅

---

### Recall: "Did I catch all the easy queries?"

**Recall** = Of all queries that are actually easy, how many did I correctly identify?

```
Recall = True Positives / (True Positives + False Negatives)
```

**Example:**
- There are 900 actually easy queries
- Model correctly identifies 750 as easy (True Positives)
- Model misses 150 easy queries (False Negatives - I missed them!)
- **Recall** = 750 / (750 + 150) = **83.3%**

**High Recall** = I catch most of the easy queries ✅

---

### F1-Score: "The Best of Both Worlds"

**F1-score** combines Precision and Recall into a single number:

```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

**Why the harmonic mean?**
- It penalizes models that are good at one but bad at the other
- A model with Precision=1.0 and Recall=0.5 gets F1=0.67 (not 0.75!)
- Forces the model to balance both metrics

---

## Visual Example

### Scenario: Predicting Easy Queries

```
Actual Easy Queries: 900
Actual Hard Queries: 100
Total: 1000
```

### Model A: High Precision, Low Recall

```
Predictions:
- Predicted Easy: 500
  - Actually Easy: 480 (True Positives)
  - Actually Hard: 20 (False Positives)
- Predicted Hard: 500
  - Actually Easy: 420 (False Negatives - missed!)
  - Actually Hard: 80 (True Negatives)

Precision = 480 / (480 + 20) = 96.0% ✅ (Very precise!)
Recall = 480 / (480 + 420) = 53.3% ❌ (Misses many easy queries!)
F1 = 2 × (0.96 × 0.533) / (0.96 + 0.533) = 0.684
```

**Problem**: Model is too conservative - misses many easy queries!

---

### Model B: Low Precision, High Recall

```
Predictions:
- Predicted Easy: 950
  - Actually Easy: 850 (True Positives)
  - Actually Hard: 100 (False Positives - all hard queries predicted as easy!)
- Predicted Hard: 50
  - Actually Easy: 50 (False Negatives)
  - Actually Hard: 0 (True Negatives)

Precision = 850 / (850 + 100) = 89.5% ❌ (Many false positives!)
Recall = 850 / (850 + 50) = 94.4% ✅ (Catches most easy queries!)
F1 = 2 × (0.895 × 0.944) / (0.895 + 0.944) = 0.919
```

**Problem**: Model is too aggressive - labels everything as easy!

---

### Model C: Balanced (Optimal F1)

```
Predictions:
- Predicted Easy: 800
  - Actually Easy: 750 (True Positives)
  - Actually Hard: 50 (False Positives)
- Predicted Hard: 200
  - Actually Easy: 150 (False Negatives)
  - Actually Hard: 50 (True Negatives)

Precision = 750 / (750 + 50) = 93.75% ✅
Recall = 750 / (750 + 150) = 83.3% ✅
F1 = 2 × (0.9375 × 0.833) / (0.9375 + 0.833) = 0.882
```

**Best**: Balanced performance - good at both precision and recall!

---

## F1-Score Interpretation

| F1-Score | Interpretation |
|----------|----------------|
| **0.9 - 1.0** | Excellent - Model is very good at both precision and recall |
| **0.8 - 0.9** | Good - Model performs well on both metrics |
| **0.7 - 0.8** | Fair - Model is decent but could be improved |
| **0.6 - 0.7** | Poor - Model struggles with one or both metrics |
| **< 0.6** | Very Poor - Model is not reliable |

---

## Why F1-Score for This Project?

### Our Project Context

- **Task**: Predict which queries are "easy" (Top-1 correct) vs "hard" (Top-1 wrong)
- **Class Imbalance**: 89.4% easy, 10.6% hard
- **Goal**: 
  - ✅ Correctly identify easy queries (skip re-ranking, save time)
  - ✅ Correctly identify hard queries (apply re-ranking, improve accuracy)

### Why F1-Score is Perfect Here

1. **Handles Imbalance**: Works well when one class (easy) is much larger than the other (hard)
2. **Balanced Metric**: Ensures we don't miss hard queries OR falsely label easy ones
3. **Actionable**: High F1-score means the model is actually useful for our task
4. **Standard Practice**: Common metric for binary classification with imbalanced classes

---

## Our Model's F1-Score

### Validation Results

- **Optimal Threshold**: 0.390
- **F1-Score**: **0.9585** (95.85%)
- **Precision**: 0.9497 (94.97%)
- **Recall**: 0.9674 (96.74%)

**Interpretation**: 
- ✅ **Excellent performance** (F1 > 0.9)
- ✅ **High Precision**: When we predict "easy", we're right 95% of the time
- ✅ **High Recall**: We catch 97% of all easy queries
- ✅ **Well Balanced**: Both precision and recall are high

---

## F1-Score vs Other Metrics

### Comparison Table

| Metric | Formula | What It Measures | Best For |
|--------|----------|------------------|----------|
| **Accuracy** | (TP + TN) / Total | Overall correctness | Balanced classes |
| **Precision** | TP / (TP + FP) | "Am I right when I say easy?" | Minimize false positives |
| **Recall** | TP / (TP + FN) | "Did I catch all easy queries?" | Minimize false negatives |
| **F1-Score** | 2 × (P × R) / (P + R) | Balance of precision & recall | **Imbalanced classes** ✅ |

### When to Use Each

- **Accuracy**: When classes are balanced (50/50)
- **Precision**: When false positives are costly (e.g., medical diagnosis)
- **Recall**: When false negatives are costly (e.g., fraud detection)
- **F1-Score**: When you need balance (e.g., **our project** ✅)

---

## How F1-Score is Calculated in Our Code

### Location: `extension_6_1/stage_3_train_logreg_easy_queries.py`

```python
from sklearn.metrics import f1_score, precision_score, recall_score

# For each threshold, compute F1-score
for threshold in thresholds:
    y_pred = (y_val_probs >= threshold).astype(int)  # Predictions
    
    # Compute F1-score
    f1 = f1_score(y_val, y_pred, zero_division=0)
    
    # Also compute precision and recall
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    
    # Select threshold with highest F1-score
    if f1 > best_f1:
        best_f1 = f1
        optimal_threshold = threshold
```

---

## Summary

**F1-Score** is:
- ✅ A single number (0 to 1) that measures model performance
- ✅ The harmonic mean of Precision and Recall
- ✅ Perfect for imbalanced classes (like our 89.4% easy vs 10.6% hard)
- ✅ Ensures the model is good at both:
  - Not missing easy queries (high recall)
  - Not falsely labeling hard queries as easy (high precision)

**Our Model**: F1-score = **0.9585** (95.85%) = **Excellent performance!** ✅

---

## Further Reading

- **Confusion Matrix**: Visual representation of TP, FP, TN, FN
- **ROC-AUC**: Another metric for binary classification
- **Precision-Recall Curve**: Shows trade-off between precision and recall

