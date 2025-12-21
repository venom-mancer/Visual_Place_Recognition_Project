# Threshold Calibration Explained (Beginner Level)

## What is a Threshold?

Think of the logistic regression model as a **confidence meter** that gives each query a score from 0 to 1:
- **Score close to 1.0** = "I'm very confident this query is EASY (correct)"
- **Score close to 0.0** = "I'm very confident this query is HARD (wrong)"

But we need to **decide** at what score we say "this is easy" vs "this is hard".

**The threshold is that decision point.**

### Example:
```
Query 1: Score = 0.95 → Above threshold 0.5 → EASY ✅
Query 2: Score = 0.30 → Below threshold 0.5 → HARD ❌
Query 3: Score = 0.60 → Above threshold 0.5 → EASY ✅
```

---

## What is Threshold Calibration?

**Threshold calibration** means **finding the best threshold** for a specific dataset.

### Why Do We Need It?

Imagine you have a **thermometer** that measures temperature:
- In **San Francisco** (cool weather): 20°C feels warm → threshold = 15°C
- In **Tokyo** (warmer weather): 20°C feels normal → threshold = 18°C
- In **SVOX** (hot weather): 20°C feels cool → threshold = 22°C

**Same thermometer, different thresholds for different places!**

Similarly:
- **SF-XS dataset**: Model gives scores around 0.5-0.6 → threshold = 0.41 works
- **Tokyo-XS dataset**: Model gives scores around 0.99 → threshold = 0.41 is too low!
- **SVOX dataset**: Model gives scores around 1.0 → threshold = 0.41 is way too low!

---

## The Problem We're Solving

### Without Calibration (Fixed Threshold):

```
SF-XS Test:
  Query scores: [0.45, 0.52, 0.38, 0.61, ...]
  Threshold: 0.41
  Result: ✅ Works! (many queries above 0.41)

Tokyo-XS Test:
  Query scores: [0.99, 0.998, 0.97, 1.0, ...]
  Threshold: 0.41 (same as SF-XS)
  Result: ❌ All queries above 0.41 → ALL predicted as EASY!
  Problem: We miss all hard queries!
```

### With Calibration (Dataset-Specific Threshold):

```
SF-XS Test:
  Query scores: [0.45, 0.52, 0.38, 0.61, ...]
  Calibrated threshold: 0.41
  Result: ✅ Works!

Tokyo-XS Test:
  Query scores: [0.99, 0.998, 0.97, 1.0, ...]
  Calibrated threshold: 0.65 (found specifically for Tokyo-XS)
  Result: ✅ Works! (some queries below 0.65 → predicted as HARD)
```

---

## How Does Calibration Work?

### Step 1: Get Model Predictions

The model gives each query a probability score:
```
Query 1: 0.95 (95% confident it's easy)
Query 2: 0.30 (30% confident it's easy)
Query 3: 0.80 (80% confident it's easy)
...
```

### Step 2: Try Different Thresholds

We test many thresholds (0.1, 0.2, 0.3, ..., 0.9) and see which one works best:

```
Threshold = 0.1:
  Query 1: 0.95 > 0.1 → EASY ✅
  Query 2: 0.30 > 0.1 → EASY ❌ (wrong! it's actually hard)
  Query 3: 0.80 > 0.1 → EASY ✅
  Accuracy: 66%

Threshold = 0.5:
  Query 1: 0.95 > 0.5 → EASY ✅
  Query 2: 0.30 < 0.5 → HARD ✅
  Query 3: 0.80 > 0.5 → EASY ✅
  Accuracy: 100% ✅ Best!

Threshold = 0.9:
  Query 1: 0.95 > 0.9 → EASY ✅
  Query 2: 0.30 < 0.9 → HARD ✅
  Query 3: 0.80 < 0.9 → HARD ❌ (wrong! it's actually easy)
  Accuracy: 66%
```

### Step 3: Choose Best Threshold

We pick the threshold that gives the **best accuracy** (or F1-score):
- **Best threshold = 0.5** (gives 100% accuracy in this example)

---

## Real Example: Tokyo-XS Dataset

### Without Calibration:

```
Model predictions for Tokyo-XS:
  All queries have scores: 0.99, 0.998, 0.97, 1.0, ...

Fixed threshold (from SF-XS): 0.41

Result:
  All queries: score > 0.41 → ALL predicted as EASY
  Hard queries detected: 0% ❌
  Problem: We miss all hard queries!
```

### With Calibration:

```
Model predictions for Tokyo-XS:
  All queries have scores: 0.99, 0.998, 0.97, 1.0, ...

Calibration process:
  1. Try threshold = 0.50 → 0% hard queries (too low)
  2. Try threshold = 0.60 → 5% hard queries
  3. Try threshold = 0.65 → 30% hard queries ✅ (best F1-score)
  4. Try threshold = 0.70 → 50% hard queries (too many)
  5. Try threshold = 0.75 → 70% hard queries (too many)

Calibrated threshold: 0.65

Result:
  Queries with score < 0.65 → predicted as HARD
  Hard queries detected: 30% ✅
  Problem solved!
```

---

## Two Types of Calibration

### Type 1: F1-Maximization (Default)

**Goal**: Find threshold that gives the **best overall performance** (F1-score)

**How it works**:
1. Try many thresholds (0.1 to 0.95)
2. For each threshold, compute F1-score
3. Pick threshold with highest F1-score

**Use when**: You want the best balance between precision and recall

### Type 2: Target Hard Query Rate

**Goal**: Find threshold that gives a **specific percentage** of hard queries

**How it works**:
1. You specify: "I want 30% hard queries"
2. System finds threshold that achieves ~30% hard queries
3. Uses that threshold

**Use when**: You have a computational budget (e.g., "I can only process 30% of queries")

---

## Visual Example

Imagine you have a **bar chart** of query scores:

```
Without Calibration (threshold = 0.41):
  |████████████████████████████|  All queries
  |                            |
  |←─── threshold 0.41 ───→   |
  |                            |
  Result: All above threshold → ALL EASY ❌

With Calibration (threshold = 0.65):
  |████████████████|███████████|  Queries
  |                |            |
  |←─ threshold ─→|            |
  |    0.65        |            |
  |                |            |
  Result: Some below → HARD ✅
          Some above → EASY ✅
```

---

## Summary

### What is Threshold Calibration?
**Finding the best threshold for a specific dataset**

### Why Do We Need It?
**Different datasets have different score distributions, so they need different thresholds**

### How Does It Work?
1. Get model predictions (scores)
2. Try different thresholds
3. Pick the best one (highest F1-score or target rate)

### When to Use It?
**Always when applying model to a new dataset that's different from training/validation**

---

## Code Example

### Without Calibration:
```python
# Use fixed threshold from validation
threshold = 0.41  # Learned on SF-XS validation

# Apply to Tokyo-XS
scores = model.predict_proba(tokyo_xs_features)[:, 1]
predictions = scores >= threshold  # All True! ❌
```

### With Calibration:
```python
# Calibrate threshold on Tokyo-XS
scores = model.predict_proba(tokyo_xs_features)[:, 1]
true_labels = tokyo_xs_labels

# Find best threshold
best_threshold = find_optimal_threshold(true_labels, scores)
# Result: best_threshold = 0.65 ✅

# Apply calibrated threshold
predictions = scores >= best_threshold  # Some True, some False ✅
```

---

## Key Takeaway

**Think of threshold calibration as adjusting a thermostat:**
- Same model (same "thermometer")
- Different datasets (different "rooms")
- Different thresholds (different "temperature settings")

**The model stays the same, but we adjust the decision point for each dataset!**

