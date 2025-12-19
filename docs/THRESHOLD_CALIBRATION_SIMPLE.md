# Threshold Calibration - Simple Explanation

## What is Threshold Calibration? (Simple Version)

Think of it like **adjusting a thermostat** for different rooms:

### The Problem:
- **Same model** (same "thermometer")
- **Different datasets** (different "rooms")
- **Different thresholds needed** (different "temperature settings")

### Example:
```
Room 1 (SF-XS): Cold room → Set thermostat to 15°C
Room 2 (Tokyo-XS): Warm room → Set thermostat to 20°C  
Room 3 (SVOX): Hot room → Set thermostat to 25°C
```

**Same thermometer, different settings for different rooms!**

---

## How It Works (Step by Step)

### Step 1: Model Gives Scores
The model looks at each query and gives it a **confidence score** (0 to 1):
- **Score = 0.95** → "I'm 95% confident this is EASY"
- **Score = 0.30** → "I'm 30% confident this is EASY" (probably HARD)

### Step 2: We Need to Decide
We need to pick a **cutoff point** (threshold):
- **Above threshold** → EASY (skip expensive re-ranking)
- **Below threshold** → HARD (do expensive re-ranking)

### Step 3: The Problem
**Different datasets have different score distributions:**

```
SF-XS Test:
  Scores: [0.45, 0.52, 0.38, 0.61, ...]  ← Lower scores
  Threshold 0.41 works ✅

Tokyo-XS Test:
  Scores: [0.99, 0.998, 0.97, 1.0, ...]  ← Very high scores!
  Threshold 0.41 doesn't work ❌ (all above threshold → all easy)
```

### Step 4: Calibration Finds the Right Threshold
**Calibration = Finding the best threshold for THIS specific dataset**

It tries many thresholds and picks the best one:
```
Try threshold = 0.1 → Accuracy: 60%
Try threshold = 0.5 → Accuracy: 80%
Try threshold = 0.65 → Accuracy: 85% ✅ BEST!
Try threshold = 0.9 → Accuracy: 70%
```

**Result: Use threshold = 0.65 for Tokyo-XS**

---

## Two Ways to Calibrate

### Method 1: Maximize F1-Score (Default)
**Goal**: Find threshold that gives best overall performance

**How**: Tries all thresholds, picks one with highest F1-score

**Use when**: You want the best balance

### Method 2: Target Hard Query Rate
**Goal**: Find threshold that gives specific percentage of hard queries

**Example**: "I want 30% hard queries"
- System finds threshold that makes ~30% of queries hard
- You control computational cost

**Use when**: You have a budget (e.g., "I can only process 30% of queries")

---

## Real Example: Tokyo-XS

### Without Calibration:
```
Model scores: [0.99, 0.998, 0.97, 1.0, ...]  (all very high!)
Fixed threshold: 0.41 (from SF-XS)

Result:
  All queries: score > 0.41
  → ALL predicted as EASY
  → 0% hard queries detected ❌
```

### With Calibration (F1-maximization):
```
Model scores: [0.99, 0.998, 0.97, 1.0, ...]

Calibration process:
  Try threshold = 0.1 → F1 = 0.79
  Try threshold = 0.2 → F1 = 0.80
  ...
  Try threshold = 0.65 → F1 = 0.85 ✅ BEST

Result:
  Queries with score < 0.65 → HARD
  Queries with score >= 0.65 → EASY
  → Some hard queries detected ✅
```

### With Calibration (Target 30% hard queries):
```
Model scores: [0.99, 0.998, 0.97, 1.0, ...]

Calibration process:
  Target: 30% hard queries
  Find threshold that makes 30% of queries hard
  → Threshold = 0.70

Result:
  30% of queries → HARD (do re-ranking)
  70% of queries → EASY (skip re-ranking)
  → Controlled computational cost ✅
```

---

## Summary

### What is it?
**Finding the best threshold for a specific dataset**

### Why do we need it?
**Different datasets need different thresholds**

### How does it work?
1. Get model predictions (scores)
2. Try different thresholds
3. Pick the best one

### When to use it?
**Always when applying model to a new dataset**

---

## Code Example

### Without Calibration:
```python
# Use fixed threshold
threshold = 0.41  # From SF-XS validation

# Apply to Tokyo-XS
scores = [0.99, 0.998, 0.97, 1.0, ...]
predictions = scores >= threshold
# Result: [True, True, True, True, ...]  ← All easy! ❌
```

### With Calibration:
```python
# Calibrate threshold on Tokyo-XS
scores = [0.99, 0.998, 0.97, 1.0, ...]
true_labels = [1, 1, 0, 1, ...]  # Ground truth

# Find best threshold
best_threshold = find_optimal_threshold(true_labels, scores)
# Result: best_threshold = 0.65 ✅

# Apply calibrated threshold
predictions = scores >= best_threshold
# Result: [True, True, True, True, ...]  ← Some hard queries detected ✅
```

---

## Key Takeaway

**Threshold calibration = Adjusting the decision point for each dataset**

- **Model stays the same** (same "thermometer")
- **Threshold changes** (different "temperature settings")
- **Each dataset gets its own optimal threshold** (different "rooms")

