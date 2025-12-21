# Understanding the Two Different "Recall" Metrics

## Overview

You're seeing two different types of "Recall" in the charts, and they measure completely different things:

1. **Chart 1 (Bottom-left): Classification Recall** - Measures how well the logistic regression model identifies "easy queries"
2. **Chart 2: VPR Recall@1** - Measures the actual Visual Place Recognition system's accuracy

---

## Chart 1 (Bottom-Left): Classification Recall vs Threshold

### What It Measures

**Classification Recall** = How well the logistic regression model detects "easy queries"

```
Classification Recall = True Positives / (True Positives + False Negatives)

Where:
- True Positives = Queries that are actually easy AND predicted as easy
- False Negatives = Queries that are actually easy BUT predicted as hard (missed!)
```

### Why It Decreases as Threshold Increases

**As threshold increases:**
- Model becomes **more conservative** (stricter criteria for "easy")
- Fewer queries are predicted as "easy"
- More easy queries are **missed** (predicted as hard instead)
- **False Negatives increase** → Recall decreases

**Example:**
- Threshold = 0.1: Model predicts 95% of queries as easy → High recall (catches most easy queries)
- Threshold = 0.9: Model predicts only 10% of queries as easy → Low recall (misses many easy queries)

### This is NOT about VPR or Image Matching

This recall measures:
- ✅ **Classification model performance** (logistic regression)
- ❌ **NOT** VPR accuracy
- ❌ **NOT** image matching performance

**It answers:** "How good is the model at identifying easy queries?"

---

## Chart 2: Adaptive VPR Recall@1 vs Threshold

### What It Measures

**VPR Recall@1** = Actual Visual Place Recognition system accuracy

```
VPR Recall@1 = (Number of queries with correct Top-1 match within 25m) / (Total queries)
```

This measures the **actual place recognition performance** of the entire system.

### Why It Increases as Threshold Increases

**As threshold increases:**
- More queries are predicted as **"hard"**
- More queries get **re-ranking** (image matching applied)
- Re-ranking improves accuracy for hard queries
- **Overall VPR Recall@1 increases**

**Example:**
- Threshold = 0.1: Only 5% of queries get re-ranking → Lower VPR Recall@1
- Threshold = 0.9: 90% of queries get re-ranking → Higher VPR Recall@1 (closer to full re-ranking)

### This IS about VPR Performance

This recall measures:
- ✅ **Actual VPR system accuracy** (place recognition)
- ✅ **After adaptive re-ranking** (some queries get image matching, some don't)
- ❌ **NOT** classification model performance

**It answers:** "How accurate is the VPR system at recognizing places?"

---

## Key Differences Summary

| Aspect | Chart 1: Classification Recall | Chart 2: VPR Recall@1 |
|--------|--------------------------------|----------------------|
| **What it measures** | Logistic regression model's ability to detect easy queries | Actual VPR system's place recognition accuracy |
| **Decreases with threshold?** | ✅ YES (model becomes more conservative) | ❌ NO (increases because more queries get re-ranking) |
| **Related to** | Classification model performance | VPR system performance |
| **Related to image matching?** | ❌ NO | ✅ YES (re-ranking uses image matching) |
| **Related to VPR?** | ❌ NO (just classification) | ✅ YES (actual VPR accuracy) |
| **What it tells us** | "How good is the model at identifying easy queries?" | "How accurate is the VPR system at recognizing places?" |

---

## Why They Behave Differently

### Classification Recall (Chart 1) - Decreases

```
Threshold ↑ → Model more conservative → Fewer "easy" predictions → 
More easy queries missed → False Negatives ↑ → Recall ↓
```

**Trade-off:**
- Lower threshold = Higher recall (catches more easy queries) but lower precision (more false positives)
- Higher threshold = Lower recall (misses easy queries) but higher precision (fewer false positives)

### VPR Recall@1 (Chart 2) - Increases

```
Threshold ↑ → More queries predicted as "hard" → More re-ranking applied → 
Better accuracy for hard queries → Overall VPR Recall@1 ↑
```

**Trade-off:**
- Lower threshold = Lower VPR Recall@1 (fewer queries get re-ranking) but **saves time** (less computation)
- Higher threshold = Higher VPR Recall@1 (more queries get re-ranking) but **costs more time** (more computation)

---

## Visual Explanation

### Chart 1: Classification Recall
```
Threshold = 0.1:  "Is this query easy?" → Model says YES to 95% → Recall = 97%
                  (Catches most easy queries)

Threshold = 0.9:  "Is this query easy?" → Model says YES to 10% → Recall = 65%
                  (Misses many easy queries - too conservative!)
```

### Chart 2: VPR Recall@1
```
Threshold = 0.1:  Only 5% queries get re-ranking → VPR Recall@1 = 91.0%
                   (Most queries use retrieval-only, some errors remain)

Threshold = 0.9:  90% queries get re-ranking → VPR Recall@1 = 92.5%
                   (Most queries get re-ranking, better accuracy)
```

---

## Why This Matters

### For Classification Model (Chart 1)
- **High recall** = Model catches most easy queries (good for saving time)
- **Low recall** = Model misses easy queries (bad - we do unnecessary re-ranking)

### For VPR System (Chart 2)
- **High Recall@1** = System accurately recognizes places (good for accuracy)
- **Low Recall@1** = System makes more mistakes (bad - wrong place matches)

### The Optimal Threshold

The optimal threshold (marked with stars) balances:
- **Classification performance** (Chart 1): Good at identifying easy queries
- **VPR performance** (Chart 2): Good place recognition accuracy
- **Computational cost**: Not too many queries get re-ranking

---

## Summary

1. **Chart 1 Recall** = Classification model's ability to detect easy queries
   - Decreases with threshold (model becomes more conservative)
   - NOT about VPR or image matching

2. **Chart 2 Recall@1** = Actual VPR system's place recognition accuracy
   - Increases with threshold (more queries get re-ranking)
   - IS about VPR performance (uses image matching for hard queries)

**They measure completely different things!**


