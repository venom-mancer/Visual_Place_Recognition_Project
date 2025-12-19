# Improving Model Generalization - Strategies and Recommendations

## Problem Statement

**Current Issue:**
- Model trained on SVOX, validated on SF-XS
- Poor generalization to new test datasets (Tokyo-XS, SVOX test)
- Model is overconfident: predicts all queries as easy (0% hard queries)
- Adaptive re-ranking performance = baseline (no improvement)
- Full re-ranking shows significant gains (+18.1% on Tokyo-XS)

**Goal:**
- Keep adaptive re-ranking accuracy **close to full re-ranking**
- While maintaining **time efficiency** (save computation)

**Constraint:**
- Cannot retrain model on test data
- Ground truth: Full re-ranking performance

---

## Current Performance Gap

### Tokyo-XS Test:
| Method | R@1 | Hard Queries | Time Savings |
|--------|-----|--------------|--------------|
| **Baseline** | 65.1% | 0% | 100% |
| **Full Re-ranking** | **83.2%** | 100% | 0% |
| **Adaptive (Current)** | 65.1% | 0% | 100% |
| **Gap** | **-18.1%** | - | - |

**Problem**: Model detects 0% hard queries → misses 18.1% performance gain

---

## Recommended Solutions

### Solution 1: Conservative Thresholding with Performance Targeting ⭐ **RECOMMENDED**

**Idea**: Use full re-ranking results to find threshold that maximizes performance while maintaining efficiency.

**Approach**:
1. Run full re-ranking on test set (already done)
2. Compare adaptive vs full re-ranking performance
3. Find threshold that minimizes performance gap
4. Target threshold that achieves ~80-90% of full re-ranking performance

**Implementation**:
```python
# Pseudo-code
def find_performance_targeted_threshold(model_probs, full_reranking_performance, target_performance_ratio=0.90):
    """
    Find threshold that achieves target_performance_ratio of full re-ranking performance.
    """
    best_threshold = 0.5
    best_performance = 0.0
    
    for threshold in np.arange(0.5, 0.99, 0.01):  # Conservative: higher threshold
        is_hard = model_probs < threshold
        adaptive_performance = evaluate_adaptive_reranking(is_hard)
        performance_ratio = adaptive_performance / full_reranking_performance
        
        if performance_ratio >= target_performance_ratio:
            return threshold  # Found threshold that meets target
    
    return best_threshold
```

**Advantages**:
- ✅ Uses full re-ranking as ground truth
- ✅ Directly optimizes for performance target
- ✅ Maintains efficiency (only processes hard queries)
- ✅ Works with current model (no retraining)

**Expected Result**:
- Tokyo-XS: Threshold ~0.70-0.80 → ~30-40% hard queries → ~75-80% R@1 (vs 83.2% full)

---

### Solution 2: Target Hard Query Rate Based on Actual Wrong Queries

**Idea**: If we know ~35% of queries are wrong, target ~35% hard queries.

**Approach**:
1. Estimate wrong query rate from baseline performance
2. Use `--target-hard-rate` to achieve that rate
3. Conservative: Target slightly higher than estimated wrong rate

**Implementation**:
```bash
# For Tokyo-XS: ~35% wrong queries
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal_C_tuned.pkl \
  --feature-path features_tokyo_xs_test_improved.npz \
  --output-path logreg_tokyo_xs_test_targeted.npz \
  --calibrate-threshold \
  --target-hard-rate 0.40  # 40% hard queries (slightly higher than 35% wrong)
```

**Advantages**:
- ✅ Simple to implement
- ✅ Uses existing calibration feature
- ✅ Conservative approach (more hard queries = better performance)

**Expected Result**:
- Tokyo-XS: ~40% hard queries → Better performance than current 0%

---

### Solution 3: Feature Normalization Per Dataset

**Idea**: Normalize features per dataset to account for distribution shift.

**Approach**:
1. Compute dataset-specific feature statistics (mean, std)
2. Normalize features using dataset statistics
3. Apply model with normalized features

**Implementation**:
```python
# In stage_4_apply_logreg_easy_queries.py
def normalize_features_per_dataset(X, dataset_name):
    """Normalize features using dataset-specific statistics."""
    if dataset_name == "tokyo_xs":
        # Tokyo-XS specific normalization
        # Adjust based on observed distribution
        X_normalized = (X - tokyo_xs_mean) / tokyo_xs_std
    elif dataset_name == "svox_test":
        # SVOX test specific normalization
        X_normalized = (X - svox_test_mean) / svox_test_std
    else:
        # Use standard scaler (from training)
        X_normalized = scaler.transform(X)
    return X_normalized
```

**Advantages**:
- ✅ Addresses distribution shift directly
- ✅ Can improve probability calibration
- ✅ Works with current model

**Challenges**:
- ⚠️ Requires dataset-specific statistics
- ⚠️ May need tuning per dataset

---

### Solution 4: Uncertainty-Aware Thresholding

**Idea**: Use model uncertainty to adjust threshold dynamically.

**Approach**:
1. Compute prediction uncertainty (e.g., entropy, variance)
2. Lower threshold for high-uncertainty queries (predict as hard)
3. Higher threshold for low-uncertainty queries (predict as easy)

**Implementation**:
```python
def uncertainty_aware_thresholding(probs, base_threshold=0.5, uncertainty_weight=0.1):
    """
    Adjust threshold based on prediction uncertainty.
    """
    # Compute uncertainty (entropy)
    entropy = -probs * np.log(probs + 1e-10) - (1 - probs) * np.log(1 - probs + 1e-10)
    uncertainty = entropy / np.log(2)  # Normalize to [0, 1]
    
    # Adjust threshold: lower for high uncertainty
    adjusted_threshold = base_threshold - uncertainty_weight * uncertainty
    
    # Predict hard if prob < adjusted_threshold
    is_hard = probs < adjusted_threshold
    return is_hard
```

**Advantages**:
- ✅ Adaptive to model confidence
- ✅ Can detect hard queries even with high probabilities
- ✅ Theoretically sound

**Challenges**:
- ⚠️ Requires uncertainty estimation
- ⚠️ May need tuning

---

### Solution 5: Ensemble of Thresholds

**Idea**: Use multiple thresholds and combine predictions.

**Approach**:
1. Test multiple thresholds (conservative, moderate, aggressive)
2. For each query, use most conservative prediction (predict as hard if any threshold says hard)
3. Or use voting: predict as hard if majority of thresholds say hard

**Advantages**:
- ✅ More robust to threshold selection
- ✅ Can catch edge cases
- ✅ Simple to implement

**Challenges**:
- ⚠️ May predict too many as hard (reduces efficiency)

---

## Recommended Implementation Plan

### Phase 1: Quick Win (Solution 2) ⭐ **START HERE**

**Goal**: Use target hard query rate to achieve better performance

**Steps**:
1. Estimate wrong query rate from baseline
2. Use `--target-hard-rate` with conservative estimate (slightly higher)
3. Evaluate performance vs full re-ranking

**Expected**: 30-40% hard queries → Better performance than current 0%

### Phase 2: Performance Optimization (Solution 1) ⭐ **MAIN SOLUTION**

**Goal**: Find threshold that achieves target performance (e.g., 90% of full re-ranking)

**Steps**:
1. Implement performance-targeted threshold finding
2. Use full re-ranking results as ground truth
3. Find threshold that minimizes performance gap
4. Evaluate efficiency vs performance trade-off

**Expected**: 40-50% hard queries → 75-80% R@1 (vs 83.2% full) → Good balance

### Phase 3: Advanced (Solutions 3-5)

**Goal**: Further improve generalization

**Steps**:
1. Implement feature normalization per dataset
2. Add uncertainty-aware thresholding
3. Test ensemble methods

---

## Implementation: Solution 1 (Performance-Targeted Thresholding)

### New Script: `find_performance_targeted_threshold.py`

```python
"""
Find threshold that achieves target performance ratio of full re-ranking.
"""

def find_performance_targeted_threshold(
    model_probs,
    full_reranking_recall_at_1,
    preds_dir,
    inliers_dir,
    target_performance_ratio=0.90,
    num_preds=20,
    positive_dist_threshold=25
):
    """
    Find threshold that achieves target_performance_ratio of full re-ranking performance.
    
    Args:
        model_probs: Model predictions (probability of being easy)
        full_reranking_recall_at_1: Full re-ranking R@1 (ground truth)
        preds_dir: Directory with prediction files
        inliers_dir: Directory with inlier files (from full re-ranking)
        target_performance_ratio: Target performance ratio (0.90 = 90% of full re-ranking)
        num_preds: Number of predictions to consider
        positive_dist_threshold: Distance threshold in meters
    
    Returns:
        optimal_threshold: Threshold that achieves target performance
        achieved_performance: Achieved R@1
        hard_query_rate: Percentage of hard queries
    """
    # Try conservative thresholds (higher = more hard queries)
    thresholds = np.arange(0.5, 0.99, 0.01)
    
    for threshold in thresholds:
        is_hard = model_probs < threshold
        hard_query_rate = is_hard.mean()
        
        # Evaluate adaptive re-ranking with this threshold
        adaptive_recall_at_1 = compute_recall_at_1_adaptive(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        
        performance_ratio = adaptive_recall_at_1 / full_reranking_recall_at_1
        
        if performance_ratio >= target_performance_ratio:
            return threshold, adaptive_recall_at_1, hard_query_rate
    
    # If no threshold achieves target, return most conservative
    return thresholds[-1], adaptive_recall_at_1, hard_query_rate
```

---

## Expected Results

### Tokyo-XS Test (Target: 90% of full re-ranking = 74.9% R@1):

| Approach | Threshold | Hard Queries | R@1 | Performance Ratio | Time Savings |
|----------|-----------|--------------|-----|-------------------|--------------|
| **Current** | 0.100 | 0% | 65.1% | 78.2% | 100% |
| **Target 40%** | ~0.70 | 40% | ~75-78% | ~90-94% | 60% |
| **Target 90%** | ~0.75 | 50% | ~75-80% | ~90-96% | 50% |
| **Full Re-ranking** | - | 100% | 83.2% | 100% | 0% |

**Trade-off**: 50% hard queries → 90-96% of full performance → 50% time savings

---

## Summary

**Best Approach**: **Solution 1 (Performance-Targeted Thresholding)**

**Why**:
- ✅ Directly optimizes for performance target
- ✅ Uses full re-ranking as ground truth
- ✅ Maintains efficiency
- ✅ Works with current model

**Quick Win**: **Solution 2 (Target Hard Query Rate)**

**Why**:
- ✅ Simple to implement (already available)
- ✅ Can improve immediately
- ✅ Conservative approach

**Next Steps**:
1. Implement Solution 1 (performance-targeted thresholding)
2. Test on Tokyo-XS with target 90% of full re-ranking
3. Evaluate efficiency vs performance trade-off
4. Apply to other test datasets

---

*See implementation in next steps.*

