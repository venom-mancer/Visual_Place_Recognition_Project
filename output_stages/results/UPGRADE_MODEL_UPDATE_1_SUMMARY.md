# Upgrade Model - Update 1: Quick Summary

## What Changed?

**From**: Logistic Regression Classifier (binary, 8 features, probability threshold)  
**To**: Random Forest Regressor (continuous, 8 features, wrong_score prediction)

## Key Strategy

1. **Predict continuous "wrong score"** (0 = correct, 1 = wrong) instead of binary classification
2. **Use 8 retrieval features** available before image matching
3. **Apply simple threshold** (0.5) on predicted score to classify hard/easy
4. **Run image matching only for hard queries** (saves 57.5% time)

## Results

| Metric | Value |
|--------|-------|
| **R@1 vs Baseline** | **+10.3%** (73.4% vs 63.1%) |
| **R@1 vs Full Re-ranking** | -4.0% (73.4% vs 77.4%) |
| **Hard Queries Detected** | 42.5% (425/1000) |
| **Time Savings** | 57.5% of image matching time |
| **Validation Accuracy** | 86.7% |

## Why It Works

1. **Direct target**: Predicts Top-1 correctness (wrong_score) - directly observable
2. **Rich features**: 8 features capture multiple uncertainty aspects
3. **Continuous score**: More nuanced than binary, better interpretability
4. **Efficient pipeline**: Image matching runs after prediction, not before

## Files

- Model: `regressor_model_final.pkl`
- Test results: `regressor_test_final.npz`
- Full report: `UPGRADE_MODEL_UPDATE_1_REGRESSOR_STRATEGY.md`

---

*Update 1 - 2025-12-18*

