# Remaining Test Datasets

## Summary

**Tested**: 1 out of 3 datasets (33.3%)  
**Remaining**: 2 datasets

---

## Test Datasets Overview

| Dataset | Status | Data Available | Logs Available | Results |
|---------|--------|----------------|----------------|---------|
| **SF-XS test** | ✅ **Completed** | ✅ Yes | ✅ Yes | ✅ Yes |
| **Tokyo-XS test** | ⏳ **Pending** | ✅ Yes | ❌ No | ❌ No |
| **SVOX test** | ⏳ **Pending** | ✅ Yes | ❌ No | ❌ No |

---

## ✅ Completed: SF-XS Test

**Location**: `data/sf_xs/test/`  
**Logs**: `logs/log_sf_xs_test/2025-12-17_21-14-10/`  
**Queries**: 1,000

### Results:
- Baseline: R@1: 63.1%
- Full Re-ranking: R@1: 77.4%
- Adaptive (Regressor): R@1: 73.4% (+10.3% vs baseline)
- Adaptive (LogReg Easy): R@1: 69.8% (+6.7% vs baseline)

---

## ⏳ Remaining: Tokyo-XS Test

**Location**: `data/tokyo_xs/test/`  
**Status**: Data available, but not tested yet

### Required Steps:
1. Run VPR evaluation on Tokyo-XS test
2. Extract features (8 improved features) → `features_tokyo_xs_test_improved.npz`
3. Apply model (Regressor or LogReg Easy)
4. Run image matching on hard queries only
5. Evaluate adaptive re-ranking

### Expected Commands:
```bash
# 1. VPR Evaluation
python VPR-methods-evaluation/main.py \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --database_folder data/tokyo_xs/test/database \
  --queries_folder data/tokyo_xs/test/queries \
  --num_preds_to_save 20 \
  --log_dir log_tokyo_xs_test

# 2. Extract Features
python -m extension_6_1.stage_1_extract_features_no_inliers \
  --preds-dir logs/log_tokyo_xs_test/.../preds \
  --z-data-path logs/log_tokyo_xs_test/.../z_data.torch \
  --output-path features_tokyo_xs_test_improved.npz

# 3. Apply Model (choose one)
# Option A: Regressor
python -m extension_6_1.stage_4_apply_regressor \
  --model-path regressor_model_final.pkl \
  --feature-path features_tokyo_xs_test_improved.npz \
  --output-path regressor_tokyo_xs_test.npz \
  --hard-queries-output hard_queries_tokyo_xs_test_regressor.txt

# Option B: LogReg Easy
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path features_tokyo_xs_test_improved.npz \
  --output-path logreg_easy_tokyo_xs_test.npz \
  --hard-queries-output hard_queries_tokyo_xs_test_logreg_easy.txt

# 4. Image Matching (adaptive)
python match_queries_preds_adaptive.py \
  --preds-dir logs/log_tokyo_xs_test/.../preds \
  --hard-queries-list hard_queries_tokyo_xs_test_*.txt \
  --out-dir logs/log_tokyo_xs_test/.../preds_superpoint-lg_adaptive \
  --matcher superpoint-lg --device cuda --num-preds 20

# 5. Evaluate
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir logs/log_tokyo_xs_test/.../preds \
  --inliers-dir logs/log_tokyo_xs_test/.../preds_superpoint-lg_adaptive \
  --logreg-output [regressor or logreg output].npz \
  --num-preds 20 --positive-dist-threshold 25 --recall-values 1 5 10 20
```

---

## ⏳ Remaining: SVOX Test

**Location**: `data/svox/`  
**Status**: Data available, but not tested yet

### Note:
- SVOX has Sun and Night subsets
- May need separate evaluation for each subset or combined

### Required Steps:
1. Run VPR evaluation on SVOX test
2. Extract features (8 improved features) → `features_svox_test_improved.npz`
3. Apply model (Regressor or LogReg Easy)
4. Run image matching on hard queries only
5. Evaluate adaptive re-ranking

### Expected Commands:
```bash
# Similar to Tokyo-XS, but with SVOX paths
# 1. VPR Evaluation
python VPR-methods-evaluation/main.py \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --database_folder data/svox/test/gallery \
  --queries_folder data/svox/test/queries \
  --num_preds_to_save 20 \
  --log_dir log_svox_test

# 2-5. Same steps as Tokyo-XS
```

---

## Progress Summary

| Metric | Value |
|--------|-------|
| **Total Test Datasets** | 3 |
| **Completed** | 1 (SF-XS test) |
| **Remaining** | 2 (Tokyo-XS test, SVOX test) |
| **Progress** | 33.3% |

---

## Next Steps

To complete testing on all datasets:

1. **Tokyo-XS test**:
   - Run full pipeline (VPR → Features → Model → Matching → Evaluation)
   - Expected time: ~2-3 hours (depending on number of queries)

2. **SVOX test**:
   - Run full pipeline (VPR → Features → Model → Matching → Evaluation)
   - Expected time: ~2-3 hours (depending on number of queries)

---

*Status: 1 of 3 test datasets completed (33.3%)*

