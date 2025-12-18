# Test Datasets Status

## Available Test Datasets

Based on the project documentation and dataset information, there are **3 test datasets**:

1. **SF-XS test** ✅ **COMPLETED**
2. **Tokyo-XS test** ⏳ **PENDING**
3. **SVOX test** ⏳ **PENDING**

---

## Test Status Summary

| Dataset | Status | Queries | Results Available |
|---------|--------|---------|-------------------|
| **SF-XS test** | ✅ **Completed** | 1,000 | Yes |
| **Tokyo-XS test** | ⏳ **Pending** | Unknown | No |
| **SVOX test** | ⏳ **Pending** | Unknown | No |

**Total Tested**: 1 out of 3 (33.3%)  
**Remaining**: 2 datasets

---

## Completed: SF-XS Test

### Results:
- **Baseline**: R@1: 63.1%, R@5: 74.8%, R@10: 78.6%, R@20: 81.4%
- **Full Re-ranking**: R@1: 77.4%, R@5: 80.3%, R@10: 80.9%, R@20: 81.4%
- **Adaptive (Regressor)**: R@1: 73.4%, R@5: 78.9%, R@10: 80.3%, R@20: 81.4%
- **Adaptive (LogReg Easy)**: R@1: 69.8%, R@5: 77.7%, R@10: 79.5%, R@20: 81.4%

### Files:
- Features: `features_sf_xs_test_improved.npz`
- Predictions: `regressor_test_final.npz`, `logreg_easy_queries_test.npz`
- Logs: `logs/log_sf_xs_test/2025-12-17_21-14-10/`

---

## Pending: Tokyo-XS Test

### Status:
- ⏳ **Not tested yet**
- No log directory found: `logs/log_tokyo_xs_test/`
- No feature files found: `features_tokyo_xs_test*.npz`

### Next Steps:
1. Run VPR evaluation on Tokyo-XS test
2. Extract features (8 improved features)
3. Apply model (Regressor or LogReg Easy)
4. Run image matching on hard queries
5. Evaluate adaptive re-ranking

---

## Pending: SVOX Test

### Status:
- ⏳ **Not tested yet**
- No log directory found: `logs/log_svox_test/`
- No feature files found: `features_svox_test*.npz`

### Note:
- SVOX has Sun and Night subsets for testing
- May need separate evaluation for each subset

### Next Steps:
1. Run VPR evaluation on SVOX test
2. Extract features (8 improved features)
3. Apply model (Regressor or LogReg Easy)
4. Run image matching on hard queries
5. Evaluate adaptive re-ranking

---

## Summary

**Tested**: 1 dataset (SF-XS test)  
**Remaining**: 2 datasets (Tokyo-XS test, SVOX test)

**Progress**: 33.3% complete

---

*Last updated: 2025-12-18*

