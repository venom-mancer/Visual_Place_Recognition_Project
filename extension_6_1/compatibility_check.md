# Compatibility Check: Extension 6.1 Code vs Current Inputs

## ✅ What Matches Correctly

### 1. Image Matching Output Structure (.torch files)
- **Location**: `logs/log_svox_train/2025-12-15_22-58-58/preds_superpoint-lg/`
- **Structure**: List of dictionaries, each containing:
  - `num_inliers` (int) - ✅ Matches what `stage_1_extract_features.py` expects
  - Other keys: `H`, `all_kpts0`, `all_kpts1`, `matched_kpts0`, etc. (not used by our code)
- **Count**: 2279 files ✅ Matches number of .txt files

### 2. Prediction Files (.txt files)
- **Location**: `logs/log_svox_train/2025-12-15_22-58-58/preds/`
- **Structure**: Contains query path and prediction paths
- **Count**: 2279 files ✅
- **Format**: Compatible with `get_list_distances_from_preds()` function

### 3. File Naming Convention
- **Code expects**: `.torch` files with same name as `.txt` files (e.g., `000.txt` → `000.torch`)
- **Actual structure**: ✅ Matches perfectly
- **Path construction**: The code constructs path as `inliers_folder / filename.replace("txt", "torch")` which works correctly

## ❌ Issue Found

### Missing z_data.torch File
- **Expected location**: `logs/log_svox_train/2025-12-15_22-58-58/z_data.torch`
- **Status**: ❌ File does not exist
- **Required by**: `stage_1_extract_features.py` needs:
  - `database_utms` - database image poses
  - `predictions` - retrieval predictions (indices)
  - `distances` - descriptor distances

### Why z_data.torch is Missing
The VPR evaluation (`VPR-methods-evaluation/main.py`) only saves `z_data.torch` when run with the `--save_for_uncertainty` flag (line 367-374).

## 🔧 Solution

You need to re-run the VPR evaluation with the `--save_for_uncertainty` flag:

```bash
python VPR-methods-evaluation/main.py \
  --num_workers 8 \
  --batch_size 32 \
  --log_dir log_svox_train \
  --method=cosplace \
  --backbone=ResNet18 \
  --descriptors_dimension=512 \
  --image_size 512 512 \
  --database_folder data/svox/images/train/gallery \
  --queries_folder data/svox/images/train/queries \
  --num_preds_to_save 20 \
  --recall_values 1 5 10 20 \
  --save_for_uncertainty \
  --device cuda
```

This will create the `z_data.torch` file in the log directory.

## ✅ Code Compatibility Summary

Once `z_data.torch` is created, all inputs will match:

| Input | Expected by Code | Actual Status |
|-------|-----------------|---------------|
| `.txt` prediction files | ✅ Required | ✅ 2279 files exist |
| `.torch` inlier files | ✅ Required | ✅ 2279 files exist |
| `z_data.torch` | ✅ Required | ❌ Missing (needs re-run with flag) |

## Next Steps

1. Re-run VPR evaluation with `--save_for_uncertainty` flag
2. Verify `z_data.torch` is created
3. Run Stage 1 feature extraction:
   ```bash
   python -m extension_6_1.stage_1_extract_features \
     --preds-dir logs/log_svox_train/2025-12-15_22-58-58/preds \
     --inliers-dir logs/log_svox_train/2025-12-15_22-58-58/preds_superpoint-lg \
     --z-data-path logs/log_svox_train/2025-12-15_22-58-58/z_data.torch \
     --output-path features_svox_train.npz
   ```

