# Solution for Mismatched Directories

## Current Situation

- **Old directory** (`2025-12-15_22-58-58`):
  - ✅ Has image matching results: `preds_superpoint-lg/` (2,279 .torch files)
  - ❌ Missing `z_data.torch`

- **New directory** (`2025-12-16_14-37-10`):
  - ✅ Has `z_data.torch` (with 11,294 queries)
  - ✅ Has predictions: `preds/` (11,294 .txt files)
  - ❌ Missing image matching results

## Problem

The new VPR evaluation processed **11,294 queries** (full dataset), while the old image matching only processed **2,279 queries** (subset).

## Solutions

### Option 1: Re-run Image Matching on New Directory (Recommended)

Run image matching on the new directory's predictions to match all 11,294 queries:

```bash
python match_queries_preds.py \
  --preds-dir logs/log_svox_train/2025-12-16_14-37-10/preds \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20
```

This will create: `logs/log_svox_train/2025-12-16_14-37-10/preds_superpoint-lg/`

**Pros**: Complete dataset, all files in one directory
**Cons**: Takes time to process 11,294 queries

### Option 2: Use Old Directory with Limited Queries

If you only want to work with the first 2,279 queries, you can:

1. Copy `z_data.torch` from new directory to old directory
2. Modify `z_data.torch` to only include first 2,279 queries (requires Python script)
3. Use old directory for Stage 1

**Pros**: Faster, no re-processing needed
**Cons**: Only 2,279 queries instead of full 11,294

### Option 3: Copy Image Matching Results (If Queries Match)

If the first 2,279 queries are identical in both directories, you can:

1. Copy image matching results from old to new directory
2. Use new directory for Stage 1 (but only process first 2,279 queries)

**Pros**: Quick solution
**Cons**: Only works if queries match exactly, limited to 2,279 queries

## Recommended Approach

**Use Option 1** - Re-run image matching on the new directory. This ensures:
- All files are in one directory
- Complete dataset (11,294 queries)
- No compatibility issues

## After Image Matching Completes

Once image matching is done on the new directory, use this for Stage 1:

```bash
python -m extension_6_1.stage_1_extract_features \
  --preds-dir logs/log_svox_train/2025-12-16_14-37-10/preds \
  --inliers-dir logs/log_svox_train/2025-12-16_14-37-10/preds_superpoint-lg \
  --z-data-path logs/log_svox_train/2025-12-16_14-37-10/z_data.torch \
  --output-path features_svox_train.npz
```


