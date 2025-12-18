# Adaptive Re-Ranking with Logistic Regression

This folder contains all scripts and data for **Extension 6.1: Adaptive Re-Ranking** using Logistic Regression to decide when to apply expensive image matching and re-ranking.

## 📁 Folder Structure

```
adaptive_reranking/
├── build_lr_dataset.py              # Build LR training dataset from VPR predictions + matcher inliers
├── train_lr_from_csv.py              # Train LR model from CSV dataset
├── tune_lr_hyperparameters.py        # Tune C and threshold hyperparameters
├── adaptive_reranking_eval.py        # Evaluate adaptive re-ranking (Option B: adaptive matching)
├── csv_files/                        # LR training/validation datasets
│   ├── lr_data_cosplace_loftr_svox_train.csv
│   ├── lr_data_cosplace_loftr_sf_xs_val.csv
│   ├── lr_data_mixvpr_superpoint-lg_svox_train.csv
│   └── lr_data_mixvpr_superpoint-lg_sf_xs_val.csv
├── tuning_results_cosplace_loftr/          # Trained LR models (one folder per VPR+Matcher combo)
│   ├── logreg_best_C10.00.pkl
│   └── tuning_summary.txt
└── tuning_results_mixvpr_superpoint-lg/
    ├── logreg_best_C3.00.pkl
    └── tuning_summary.txt
```

## 🎯 Overview

**Adaptive Re-Ranking (Option B)** makes the expensive image matching step (LoFTR/SuperGlue) adaptive:

1. **For every query**: Run matcher for **top-1 only** (cheap, needed for LR decision)
2. **Use trained LR model** to predict: Is top-1 likely correct? (`P(correct | inliers_top1)`)
3. **EASY queries** (high probability): Skip further matching, use **retrieval-only ranking**
4. **HARD queries** (low probability): Run matcher for **top-20**, then **re-rank by inliers**

**Result**: ~70% of queries skip expensive matching, saving ~3× computation while maintaining accuracy.

## 📋 Pipeline Steps

### Step 1: Generate VPR Retrieval Results

Run CosPlace/MixVPR retrieval on your datasets (this is done outside this folder):

```powershell
cd VPR-methods-evaluation
python main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=512 ...
```

### Step 2: Generate Top-1 Matcher Inliers (for LR decision)

Run matcher (LoFTR/SuperGlue) for **top-1 only** on training/validation sets:

```powershell
cd VPR-methods-evaluation
python ..\match_queries_preds.py `
  --preds-dir logs/logs_cosplace_svox_train_sun/2025-XX-XX/preds `
  --matcher loftr `
  --device cuda `
  --im-size 512 `
  --num-preds 1
```

**Repeat for:**
- SVOX train (Sun + Night) → for LR training
- SF-XS val → for LR validation

### Step 3: Build LR Training Dataset

Extract features (`inliers_top1`) and labels (`is_top1_correct`) from predictions + inliers:

```powershell
python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs VPR-methods-evaluation/logs/logs_cosplace_svox_train_sun/2025-XX-XX/preds `
              VPR-methods-evaluation/logs/logs_cosplace_svox_train_night/2025-XX-XX/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_cosplace_svox_train_sun/2025-XX-XX/preds_loftr `
                 VPR-methods-evaluation/logs/logs_cosplace_svox_train_night/2025-XX-XX/preds_loftr `
  --dataset-names svox_train_sun svox_train_night `
  --out-csv adaptive_reranking\csv_files\lr_data_cosplace_loftr_svox_train.csv `
  --vpr-method cosplace `
  --matcher-method loftr
```

**For validation dataset:**

```powershell
python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/2025-XX-XX/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/2025-XX-XX/preds_loftr `
  --dataset-names sf_xs_val `
  --out-csv adaptive_reranking\csv_files\lr_data_cosplace_loftr_sf_xs_val.csv `
  --vpr-method cosplace `
  --matcher-method loftr
```

### Step 4: Tune LR Hyperparameters

Find optimal `C` (regularization) and `threshold` (decision boundary):

```powershell
python adaptive_reranking\tune_lr_hyperparameters.py `
  --train-csv adaptive_reranking\csv_files\lr_data_cosplace_loftr_svox_train.csv `
  --val-csv adaptive_reranking\csv_files\lr_data_cosplace_loftr_sf_xs_val.csv `
  --output-dir adaptive_reranking\tuning_results_cosplace_loftr `
  --C-values 0.01 0.1 0.3 1.0 3.0 10.0 `
  --threshold-values 0.3 0.4 0.5 0.6 0.7
```

**Output**: `tuning_results_cosplace_loftr/logreg_best_C*.pkl` (saved model + scaler + threshold)

### Step 5: Evaluate Adaptive Re-Ranking on Test Sets

For each test set (SF-XS test, Tokyo-XS test, SVOX test Sun/Night):

```powershell
cd VPR-methods-evaluation
python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_cosplace_sf_xs_test\2025-XX-XX\preds `
  --top1-inliers-dir logs\logs_cosplace_sf_xs_test\2025-XX-XX\preds_loftr `
  --lr-model ..\adaptive_reranking\tuning_results_cosplace_loftr\logreg_best_C10.00.pkl `
  --num-preds 20 `
  --matcher loftr `
  --device cuda `
  --im-size 512 `
  --wandb-project vpr_ext6_1 `
  --wandb-run-name sf_xs_test_cos_loftr_adaptive
```

**Output**: Recall@1/5/10/20, % easy/hard queries, avg LoFTR pairs per query, total runtime.

## 🔄 Using for Other VPR+Matcher Combinations

### Example: MixVPR + LoFTR

1. **Generate MixVPR predictions** (outside this folder)
2. **Generate top-1 LoFTR inliers** for training/validation
3. **Build LR dataset**:
   ```powershell
   python adaptive_reranking\build_lr_dataset.py `
     --preds-dirs ... `
     --inliers-dirs ... `
     --dataset-names ... `
     --out-csv adaptive_reranking\csv_files\lr_data_mixvpr_loftr_svox_train.csv `
     --vpr-method mixvpr `
     --matcher-method loftr
   ```
4. **Tune hyperparameters**:
   ```powershell
   python adaptive_reranking\tune_lr_hyperparameters.py `
     --train-csv adaptive_reranking\csv_files\lr_data_mixvpr_loftr_svox_train.csv `
     --val-csv adaptive_reranking\csv_files\lr_data_mixvpr_loftr_sf_xs_val.csv `
     --output-dir adaptive_reranking\tuning_results_mixvpr_loftr
   ```
5. **Evaluate adaptive re-ranking**:
   ```powershell
   cd VPR-methods-evaluation
   python ..\adaptive_reranking\adaptive_reranking_eval.py `
     --preds-dir ... `
     --top1-inliers-dir ... `
     --lr-model ..\adaptive_reranking\tuning_results_mixvpr_loftr\logreg_best_C*.pkl `
     --matcher loftr ...
   ```

### Example: CosPlace + SuperGlue (SuperPoint+LightGlue)

Same steps, but:
- Use `--matcher superpoint-lg` instead of `--matcher loftr`
- Use `--matcher-method superpoint-lg` in `build_lr_dataset.py`
- Create `tuning_results_cosplace_superpoint-lg/` folder

### Full example: MixVPR + SuperPoint+LightGlue (what you ran)

Below is the **exact pipeline** used for `MixVPR + SuperPoint+LightGlue` on SVOX / SF-XS / Tokyo-XS.

- **Conventions**:
  - Commands with `python main.py` are run from `VPR-methods-evaluation/`
  - Commands with `python adaptive_reranking\...` or `python match_queries_preds.py` are run from the **project root**

#### 1) Train LR: SVOX train (Sun + Night) + SF-XS val

- **Phase 1 – VPR on SVOX train (MixVPR)** (run from `VPR-methods-evaluation/`):

```powershell
cd VPR-methods-evaluation
python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/svox/images/train/gallery `
  --queries_folder ../data/svox/images/train/queries_sun `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_svox_train_sun

python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/svox/images/train/gallery `
  --queries_folder ../data/svox/images/train/queries_night `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_svox_train_night
```

- **Phase 2 – SuperPoint+LightGlue top‑1 inliers on SVOX train** (run from **project root**):

```powershell
python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_train_sun/2025-12-17_21-10-57/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1

python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_train_night/2025-12-17_21-16-22/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1
```

- **Phase 3 – Build LR train CSV (`lr_data_mixvpr_superpoint-lg_svox_train.csv`)** (project root):

```powershell
python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs VPR-methods-evaluation/logs/logs_mixvpr_svox_train_sun/2025-12-17_21-10-57/preds `
                 VPR-methods-evaluation/logs/logs_mixvpr_svox_train_night/2025-12-17_21-16-22/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_mixvpr_svox_train_sun/2025-12-17_21-10-57/preds_superpoint-lg `
                 VPR-methods-evaluation/logs/logs_mixvpr_svox_train_night/2025-12-17_21-16-22/preds_superpoint-lg `
  --dataset-names svox_train_sun svox_train_night `
  --out-csv adaptive_reranking\csv_files\lr_data_mixvpr_superpoint-lg_svox_train.csv `
  --vpr-method mixvpr `
  --matcher-method superpoint-lg
```

- **SF-XS val: VPR + top‑1 inliers + LR val CSV**:

```powershell
cd VPR-methods-evaluation
python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/sf_xs/val/database `
  --queries_folder ../data/sf_xs/val/queries `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_sf_xs_val
```

```powershell
cd ..
python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_val/2025-12-17_21-46-57/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1

python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_val/2025-12-17_21-46-57/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_val/2025-12-17_21-46-57/preds_superpoint-lg `
  --dataset-names sf_xs_val `
  --out-csv adaptive_reranking\csv_files\lr_data_mixvpr_superpoint-lg_sf_xs_val.csv `
  --vpr-method mixvpr `
  --matcher-method superpoint-lg
```

- **Tune LR (`tuning_results_mixvpr_superpoint-lg`)**:

```powershell
python adaptive_reranking\tune_lr_hyperparameters.py `
  --train-csv adaptive_reranking\csv_files\lr_data_mixvpr_superpoint-lg_svox_train.csv `
  --val-csv adaptive_reranking\csv_files\lr_data_mixvpr_superpoint-lg_sf_xs_val.csv `
  --output-dir adaptive_reranking\tuning_results_mixvpr_superpoint-lg
```

The best model is saved as  
`adaptive_reranking\tuning_results_mixvpr_superpoint-lg\logreg_best_C3.00.pkl`.

#### 2) Adaptive pipeline on test sets (SVOX test Sun/Night, SF-XS test, Tokyo-XS test)

- **Phase 1 – VPR on each test set** (run from `VPR-methods-evaluation/`):

```powershell
python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/svox/images/test/gallery `
  --queries_folder ../data/svox/images/test/queries_sun `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_svox_test_sun

python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/svox/images/test/gallery `
  --queries_folder ../data/svox/images/test/queries_night `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_svox_test_night

python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/sf_xs/test/database `
  --queries_folder ../data/sf_xs/test/queries `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_sf_xs_test

python main.py --method=mixvpr --backbone=ResNet50 --descriptors_dimension=4096 --image_size 320 320 `
  --database_folder ../data/tokyo_xs/test/database `
  --queries_folder ../data/tokyo_xs/test/queries `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_mixvpr_tokyo_xs_test
```

- **Phase 2 – SuperPoint+LG top‑1 inliers for all test sets** (project root):

```powershell
python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_test_sun/2025-12-17_22-49-10/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1

python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_test_night/2025-12-17_23-02-33/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1

python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_test/2025-12-17_23-07-12/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1

python match_queries_preds.py --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_tokyo_xs_test/2025-12-17_23-13-42/preds `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 1
```

- **Phase 3 – Adaptive evaluation (run from `VPR-methods-evaluation/`)**:

```powershell
cd VPR-methods-evaluation

python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_mixvpr_svox_test_sun\2025-12-17_22-49-10\preds `
  --top1-inliers-dir logs\logs_mixvpr_svox_test_sun\2025-12-17_22-49-10\preds_superpoint-lg `
  --lr-model ..\adaptive_reranking\tuning_results_mixvpr_superpoint-lg\logreg_best_C3.00.pkl `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20 `
  --wandb-project vpr_ext6_1 --wandb-run-name svox_test_sun_mixvpr_superpoint-lg_adaptive

python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_mixvpr_svox_test_night\2025-12-17_23-02-33\preds `
  --top1-inliers-dir logs\logs_mixvpr_svox_test_night\2025-12-17_23-02-33\preds_superpoint-lg `
  --lr-model ..\adaptive_reranking\tuning_results_mixvpr_superpoint-lg\logreg_best_C3.00.pkl `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20 `
  --wandb-project vpr_ext6_1 --wandb-run-name svox_test_night_mixvpr_superpoint-lg_adaptive

python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_mixvpr_sf_xs_test\2025-12-17_23-07-12\preds `
  --top1-inliers-dir logs\logs_mixvpr_sf_xs_test\2025-12-17_23-07-12\preds_superpoint-lg `
  --lr-model ..\adaptive_reranking\tuning_results_mixvpr_superpoint-lg\logreg_best_C3.00.pkl `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20 `
  --wandb-project vpr_ext6_1 --wandb-run-name sf_xs_test_mixvpr_superpoint-lg_adaptive

python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_mixvpr_tokyo_xs_test\2025-12-17_23-13-42\preds `
  --top1-inliers-dir logs\logs_mixvpr_tokyo_xs_test\2025-12-17_23-13-42\preds_superpoint-lg `
  --lr-model ..\adaptive_reranking\tuning_results_mixvpr_superpoint-lg\logreg_best_C3.00.pkl `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20 `
  --wandb-project vpr_ext6_1 --wandb-run-name tokyo_xs_test_mixvpr_superpoint-lg_adaptive
```

#### 3) Full (non‑adaptive) pipeline for comparison

- **Phase 1 – Get SuperPoint‑LG inliers for top‑20** (project root):

```powershell
python match_queries_preds.py `
  --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_test_sun/2025-12-17_22-49-10/preds `
  --out-dir  VPR-methods-evaluation/logs/logs_mixvpr_svox_test_sun/2025-12-17_22-49-10/preds_superpoint-lg_full20 `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20

python match_queries_preds.py `
  --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_svox_test_night/2025-12-17_23-02-33/preds `
  --out-dir  VPR-methods-evaluation/logs/logs_mixvpr_svox_test_night/2025-12-17_23-02-33/preds_superpoint-lg_full20 `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20

python match_queries_preds.py `
  --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_test/2025-12-17_23-07-12/preds `
  --out-dir  VPR-methods-evaluation/logs/logs_mixvpr_sf_xs_test/2025-12-17_23-07-12/preds_superpoint-lg_full20 `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20

python match_queries_preds.py `
  --preds-dir VPR-methods-evaluation/logs/logs_mixvpr_tokyo_xs_test/2025-12-17_23-13-42/preds `
  --out-dir  VPR-methods-evaluation/logs/logs_mixvpr_tokyo_xs_test/2025-12-17_23-13-42/preds_superpoint-lg_full20 `
  --matcher superpoint-lg --device cuda --im-size 512 --num-preds 20
```

- **Phase 2 – Re-ranking (run from `VPR-methods-evaluation/`)**:

```powershell
cd VPR-methods-evaluation

python ..\reranking.py `
  --preds-dir   logs\logs_mixvpr_svox_test_sun\2025-12-17_22-49-10\preds `
  --inliers-dir logs\logs_mixvpr_svox_test_sun\2025-12-17_22-49-10\preds_superpoint-lg_full20 `
  --num-preds 20 --matcher superpoint-lg --vpr-method mixvpr

python ..\reranking.py `
  --preds-dir   logs\logs_mixvpr_svox_test_night\2025-12-17_23-02-33\preds `
  --inliers-dir logs\logs_mixvpr_svox_test_night\2025-12-17_23-02-33\preds_superpoint-lg_full20 `
  --num-preds 20 --matcher superpoint-lg --vpr-method mixvpr

python ..\reranking.py `
  --preds-dir   logs\logs_mixvpr_sf_xs_test\2025-12-17_23-07-12\preds `
  --inliers-dir logs\logs_mixvpr_sf_xs_test\2025-12-17_23-07-12\preds_superpoint-lg_full20 `
  --num-preds 20 --matcher superpoint-lg --vpr-method mixvpr

python ..\reranking.py `
  --preds-dir   logs\logs_mixvpr_tokyo_xs_test\2025-12-17_23-13-42\preds `
  --inliers-dir logs\logs_mixvpr_tokyo_xs_test\2025-12-17_23-13-42\preds_superpoint-lg_full20 `
  --num-preds 20 --matcher superpoint-lg --vpr-method mixvpr
```

## 📊 Expected Results

For **CosPlace + LoFTR** on SVOX test Sun:
- **Easy queries**: ~70-75% (skip expensive matching)
- **Hard queries**: ~25-30% (full LoFTR + re-ranking)
- **Recall@1**: ~85-90% (comparable to full re-ranking)
- **Avg LoFTR pairs/query**: ~6-7 (vs 20 for full re-ranking) → **~3× cost savings**

## 📝 Notes

- **All scripts should be run from the project root** (not from inside `adaptive_reranking/`)
- CSV files are stored in `csv_files/` for organization
- Each VPR+Matcher combination gets its own `tuning_results_*/` folder
- The adaptive script **never overwrites** your existing log folders (reads top-1 inliers, writes nothing)

## 🔧 Dependencies

- `util.py` (in project root) - for reading predictions and computing distances
- `setup_temp_dir.py` (in project root) - for temporary file handling
- `image-matching-models/` - for LoFTR/SuperGlue matching
- Standard ML libraries: `sklearn`, `pandas`, `numpy`, `torch`, `joblib`

