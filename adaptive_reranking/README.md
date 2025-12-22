Adaptive Re-Ranking (Extension 6.1)

This folder implements an adaptive re-ranking strategy for Visual Place Recognition (VPR): run expensive image matching and re-ranking only on “hard” queries, while skipping it on “easy” queries to save compute.

## Overview (what “adaptive” means here)

- **Retrieval stage** (VPR): produces top-K candidates per query (stored as `preds/*.txt` in `VPR-methods-evaluation/logs/...`).
- **Matching stage** (image matcher): produces inlier counts between the query and retrieved candidates (stored as `preds_<matcher>/*.torch`).
- **Decision model** (Logistic Regression): uses **`inliers_top1`** to predict how likely the retrieval top-1 is correct, and decides whether to re-rank.

Decision rule:
- **EASY** if \(P(\text{top1 correct} \mid \text{inliers\_top1}) \ge t\)  -> keep retrieval ranking (no extra matching)
- **HARD** otherwise -> run matcher for top-K and re-rank by inliers

## What’s in this folder

- **`build_lr_dataset.py`**: builds LR datasets (CSV) from `preds/*.txt` + top-1 inliers (`*.torch`), with:
  - **feature**: `inliers_top1`
  - **label**: `is_top1_correct` (computed by distance <= 25m; falls back to positives list if needed)
- **`build_all_training_csvs.py`**: convenience wrapper that builds **three** SVOX-train CSVs (sun, night, combined) into `csv_files/<Combo>/`.
- **`tune_lr_hyperparameters.py`**: trains 3 LR models (sun/night/combined), tunes `C` and selects decision threshold `t` on the validation set, and saves:
  - `lr_model_{sun,night,combined}_C*.pkl`
  - `threshold_vs_r1_plot.png`
  - `tuning_summary.txt`
- **`adaptive_reranking_eval.py`**: evaluates the adaptive strategy on a single test set (does on-the-fly matching for HARD queries only).
- **`batch_eval_combo.py`**: runs `adaptive_reranking_eval.py` for a single combo across multiple datasets and LR models, writing `summary.csv`, `summary.txt`, and `raw.log`.

## Folder structure (current)

The main working area is `csv_files/<ComboName>/` (one folder per VPR+matcher combo). Example layout:

```
adaptive_reranking/
  csv_files/
    Cosplace_Loftr/
      lr_data_..._svox_train_sun.csv
      lr_data_..._svox_train_night.csv
      lr_data_..._svox_train.csv
      lr_data_..._sf_xs_val.csv
      tuning_results/
        lr_model_sun_C*.pkl
        lr_model_night_C*.pkl
        lr_model_combined_C*.pkl
        threshold_vs_r1_plot.png
        tuning_summary.txt
      tuning_results_forceC1/            # optional (when using --force-C 1)
      batch_eval_YYYYMMDD_HHMMSS/
        summary.csv
        summary.txt
        raw.log
```

## End-to-end pipeline (runbook)

The pipeline has 5 building blocks. All paths below assume you’re in the **project root**, unless stated otherwise.

### 1) Run VPR to generate `preds/` (outside this folder)

Run from `VPR-methods-evaluation/` to generate `logs/<log_dir>/<timestamp>/preds`:

```powershell
cd VPR-methods-evaluation
python main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=512 --image_size 512 512 `
  --database_folder ../data/sf_xs/val/database `
  --queries_folder  ../data/sf_xs/val/queries `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_cosplace_sf_xs_val
```

### 2) Compute matcher inliers (`match_queries_preds.py`)

Run from `VPR-methods-evaluation/`.

- **Top-1 only** (needed for LR decision + CSV building):

```powershell
cd VPR-methods-evaluation
python ..\match_queries_preds.py --preds-dir logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --matcher loftr --device cuda --im-size 512 --num-preds 1
```

- **Top-20** (needed for full re-ranking and for threshold analysis on val):

```powershell
cd VPR-methods-evaluation
python ..\match_queries_preds.py --preds-dir logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --out-dir  logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr_full20 `
  --matcher loftr --device cuda --im-size 512 --num-preds 20
```

### 3) Build LR CSVs

- **Training CSVs** (SVOX train sun/night/combined) in one command:

```powershell
python adaptive_reranking\build_all_training_csvs.py `
  --vpr-method cosplace `
  --matcher-method loftr `
  --preds-dir-sun   VPR-methods-evaluation/logs/logs_cosplace_svox_train_sun/<timestamp>/preds `
  --inliers-dir-sun VPR-methods-evaluation/logs/logs_cosplace_svox_train_sun/<timestamp>/preds_loftr `
  --preds-dir-night   VPR-methods-evaluation/logs/logs_cosplace_svox_train_night/<timestamp>/preds `
  --inliers-dir-night VPR-methods-evaluation/logs/logs_cosplace_svox_train_night/<timestamp>/preds_loftr `
  --output-dir csv_files
```

- **Validation CSV** (SF-XS val) using top-1 inliers:

```powershell
python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs   VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr `
  --dataset-names sf_xs_val `
  --out-csv adaptive_reranking/csv_files/Cosplace_Loftr/lr_data_cosplace_loftr_sf_xs_val.csv `
  --vpr-method cosplace `
  --matcher-method loftr
```

### 4) Tune LR models + decision threshold on validation

This script expects that inside `--csv-folder` you have:
- `*_svox_train_sun.csv`, `*_svox_train_night.csv`, `*_svox_train.csv`
- `*_sf_xs_val.csv`

Run:

```powershell
python adaptive_reranking\tune_lr_hyperparameters.py `
  --csv-folder adaptive_reranking/csv_files/Cosplace_Loftr `
  --val-preds-dir        VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --val-top20-inliers-dir VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr_full20
```

Outputs are written to `adaptive_reranking/csv_files/Cosplace_Loftr/tuning_results/` by default.

Optional: force a specific `C` (useful for ablations):

```powershell
python adaptive_reranking\tune_lr_hyperparameters.py `
  --csv-folder adaptive_reranking/csv_files/Cosplace_Loftr `
  --val-preds-dir        VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --val-top20-inliers-dir VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr_full20 `
  --force-C 1.0 `
  --output-dir adaptive_reranking/csv_files/Cosplace_Loftr/tuning_results_forceC1
```

### 5) Evaluate adaptive re-ranking on test sets

Run from `VPR-methods-evaluation/` (important on Windows; see notes below):

```powershell
cd VPR-methods-evaluation
python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_cosplace_sf_xs_test\<timestamp>\preds `
  --top1-inliers-dir logs\logs_cosplace_sf_xs_test\<timestamp>\preds_loftr `
  --lr-model ..\adaptive_reranking\csv_files\Cosplace_Loftr\tuning_results\lr_model_combined_C0.01.pkl `
  --num-preds 20 --matcher loftr --device cuda --im-size 512
```

This reports:
- adaptive Recall@N (`recall@1`, `recall@5`, `recall@10`, `recall@20`)
- % easy / % hard
- matching cost (`avg_total_pairs_incl_top1` and totals)
- runtimes

## Batch evaluation (recommended)

If you have 3 LR models in a tuning folder (`lr_model_{sun,night,combined}_*.pkl`), you can evaluate the combo across datasets with one command:

```powershell
python adaptive_reranking\batch_eval_combo.py `
  --combo-name "Cosplace+Loftr" `
  --matcher loftr `
  --lr-models-dir adaptive_reranking/csv_files/Cosplace_Loftr/tuning_results `
  --sf-xs-test-preds   VPR-methods-evaluation/logs/logs_cosplace_sf_xs_test/<timestamp>/preds `
  --tokyo-xs-test-preds VPR-methods-evaluation/logs/logs_cosplace_tokyo_xs_test/<timestamp>/preds `
  --svox-sun-test-preds  VPR-methods-evaluation/logs/logs_cosplace_svox_test_sun/<timestamp>/preds `
  --svox-night-test-preds VPR-methods-evaluation/logs/logs_cosplace_svox_test_night/<timestamp>/preds
```

It will create a new `batch_eval_YYYYMMDD_HHMMSS/` folder next to `--lr-models-dir` and write:
- `summary.csv`: easy to paste into Excel
- `summary.txt`: readable table
- `raw.log`: full logs for all sub-runs

## How tuning works (what is optimized)

`tune_lr_hyperparameters.py` does two separate selections:

- **Best `C`**: selected on validation via **ROC-AUC** of predicting `is_top1_correct` from `inliers_top1`.
- **Best threshold `t`**: selected on validation by maximizing **classification accuracy** of the easy/hard decision, i.e. whether the LR decision matches the ground-truth “top-1 correct?” label.

The script still produces a **Threshold vs R@1** plot on validation, because the project writeup asks for dataset-based threshold analysis. The chosen threshold is simply not selected by maximizing R@1 (SF-XS val is often too easy and can lead to degenerate thresholds if you optimize R@1 directly).

## Windows notes / common issues

- **Run eval from `VPR-methods-evaluation/`**: many `preds/*.txt` files contain relative paths, so running `adaptive_reranking_eval.py` from elsewhere can trigger file-not-found errors.
- **If you only have `*_full20`**: you can use it as the top-1 source too (top-1 is just index 0).
  - For CLI eval: set `--top1-inliers-dir` to the same `preds_<matcher>_full20` folder.
  - For batch eval: it expects a `preds_<matcher>` folder to exist; easiest workaround is to copy it:

```powershell
robocopy "...\preds_superpoint-lg_full20" "...\preds_superpoint-lg" /E
```

## Dependencies

- `util.py` and `setup_temp_dir.py` live in the project root (imported by these scripts).
- Matchers are loaded via the local `image-matching-models/` package.
- Python deps: `torch`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `joblib`, `tqdm`, `wandb` (optional).

