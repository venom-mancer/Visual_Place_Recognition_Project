## Adaptive Re-Ranking (Extension 6.1)

This folder contains the code + artifacts for **adaptive re-ranking** in VPR.

- **Goal**: save computation by running expensive image matching / re-ranking only when a query is likely “hard”.
- **What we evaluated**: both **full (non-adaptive) re-ranking** and **adaptive re-ranking**, using **two VPR methods** and **two image matchers**, across the provided test sets.

### Decision rule

We train a Logistic Regression classifier on a single feature:
- **feature**: `inliers_top1` = number of inliers between the query and the retrieval top‑1 candidate

The LR outputs:
- a **probability that the retrieval top‑1 is correct**, based only on `inliers_top1`

Then:
- **EASY** if this probability is **>= threshold `t`** -> **don’t re-rank**
- **HARD** otherwise -> **re-rank**

Note: we do **not** choose `t` from the test set. We choose `t` on validation, then apply it to test sets. The **%easy/%hard on each test set** is what reflects each dataset’s inlier distribution.

## Folder map (complete)

```
adaptive_reranking/
  README.md
  build_lr_dataset.py              # build CSV rows (inliers_top1, is_top1_correct)
  build_all_training_csvs.py       # build 3 train CSVs (sun, night, combined)
  tune_lr_hyperparameters.py       # tune LR (C + decision threshold), save models + plots
  adaptive_reranking_eval.py       # evaluate adaptive strategy on a single dataset
  batch_eval_combo.py              # batch-eval a combo: writes summary.csv/summary.txt/raw.log
  train_lr_from_csv.py             # optional: plain LR training (not used by main pipeline)
  csv_files/
    <OneFolderPerVPR+Matcher>/
      lr_data_*_svox_train_sun.csv
      lr_data_*_svox_train_night.csv
      lr_data_*_svox_train.csv
      lr_data_*_sf_xs_val.csv
      tuning_results/
        lr_model_sun_C*.pkl
        lr_model_night_C*.pkl
        lr_model_combined_C*.pkl
        threshold_vs_r1_plot.png
        tuning_summary.txt
      batch_eval_YYYYMMDD_HHMMSS/
        summary.csv
        summary.txt
        raw.log
```

## Example (one combo end-to-end)

Below is a **single worked example** (pick one combo and follow it end-to-end). You can repeat the same steps for the other method/matcher choices by changing `--method` and `--matcher` and the `logs_*` names.

### Conventions

- Run these from **`VPR-methods-evaluation/`**:
  - `python main.py ...`
  - `python ..\match_queries_preds.py ...`
  - `python ..\reranking.py ...`
- Run these from the **project root**:
  - `python adaptive_reranking\...`

### A) Full pipeline (non-adaptive re-ranking)

1) **Run VPR retrieval** (produces `preds/*.txt`):

```powershell
cd VPR-methods-evaluation
python main.py --method=cosplace --backbone=ResNet50 --descriptors_dimension=512 --image_size 512 512 `
  --database_folder ../data/sf_xs/test/database `
  --queries_folder  ../data/sf_xs/test/queries `
  --num_preds_to_save 20 --recall_values 1 5 10 20 `
  --log_dir logs_example_sf_xs_test
```

2) **Compute top‑20 inliers** (matcher run for K=20):

```powershell
cd VPR-methods-evaluation
python ..\match_queries_preds.py `
  --preds-dir logs/logs_example_sf_xs_test/<timestamp>/preds `
  --out-dir   logs/logs_example_sf_xs_test/<timestamp>/preds_loftr_full20 `
  --matcher loftr --device cuda --im-size 512 --num-preds 20
```

3) **Full re-ranking** (re-rank every query):

```powershell
cd VPR-methods-evaluation
python ..\reranking.py `
  --preds-dir   logs\logs_example_sf_xs_test\<timestamp>\preds `
  --inliers-dir logs\logs_example_sf_xs_test\<timestamp>\preds_loftr_full20 `
  --num-preds 20 --matcher loftr --vpr-method cosplace
```

### B) Adaptive pipeline (LR decides whether to re-rank)

1) **Compute top‑1 inliers** (cheap, used only for the LR decision):

We compute **top‑1 inliers for every split**:
- **train + val**: to build the LR CSVs (`inliers_top1` + label)
- **test**: for the LR decision at test time (decide EASY vs HARD for each query)

```powershell
cd VPR-methods-evaluation
python ..\match_queries_preds.py `
  --preds-dir logs/logs_example_sf_xs_test/<timestamp>/preds `
  --matcher loftr --device cuda --im-size 512 --num-preds 1
```

2) **Build training CSVs** (SVOX train sun/night/combined):

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

3) **Build validation CSV** (SF‑XS val, also needs top‑1 inliers):

```powershell
python adaptive_reranking\build_lr_dataset.py `
  --preds-dirs   VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --inliers-dirs VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr `
  --dataset-names sf_xs_val `
  --out-csv adaptive_reranking/csv_files/Cosplace_Loftr/lr_data_cosplace_loftr_sf_xs_val.csv `
  --vpr-method cosplace `
  --matcher-method loftr
```

4) **Tune LR + choose threshold on validation** (also needs validation top‑20 inliers):

Why do we compute **top‑20 inliers for SF‑XS val**?
- To plot/analyze how validation **R@1 changes vs threshold** (the script needs inliers for the HARD branch to simulate “re-rank top‑K” on the validation set).
- In a *real* adaptive pipeline on test sets, we **do not** want to precompute top‑20 for every query (that would waste the compute we’re trying to save). Test-time only needs **top‑1 for all queries**, then **top‑K only for queries predicted HARD**.

```powershell
python adaptive_reranking\tune_lr_hyperparameters.py `
  --csv-folder adaptive_reranking/csv_files/Cosplace_Loftr `
  --val-preds-dir         VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds `
  --val-top20-inliers-dir VPR-methods-evaluation/logs/logs_cosplace_sf_xs_val/<timestamp>/preds_loftr_full20
```

This step trains and saves **3 LR models** (sun / night / combined) under:
- `adaptive_reranking/csv_files/Cosplace_Loftr/tuning_results/`

5) **Run adaptive evaluation on SF‑XS test**:

Tip: instead of manually running each LR model on each test set, you can use `batch_eval_combo.py`.
For a single VPR+matcher combo you typically have **3 LR models** (sun/night/combined) and **4 test sets**, so that’s **12 runs** — `batch_eval_combo.py` automates this and writes `summary.csv`, `summary.txt`, and `raw.log`.

```powershell
cd VPR-methods-evaluation
python ..\adaptive_reranking\adaptive_reranking_eval.py `
  --preds-dir logs\logs_example_sf_xs_test\<timestamp>\preds `
  --top1-inliers-dir logs\logs_example_sf_xs_test\<timestamp>\preds_loftr `
  --lr-model ..\adaptive_reranking\csv_files\Cosplace_Loftr\tuning_results\lr_model_combined_C0.01.pkl `
  --num-preds 20 --matcher loftr --device cuda --im-size 512
```

## Tuning objective (what the script actually optimizes)

`tune_lr_hyperparameters.py` does:
- **Selects `C`** by maximizing **ROC-AUC** on the validation CSV.
- **Selects threshold `t`** by maximizing **classification accuracy** of the easy/hard decision on validation.

It still plots **Threshold vs R@1** for analysis and reporting, but the selected threshold is not chosen by maximizing R@1 (SF‑XS val is often very easy).

## Windows note (important)

Run evaluation from `VPR-methods-evaluation/` so that paths inside `preds/*.txt` resolve correctly on Windows.

