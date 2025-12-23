## SIA-style Fix: `inliers_top1` gate (solves Tokyo “skip reranking”)

### Problem (what we saw on Tokyo)

In the original adaptive pipeline, Tokyo-XS often ended up with **0% hard queries**, which means:
- **No image matching / reranking was executed**
- Adaptive performance collapsed to **baseline retrieval** (R@1 ≈ 65.1%)

This is **not** just an indexing issue — it’s mainly a **generalization / calibration** issue:
- The 3-way retrieval-feature models (the “easy_score” LR models trained/validated on other data) can output **P(easy) ≈ 1.0** for almost all Tokyo queries, including wrong ones.
- With a saved SF-XS validation threshold, this yields **hard ≈ 0%**.

### Why SIA v2 doesn’t skip

SIA v2 uses a gate based on a single robust feature:
- **`inliers_top1`** = number of inliers between query and the retrieval top‑1 candidate

This feature is:
- available **before** full reranking (only needs top‑1 matching)
- strongly correlated with top‑1 correctness across datasets
- more stable than purely retrieval-confidence features

So the gate produces a **non-empty hard subset** on Tokyo and triggers reranking where needed.

---

## Implementation on `Amir_V2`

### New scripts

- `tools/sia_inliers_top1_train.py`
  - Builds a training set from `preds/*.txt` + `.torch` (top‑20 is fine; we only read the first match result)
  - Trains Logistic Regression to predict **P(top1_correct)** from **`inliers_top1`**
  - Chooses:
    - **C** by validation ROC-AUC
    - **threshold** by maximizing **adaptive Recall@1** on validation

- `tools/sia_inliers_top1_apply.py`
  - Applies the trained gate to a dataset split and writes:
    - `.npz` with `probs/is_easy/is_hard/hard_query_indices`
    - `.txt` list of hard query IDs

### Training configuration used

- **Train**: SVOX train
  - `logs/log_svox_train/2025-12-16_17-08-46/preds`
  - `logs/log_svox_train/2025-12-16_17-08-46/preds_superpoint-lg`
- **Validation**: SF-XS val
  - `logs/log_sf_xs_val/2025-12-16_21-55-53/preds`
  - `logs/log_sf_xs_val/2025-12-16_21-55-53/preds_superpoint-lg`

Trained model saved to:
- `trained_models/sia_inliers_top1_gate.pkl`

Training output (printed):
- `val_roc_auc ≈ 0.960`
- `optimal_threshold(P(easy)) ≈ 0.355`

---

## Tokyo-XS results (after SIA-style gate)

Applied gate on:
- `log_tokyo_xs_test/2025-12-18_14-43-02/preds`
- `log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg`

Outputs:
- `temp/sia_tokyo_gate.npz`
- `temp/sia_tokyo_hard.txt` (**129 hard queries**)

### Key outcome: reranking is no longer skipped

- **Hard queries detected**: **129 / 315 = 41.0%**
- **Time savings (matching)**: **59.0%**

### R@1 comparison

- **Baseline retrieval (CosPlace)**: **65.1%**
- **Full reranking (SuperPoint+LG)**: **83.17%**
- **Adaptive (SIA-style inliers_top1 gate)**: **83.5%**

This means the SIA-style gate:
- avoids the Tokyo “skip” failure mode (hard% is not 0)
- recovers essentially **full reranking** accuracy while saving compute

---

## How to reproduce (commands)

### 1) Train gate

```bash
python tools/sia_inliers_top1_train.py ^
  --train-preds logs/log_svox_train/2025-12-16_17-08-46/preds ^
  --train-inliers logs/log_svox_train/2025-12-16_17-08-46/preds_superpoint-lg ^
  --val-preds logs/log_sf_xs_val/2025-12-16_21-55-53/preds ^
  --val-inliers logs/log_sf_xs_val/2025-12-16_21-55-53/preds_superpoint-lg ^
  --out-model trained_models/sia_inliers_top1_gate.pkl
```

### 2) Apply to Tokyo

```bash
python tools/sia_inliers_top1_apply.py ^
  --model trained_models/sia_inliers_top1_gate.pkl ^
  --preds-dir log_tokyo_xs_test/2025-12-18_14-43-02/preds ^
  --inliers-dir log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg ^
  --out-npz temp/sia_tokyo_gate.npz ^
  --out-hard-txt temp/sia_tokyo_hard.txt
```

### 3) Evaluate adaptive reranking

```bash
python -m extension_6_1.stage_5_adaptive_reranking_eval ^
  --preds-dir log_tokyo_xs_test/2025-12-18_14-43-02/preds ^
  --inliers-dir log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg ^
  --logreg-output temp/sia_tokyo_gate.npz ^
  --num-preds 20 ^
  --positive-dist-threshold 25 ^
  --recall-values 1
```


