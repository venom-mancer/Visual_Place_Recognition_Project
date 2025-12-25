## Amir_V2 Fix: `inliers_top1` gate (true adaptive pipeline, fixes Tokyo skip)

### What changed

We switched the adaptive “hard/easy” decision to a **robust, dataset-stable signal**:
- **`inliers_top1`** (stored as `num_inliers_top1`): number of matcher inliers between the query and its **retrieval top‑1** candidate.

This avoids the failure mode where retrieval-only confidence features can become overconfident on Tokyo and predict ~0% hard queries.

### Why this fixes Tokyo “skip reranking”

Previously, Tokyo could end up with **0 detected hard queries**, so the pipeline skipped matching/reranking entirely.

With Amir_V2:
- `inliers_top1` varies meaningfully on Tokyo.
- Logistic regression on `inliers_top1` produces a **non-empty hard subset**, so **top‑K matching runs only for those**.

### The correct adaptive pipeline (important)

This is the intended execution order:

1. **Retrieval** (VPR) → `preds/*.txt`
2. **Top‑1 matching for ALL queries** → get `inliers_top1` (cheap)
3. **Logistic Regression gate** → predict **hard vs easy**
4. **Top‑K matching ONLY for hard queries** (expensive part)
5. **Adaptive reranking evaluation**

### Tools added (Amir_V2)

- **Train gate** (uses `num_inliers_top1` + labels from feature npz):
  - `tools/amir_v2_inliers_top1_train.py`
- **Apply gate** (creates `is_easy/is_hard` + hard query list):
  - `tools/amir_v2_inliers_top1_apply.py`
- **Run true adaptive pipeline end-to-end for one dataset**:
  - `tools/amir_v2_run_true_adaptive_pipeline.py`

### Tokyo-XS outcome (high level)

On Tokyo-XS, Amir_V2 detects a meaningful hard fraction (≈ 41%) and reaches **~full reranking R@1** while saving matching compute (≈ 59%).


