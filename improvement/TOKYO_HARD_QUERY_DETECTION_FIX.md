## Summary

On `Amir_Extention_6.1`, **Tokyo-XS adaptive reranking often “missed” hard queries**, meaning `match_queries_preds_adaptive.py` did not run image matching for many truly hard Tokyo queries.  
On `Amir_V2`, this is fixed by making the pipeline **index-stable** and **query-id aligned**, similar in spirit to `extension_6.1_SIA_version2`’s stable, self-contained adaptive workflow.

This document explains:
- **What was improved**
- **Why Tokyo was failing**
- **What changed (files + behavior)**
- **How to run/verify in THIS repo** (without you needing to know paths up front)

---

## What was improved (compared to your current branch)

### 1) Stage 4 output is now aligned to original query IDs

**File**: `extension_6_1/stage_4_apply_logreg_easy_queries.py` (fixed on branch `Amir_V2`)

- Before: the script **dropped NaN rows** and then computed `hard_query_indices` on the shortened array.
- After: the script **keeps the original query index space** (same length as the dataset query list) and expands predictions back to full-length arrays.

### 2) Adaptive image matching now uses the query file ID, not list position

**File**: `match_queries_preds_adaptive.py` (fixed on branch `Amir_V2`)

- Before: it used `enumerate(...)` index (`idx`) to decide hard/easy.
- After: it uses the **query id from the filename** (`int(Path(txt_file).stem)`), which matches how predictions are stored (`preds/0.txt`, `preds/1.txt`, …).

This makes the hard-query list robust to any subtle ordering / filtering issues.

---

## Why Tokyo-XS was failing (root cause)

Tokyo is the dataset where feature edge cases (e.g. NaNs) are more likely to appear. Your original pipeline had two compounding assumptions:

### Root cause A — NaN filtering breaks index mapping

In `stage_4_apply_logreg_easy_queries.py` on the original branch:
- You computed a `valid_mask` and **removed rows with NaNs**.
- Then you wrote `hard_query_indices = np.where(is_hard)[0]`.

Those indices referred to the **filtered array**, not the original query IDs.  
So the `hard_queries_tokyo_xs_*.txt` file often referenced the wrong query IDs.

### Root cause B — adaptive matcher used list position (`idx`) instead of query id

In `match_queries_preds_adaptive.py` on the original branch:
- You checked `if idx not in hard_query_indices`.

Even if `hard_query_indices` was “right”, this assumes “`idx` == query_id”, which is only safe if:
- nothing was filtered earlier
- and file ordering is perfectly consistent with how indices were produced

With Tokyo, these assumptions frequently break → image matching runs for the wrong subset → “hard queries not detected”.

---

## What changed on `Amir_V2` (exact behavior)

### `extension_6_1/stage_4_apply_logreg_easy_queries.py`

- **No longer drops NaN rows** from the output indexing.
- Predicts on valid rows, then **expands `probs/is_easy/is_hard` back to full length**.
- **Conservative default**: any query with NaN features is treated as **HARD** (so we don’t miss real hard queries).
- `hard_queries_output` now contains **true query IDs**.

### `match_queries_preds_adaptive.py`

- Reads `preds/*.txt`, and for each file uses:
  - `q_id = int(stem)` (e.g. `123.txt` → query id 123)
- Compares `q_id` against the hard-query list (which is also query ids).
- Emits a warning if the hard-query file contains ids that do not exist in `preds/`.

---

## How to run / verify for Tokyo in this repo

### 0) Confirm branch

You must be on:
- `Amir_V2`

### 1) Find your Tokyo preds directory

This repo already contains Tokyo logs under:
- `log_tokyo_xs_test/<timestamp>/preds`

Examples observed in this workspace:
- `log_tokyo_xs_test/2025-12-18_14-24-37/preds`
- `log_tokyo_xs_test/2025-12-18_14-43-02/preds`

Pick the one you want to evaluate.

### 2) Produce (or reuse) Tokyo feature file

Most scripts in this repo expect Tokyo features here:
- `data/features_and_predictions/features_tokyo_xs_test_improved.npz`

If this file doesn’t exist yet, generate it using your pipeline’s feature extraction stage (already documented in `docs/`).

### 3) Run Stage 4 to generate hard query list (Tokyo)

Run Stage 4 with a hard-queries output file:
- `data/features_and_predictions/hard_queries_tokyo_xs_test.txt`

If your Tokyo feature file includes labels and you want dataset-specific calibration, use `--calibrate-threshold`.

### 4) Run adaptive image matching only for hard queries

Run `match_queries_preds_adaptive.py` on:
- `--preds-dir log_tokyo_xs_test/<timestamp>/preds`
- `--hard-queries-list data/features_and_predictions/hard_queries_tokyo_xs_test.txt`
- `--out-dir log_tokyo_xs_test/<timestamp>/preds_superpoint-lg_adaptive` (or similar)

### 5) Evaluate adaptive reranking

Use:
- `python -m extension_6_1.stage_5_adaptive_reranking_eval --preds-dir ... --inliers-dir ... --logreg-output ...`

### 6) Quick sanity checks (what “fixed” looks like)

- The console output from Stage 4 should show a **non-trivial number of hard queries** for Tokyo (not near 0% unless Tokyo is genuinely trivial).
- `match_queries_preds_adaptive.py` should report “Hard queries to process: <N>” where `N` matches the hard-query list size (minus any missing ids).
- The number of `.torch` files in the adaptive inliers folder should match the number of queries (easy queries get empty `.torch` for compatibility).

---

## Notes / consistency with existing docs

Some docs mention “probability of being easy” vs “probability of being hard”.  
What matters for the Tokyo bug is **not the class semantics**, but that the **hard query list must map to the correct query IDs**. `Amir_V2` fixes that mapping.


