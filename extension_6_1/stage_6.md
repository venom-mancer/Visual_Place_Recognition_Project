## Extension 6.1 – Stage 6: Running Experiments and Reporting Results

### Goal of Stage 6
- Run the full adaptive re-ranking pipeline on the chosen dataset splits and **collect final metrics**.
- Compare:
  - Retrieval-only (CosPlace, no re-ranking),
  - Always re-ranking (re-rank all queries),
  - Adaptive re-ranking (logistic regression: easy → retrieval-only, hard → re-rank).
- Summarize the trade-off between **accuracy (Recall@N)** and **cost** (how many queries are re-ranked).

### 1. Train logistic regression (SVOX train)
First, make sure you have a feature file for SVOX train:

```bash
python -m vpr_uncertainty.extract_features \
  --preds-dir PATH_SVOX_TRAIN_PREDS \
  --inliers-dir PATH_SVOX_TRAIN_INLIERS \
  --z-data-path PATH_SVOX_TRAIN_Z_DATA \
  --output-path features_svox_train.npz
```

Then train the model:

```bash
python -m vpr_uncertainty.train_logreg \
  --train-features features_svox_train.npz \
  --output-model logreg_svox.pkl
```

### 2. Prepare and evaluate each split
For each split (e.g. `sf_xs` test, `tokyo_xs` test, `svox` test):

#### a) Extract per-query features

```bash
python -m vpr_uncertainty.extract_features \
  --preds-dir PATH_TEST_PREDS \
  --inliers-dir PATH_TEST_INLIERS \
  --z-data-path PATH_TEST_Z_DATA \
  --output-path features_SPLIT.npz
```

#### b) Apply logistic regression (easy / hard decisions)

```bash
python -m vpr_uncertainty.apply_logreg \
  --model-path logreg_svox.pkl \
  --feature-path features_SPLIT.npz \
  --output-path logreg_SPLIT_outputs.npz
```

This produces an `.npz` file with:
- `probs` – \( p(\text{Top‑1 correct}) \) for each query,
- `is_easy` – boolean mask (True = easy),
- `is_hard` – boolean mask (True = hard),
- `labels` – ground-truth Top‑1 correctness (0/1).

#### c) Evaluate adaptive re-ranking Recall@N

```bash
python -m vpr_uncertainty.adaptive_reranking_eval \
  --preds-dir PATH_TEST_PREDS \
  --inliers-dir PATH_TEST_INLIERS \
  --logreg-output logreg_SPLIT_outputs.npz \
  --num-preds 100 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20 100
```

This prints Recall@N for the adaptive strategy on that split.

**Why include Recall@100?**
- The original `reranking.py` script uses the default recall values `[1, 5, 10, 20, 100]`, so keeping `100` here makes the adaptive method **directly comparable** to your existing baselines.
- Recall@100 is less critical for strict localization than Recall@1, but it:
  - Shows an **upper bound** when you allow many candidates.
  - Lets you check that the adaptive strategy does not degrade performance in the **long tail** compared to always re-ranking.
- If desired, you can remove `100` from `--recall-values` and focus on `[1, 5, 10, 20]`; the code will still work the same.

### 3. What to compare in your report
For each dataset (SF-XS test, Tokyo-XS test, SVOX test), you should ideally have:

- **Retrieval-only** (CosPlace, no re-ranking):
  - R@1, R@5, R@10, ...
- **Always re-ranking** (e.g. `reranking.py` over all queries):
  - R@1, R@5, R@10, ...
- **Adaptive re-ranking (our method)**:
  - R@1, R@5, R@10, ...
  - % of queries where `is_hard` is True (fraction of queries that were re-ranked).

You can summarize this in a table like:

| Dataset      | Method              | R@1  | R@5  | R@10 | % queries re-ranked |
|-------------|---------------------|------|------|------|----------------------|
| SF-XS test  | Retrieval-only      | ...  | ...  | ...  | 0%                   |
| SF-XS test  | Always re-ranking   | ...  | ...  | ...  | 100%                 |
| SF-XS test  | Adaptive (logreg)   | ...  | ...  | ...  | X%                   |
| Tokyo-XS    | ...                 | ...  | ...  | ...  | ...                  |
| SVOX test   | ...                 | ...  | ...  | ...  | ...                  |

In the discussion, highlight whether adaptive re-ranking:
- Achieves **similar R@1** to always re-ranking, and
- Reduces the **fraction of queries** for which expensive re-ranking (SuperPoint+LightGlue) is run.


