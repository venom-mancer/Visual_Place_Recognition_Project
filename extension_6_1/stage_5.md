## Extension 6.1 – Stage 5: Adaptive Re-ranking Evaluation

### Goal of Stage 5
- Use the **easy/hard** decisions from the logistic regression model to evaluate an **adaptive pipeline**:
  - **Easy queries** → use **retrieval-only** ranking.
  - **Hard queries** → use **re-ranked** ranking (based on inliers).
- Compute Recall@N for this adaptive strategy.

### Inputs to Stage 5 script
- `--preds-dir`: directory with prediction `.txt` files (retrieval outputs, one per query).
- `--inliers-dir`: directory with `.torch` files (image matching outputs, one per query).
- `--logreg-output`: `.npz` file produced by `apply_logreg.py`, containing:
  - `probs` – predicted probability that Top‑1 is correct.
  - `is_easy` – boolean array (True = easy, False = hard).
  - `is_hard` – boolean array (True = hard, False = easy).
  - `labels` – ground-truth Top‑1 correctness labels.
- Parameters:
  - `--num-preds`: how many predictions to consider for re-ranking.
  - `--positive-dist-threshold`: distance (m) threshold for a prediction to be considered correct.
  - `--recall-values`: list of N values for Recall@N (e.g. `[1, 5, 10, 20, 100]`).

### Behaviour
- For each query (index `i`):
  - Compute the list of geodesic distances from predictions (`geo_dists`) using `get_list_distances_from_preds`.
  - If `is_easy[i]`:
    - Use **retrieval-only** ordering (`geo_dists`) to check if any of the top‑N predictions is within the distance threshold.
  - If `is_hard[i]`:
    - Load inliers from the `.torch` file.
    - Sort predictions by `num_inliers` (as in `reranking.py`) and reorder `geo_dists` accordingly.
    - Use this **re-ranked** ordering to compute Recall@N.
- Aggregate over all queries to report Recall@N for the adaptive method.


