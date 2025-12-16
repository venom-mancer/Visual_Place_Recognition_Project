## Extension 6.1 – Stage 1: Per-query Feature & Label Extraction

### Goal of Stage 1
- Build a **structured dataset** where each row corresponds to **one query image**, containing:
  - **Features** that describe how confident/uncertain the system is about the Top‑1 retrieval.
  - A **label** telling us whether the Top‑1 prediction is actually correct or not.
- This dataset will be used later to **train the logistic regression model** for adaptive re-ranking.

### Inputs used in Stage 1
- **Prediction text files (`.txt`)** in `--preds-dir`:
  - One file per query.
  - Each file contains the retrieval results (e.g., indices/IDs of the top‑K retrieved database images).
  - These are **not images**, but outputs of the retrieval step.
- **Inliers result files (`.torch`)** in `--inliers-dir`:
  - One file per query.
  - Produced by the image matching (e.g., SuperPoint + LightGlue) between the query and retrieved images.
  - Contain, among other things, the **number of inliers** for the first retrieved image.
- **`z_data` file (`--z-data-path`)**:
  - A `.pt`/`.pth` PyTorch file that stores:
    - `database_utms`: poses of database images.
    - `predictions`: indices of retrieved neighbors.
    - `distances`: descriptor distances (e.g., L2 or dot product distance values) between query and database images.

### What we compute per query
For each query, we compute:

- **Label (`labels[itr]`)**:
  - Use `get_list_distances_from_preds(txt_file_query)` to obtain the geographic distances (in meters) from predictions to the ground truth.
  - If the first distance `geo_dists[0]` is **≤ positive-dist-threshold** (default 25 m):
    - Set `labels[itr] = 1.0` (Top‑1 prediction is **correct**).
  - Otherwise:
    - Set `labels[itr] = 0.0` (Top‑1 prediction is **wrong**).

- **Feature 1 – `num_inliers[itr]`**:
  - Load the corresponding `.torch` file (same stem as the `.txt` file).
  - Read `query_inliers_results[0]["num_inliers"]`.
  - This counts how many good matches (after RANSAC) exist between the query and the **first retrieved image**.

- **Feature 2 – `top1_distance[itr]`**:
  - From `dists[itr][0]` in the `z_data` file.
  - This is the descriptor distance between the query and its **closest** database image.

- **Feature 3 – `peakiness[itr]`**:
  - If there is at least a second neighbor (`dists.shape[1] > 1`):
    - `peakiness[itr] = dists[itr][0] / (dists[itr][1] + 1e-8)`.
  - Else:
    - `peakiness[itr] = 1.0`.
  - This measures how much better the best match is compared to the second best (ambiguity).

- **Feature 4 – `sue_score[itr]` (placeholder for now)**:
  - Currently stored as zeros to keep the interface ready.
  - Later we can extend this to store per-query SUE values if we decide to use them directly as features.

### Output of Stage 1
- All arrays are saved into a single compressed NumPy file using:
  - `np.savez_compressed(args.output_path, ...)`
- The saved file (default: `features.npz`) contains:
  - `labels` – shape `(num_queries,)`
  - `num_inliers` – shape `(num_queries,)`
  - `top1_distance` – shape `(num_queries,)`
  - `peakiness` – shape `(num_queries,)`
  - `sue_score` – shape `(num_queries,)` (currently zeros)

This file is the **input dataset** for the next stages (splitting into train/val/test and training logistic regression).


