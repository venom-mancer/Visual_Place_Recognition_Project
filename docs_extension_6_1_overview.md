## Extension 6.1 – Adaptive Re-ranking with Logistic Regression (Concept Summary)

### Visual Place Recognition (VPR) pipeline (what the project does)
- **Goal**: Given a query image, find its location by retrieving the most similar images from a database with known poses.
- **Steps**:
  - **Descriptor extraction**: A VPR model (e.g., NetVLAD, CosPlace, MixVPR, MegaLoc) converts each image into a global descriptor vector.
  - **Retrieval**: For a query, compute distances between its descriptor and all database descriptors and retrieve the top-\(K\) nearest neighbors (e.g., with L2 distance or dot product).
  - **Prediction**: The simplest prediction takes the **Top‑1** retrieved database image as the location estimate.
  - **Evaluation**: Use **Recall@N** (often Recall@1 with a distance threshold like 25 m) to measure how often a correct database image appears in the top-\(N\) results.

### Re-ranking with Image Matching
- **Motivation**: Pure descriptor retrieval can be wrong/ambiguous, especially in visually similar places.
- **Image matching methods** (e.g., SuperPoint + LightGlue, SuperGlue, LoFTR) compare **local features** between:
  - The **query image** and
  - Each retrieved database image.
- They produce **matches** and then a geometric check (e.g., RANSAC) filters them, leaving only **inliers** (consistent matches).
- **Re-ranking**: We can re-rank the retrieved list according to some score derived from local matches (e.g., number of inliers) to improve localization accuracy, but this step is **computationally expensive**.

### Uncertainty / confidence estimation
To decide when to trust retrieval and when to run expensive re-ranking, we estimate **how confident** the system is in its Top‑1 prediction:

- **Top‑1 correctness label**:
  - For each query, we check if the Top‑1 prediction is within a distance threshold (e.g., 25 m).
  - If yes → label = 1 (**correct**), else → label = 0 (**wrong**).
  - In the code, this is stored in `matched_array_for_aucpr`.

- **Existing confidence/uncertainty scores**:
  - **L2-based score**: uses the distance of the Top‑1 retrieved descriptor (`dists[itr][0]`) – smaller distance implies higher confidence.
  - **PA (peakiness) score**: ratio `dists[itr][0] / dists[itr][1]` (Top‑1 over Top‑2 distance) – when this ratio is small, the best match is much better than the second, suggesting higher confidence.
  - **SUE score**: uses the top-\(K\) neighbors and their geographic poses to estimate how spatially concentrated they are; low variance means higher confidence.
  - **Inliers count**: number of inlier matches between the query and the first retrieved image; more inliers usually mean higher confidence.

These scores can be evaluated using precision–recall and AUC-PR, as seen in `vpr_uncertainty/eval.py` and `baselines.py`.

### Hard vs easy queries
- **Intuition**:
  - An **easy** query is one where the retrieval system is quite sure: the Top‑1 match is clearly better than others, descriptors are close, inliers are high, neighbors are spatially consistent.
  - A **hard** query is more ambiguous: descriptors are less distinctive, inliers are low, neighbors are spread out, and Top‑1 is often wrong.
- **From the project PDF**:
  - Hard queries can be characterized by having a **low number of inliers** between the query and the first retrieved image compared to the distribution across all queries.
  - The idea is to **only apply re-ranking on hard queries** to save computation while keeping good accuracy.

### Logistic regression for adaptive re-ranking (Extension 6.1)
- **Goal**: Learn a simple model that, given per-query features, outputs a probability that the Top‑1 prediction is correct (or incorrect).
- **Features per query** (what we will use):
  - `num_inliers`: from image matching (SuperPoint + LightGlue) between the query and its Top‑1 retrieved image.
  - `top1_distance`: the descriptor distance for the Top‑1 retrieved image (`dists[itr][0]`).
  - `peakiness`: ratio between Top‑1 and Top‑2 distances (`dists[itr][0] / dists[itr][1]`).
  - `sue_score`: spatial uncertainty estimate from the top-\(K\) neighbors.
- **Label per query**:
  - `correct = 1` if Top‑1 is within 25 m; `correct = 0` otherwise.
- **Logistic regression**:
  - Input: the feature vector `[num_inliers, top1_distance, peakiness, sue_score]`.
  - Output: \(p(\text{Top‑1 is correct} \mid \text{features})\).
  - We will train it on **SVOX train** (excluding GSV-XS), validate it on **SF-XS val**, and evaluate on all test sets.

### Adaptive re-ranking rule
- After training logistic regression, we define a rule:
  - If \(p(\text{Top‑1 correct}) \geq T\) → **query is easy** → **skip re-ranking** (use retrieval-only result).
  - If \(p(\text{Top‑1 correct}) < T\) → **query is hard** → **apply re-ranking** (SuperPoint + LightGlue), and use the re-ranked list.
- The threshold \(T\) is chosen based on validation performance (e.g., on SF-XS val), trading off:
  - **Accuracy** (e.g., Recall@1) and
  - **Cost** (fraction of queries where we run the expensive re-ranking).


