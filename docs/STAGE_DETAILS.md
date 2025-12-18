# Stage Details - Comprehensive Reference

This document provides detailed information about each stage of the adaptive re-ranking pipeline, integrated from the original `stage_*.md` files with updates to reflect the current implementation.

---

## Stage 0: VPR Evaluation (Baseline)

**Goal**: Run Visual Place Recognition to get initial retrieval results.

**Script**: `VPR-methods-evaluation/main.py`

**Outputs**:
- `log_[dataset]/[timestamp]/preds/` - Prediction files (.txt, .jpg)
- `log_[dataset]/[timestamp]/z_data.torch` - Descriptors, distances, poses

See [Pipeline Guide](PIPELINE_GUIDE.md#stage-0-vpr-evaluation-baseline) for detailed commands.

---

## Stage 1: Per-query Feature & Label Extraction

**Goal**: Build a structured dataset where each row corresponds to one query image, containing:
- **Features** that describe how confident/uncertain the system is about the Top-1 retrieval
- A **label** telling us whether the Top-1 prediction is actually correct or not

This dataset will be used later to train the logistic regression model for adaptive re-ranking.

### Inputs

- **Prediction text files (`.txt`)** in `--preds-dir`:
  - One file per query
  - Each file contains retrieval results (indices/IDs of top-K retrieved database images)
  - These are outputs of the retrieval step, not images

- **`z_data` file (`--z-data-path`)**:
  - A `.pt`/`.pth` PyTorch file that stores:
    - `database_utms`: poses of database images
    - `predictions`: indices of retrieved neighbors
    - `distances`: descriptor distances (L2 or dot product) between query and database images

### What We Compute Per Query

#### Label (`labels[itr]`)
- Use `get_list_distances_from_preds(txt_file_query)` to obtain geographic distances (in meters) from predictions to ground truth
- If first distance `geo_dists[0]` ≤ `positive-dist-threshold` (default 25 m):
  - Set `labels[itr] = 1.0` (Top-1 prediction is **correct**)
- Otherwise:
  - Set `labels[itr] = 0.0` (Top-1 prediction is **wrong**)

#### Features (8 total - all available before image matching)

1. **`top1_distance[itr]`**:
   - From `dists[itr][0]` in the `z_data` file
   - Descriptor distance between query and its closest database image
   - Higher = more uncertain

2. **`peakiness[itr]`**:
   - If there is at least a second neighbor: `peakiness[itr] = dists[itr][0] / (dists[itr][1] + 1e-8)`
   - Else: `peakiness[itr] = 1.0`
   - Measures how much better the best match is compared to the second best (ambiguity)
   - Lower = more ambiguous

3. **`sue_score[itr]`** (Spatial Uncertainty Estimate):
   - Uses top-K neighbors and their geographic poses
   - Computes weighted variance of neighbor positions
   - Higher = more spatially uncertain
   - **Fixed**: Normalized distances and adjusted slope to prevent numerical underflow

4. **`topk_distance_spread[itr]`**:
   - Standard deviation of top-5 distances
   - Higher = more spread out (uncertain)

5. **`top1_top2_similarity[itr]`**:
   - Distance ratio (Top2/Top1): `dists[itr][1] / (dists[itr][0] + 1e-8)`
   - Lower = Top-1 and Top-2 are very similar (ambiguous)

6. **`top1_top3_ratio[itr]`**:
   - Distance ratio (Top1/Top3): `dists[itr][0] / (dists[itr][2] + 1e-8)`
   - Lower = Top-1 is not much better than Top-3

7. **`top2_top3_ratio[itr]`**:
   - Distance ratio (Top2/Top3): `dists[itr][1] / (dists[itr][2] + 1e-8)`
   - Captures ambiguity in second-tier candidates

8. **`geographic_clustering[itr]`**:
   - Average pairwise distance of top-K positions
   - Higher = candidates are spread out geographically (uncertain)

### Output

All arrays are saved into a single compressed NumPy file using `np.savez_compressed()`.

The saved file (e.g., `data/features_and_predictions/features_sf_xs_test_improved.npz`) contains:
- `labels` – shape `(num_queries,)` - Top-1 correctness (1 = correct, 0 = wrong)
- `top1_distance` – shape `(num_queries,)`
- `peakiness` – shape `(num_queries,)`
- `sue_score` – shape `(num_queries,)`
- `topk_distance_spread` – shape `(num_queries,)`
- `top1_top2_similarity` – shape `(num_queries,)`
- `top1_top3_ratio` – shape `(num_queries,)`
- `top2_top3_ratio` – shape `(num_queries,)`
- `geographic_clustering` – shape `(num_queries,)`

This file is the **input dataset** for the next stages (training and evaluation).

---

## Stage 2: Feature File Validation & Preparation

**Goal**: Validate and inspect feature files produced by Stage 1 to ensure data quality. Prepare feature files for different dataset splits (train, validation, test). Understand feature distributions before training.

### Inputs

- **Feature files (`.npz`)** produced by Stage 1:
  - `data/features_and_predictions/features_svox_train_improved.npz` – Training features (from SVOX train split)
  - `data/features_and_predictions/features_sf_xs_val_improved.npz` – Validation features (from SF-XS validation split)
  - Test feature files (e.g., `features_sf_xs_test_improved.npz`, `features_tokyo_xs_test_improved.npz`, `features_svox_test_improved.npz`)

### What Stage 2 Involves

#### 1. Feature File Inspection
- **Load and verify** structure of `.npz` files:
  - Check that all required arrays are present: `labels`, `top1_distance`, `peakiness`, `sue_score`, and 5 additional features
  - Verify array shapes are consistent (all should have same length = number of queries)
  - Check data types and value ranges

#### 2. Data Quality Checks
- **Label distribution**:
  - Count queries with `label = 1` (correct Top-1) vs `label = 0` (incorrect Top-1)
  - Check for class imbalance (important for logistic regression training)
- **Feature statistics**:
  - Compute basic statistics (mean, std, min, max) for each feature
  - Check for missing values or invalid entries (NaN, Inf)
  - Identify potential outliers

#### 3. Feature Exploration (Optional)
- **Visualize feature distributions**:
  - Histograms or box plots for each feature
  - Compare feature distributions between correct (`label=1`) and incorrect (`label=0`) queries
  - This helps understand which features are most discriminative

#### 4. Preparation for Training
- Ensure feature files are ready for Stage 3:
  - Training file: `features_svox_train_improved.npz` (required)
  - Validation file: `features_sf_xs_val_improved.npz` (for model validation and threshold selection)
  - All feature arrays should be properly formatted

### Implementation Note

In the current implementation, Stage 2 is **implicitly handled** by:
1. The `load_feature_file()` function in training/application scripts
2. The training script which validates inputs before training
3. Manual inspection when needed using Python scripts or notebooks

For a more formal Stage 2, you could create a dedicated validation script that:
- Loads all feature files
- Performs comprehensive checks
- Generates a validation report
- Flags any issues before proceeding to training

### Why Stage 2 is Important

- **Catch errors early**: Identify issues in feature extraction before training
- **Understand your data**: Know the characteristics of your training set
- **Debug problems**: If training fails, check if the issue is in the data
- **Inform decisions**: Feature statistics can guide hyperparameter choices (e.g., regularization strength)

---

## Stage 3: Train Logistic Regression Model

**Goal**: Use per-query feature files to train a logistic regression model that predicts the probability that Top-1 is correct (i.e., the query is "easy"). This model will later be used to decide whether a query is easy (skip re-ranking) or hard (apply re-ranking).

### Inputs

- **Training feature file**: `data/features_and_predictions/features_svox_train_improved.npz`
  - Contains arrays: `labels`, `top1_distance`, `peakiness`, `sue_score`, and 5 additional features
- **Validation feature file**: `data/features_and_predictions/features_sf_xs_val_improved.npz`
  - Used to check model generalization and find optimal threshold

### Model and Features

- **Features per query** (input to model): 8 improved features
  - `top1_distance`, `peakiness`, `sue_score`
  - `topk_distance_spread`, `top1_top2_similarity`, `top1_top3_ratio`, `top2_top3_ratio`, `geographic_clustering`
- **Label per query**: `labels` (1 = correct, 0 = wrong)
- **Target**: `easy_score = labels` (1 = easy/correct, 0 = hard/wrong)
- **Model**: Logistic Regression classifier
  - Implemented with `sklearn.linear_model.LogisticRegression`
  - `class_weight='balanced'` to handle class imbalance
  - Trained to predict probability that `label = 1` (Top-1 correct/easy)

### Process

1. Load training and validation features
2. Build feature matrix (8 features)
3. Scale features using `StandardScaler` (mean=0, std=1)
4. Train Logistic Regression with `class_weight='balanced'`
5. **Find optimal threshold** on validation set:
   - Test thresholds from 0.1 to 0.95 in 0.01 steps
   - Select threshold that maximizes F1-score (default) or targets specific recall rate
   - This learned threshold prevents hard thresholding
6. Save model bundle

### Output

A serialized model file (e.g., `logreg_easy_queries_optimal.pkl`) containing:
- `model`: Trained Logistic Regression model
- `scaler`: StandardScaler for feature normalization
- `optimal_threshold`: Learned threshold (e.g., 0.410)
- `feature_names`: List of feature names used
- `threshold_method`: Method used to find threshold (e.g., "f1")
- `target_type`: Target variable type (e.g., "easy_score")

This file will be loaded in Stage 4 to compute probabilities for new queries and define the adaptive re-ranking rule.

---

## Stage 4: Apply Model and Define Adaptive Re-ranking Rule

**Goal**: Use the trained logistic regression model to compute, for each query, the probability that Top-1 is correct. Define the adaptive re-ranking rule using the optimal threshold (learned from validation, not hard-coded) to classify queries as easy (skip re-ranking) or hard (apply re-ranking).

### Inputs

- **Trained model file**: `logreg_easy_queries_optimal.pkl` (from Stage 3)
- **Feature file**: `data/features_and_predictions/features_sf_xs_test_improved.npz` (for test split)

### Process

1. Load model bundle (model, scaler, optimal threshold, feature names)
2. Load test features
3. Build feature matrix using expected feature names from model
4. Scale features using saved scaler
5. Predict probabilities: `probs = model.predict_proba(X_scaled)[:, 1]` (probability of being easy)
6. **Apply optimal threshold** (learned, not hard-coded):
   - If `probs >= optimal_threshold` → **easy query** → **skip re-ranking**
   - If `probs < optimal_threshold` → **hard query** → **apply re-ranking**
7. Save outputs

### Outputs

- **`.npz` file** (e.g., `logreg_easy_test.npz`) containing:
  - `probs` – predicted probability that Top-1 is correct (probability of being easy) per query
  - `is_easy` – boolean array: `True` if query is easy (skip re-ranking)
  - `is_hard` – boolean array: `True` if query is hard (apply re-ranking)
  - `labels` – ground-truth Top-1 correctness labels (0/1)
- **Hard queries list** (e.g., `data/features_and_predictions/hard_queries_test.txt`):
  - Text file with one query index per line (indices of hard queries)
  - Used by adaptive image matching script to process only hard queries

### Key Difference from Original Design

- **No hard thresholding**: Uses optimal threshold learned from validation set (e.g., 0.410)
- **Not fixed at 0.5**: Threshold is data-driven and optimal for the validation set
- **Predicts "easy" directly**: More natural than predicting "hard"

---

## Stage 5: Adaptive Image Matching

**Goal**: Run image matching **only** for hard queries identified by the model. This saves computation time by skipping expensive image matching for easy queries.

**Script**: `match_queries_preds_adaptive.py`

### Process

- Load hard query indices from file
- For each hard query: Run SuperPoint + LightGlue matching
- For each easy query: Create empty .torch file (for compatibility)
- Save inlier counts to .torch files

### Output

Directory with .torch files (one per query) containing inlier counts for hard queries.

---

## Stage 6: Evaluate Adaptive Re-ranking

**Goal**: Use the easy/hard decisions from the logistic regression model to evaluate an adaptive pipeline. For easy queries, use retrieval-only ranking. For hard queries, use re-ranked ranking (based on inliers). Compute Recall@N for this adaptive strategy.

### Inputs

- `--preds-dir`: Directory with prediction `.txt` files (retrieval outputs, one per query)
- `--inliers-dir`: Directory with `.torch` files (image matching outputs, one per query)
- `--logreg-output`: `.npz` file produced by Stage 4, containing:
  - `probs` – predicted probability that Top-1 is correct
  - `is_easy` – boolean array (True = easy, False = hard)
  - `is_hard` – boolean array (True = hard, False = easy)
  - `labels` – ground-truth Top-1 correctness labels
- `--num-preds`: How many predictions to consider for re-ranking (default: 20)
- `--positive-dist-threshold`: Distance (m) threshold for a prediction to be considered correct (default: 25)
- `--recall-values`: List of N values for Recall@N (e.g., `[1, 5, 10, 20]`)

### Behavior

- For each query (index `i`):
  1. Compute list of geodesic distances from predictions (`geo_dists`) using `get_list_distances_from_preds`
  2. **If `is_easy[i]`** (easy query):
     - Use **retrieval-only** ordering (`geo_dists`) to check if any of the top-N predictions is within the distance threshold
     - No re-ranking applied (saves computation time)
  3. **If `is_hard[i]`** (hard query):
     - Load inliers from the corresponding `.torch` file
     - Sort predictions by `num_inliers` (as in `reranking.py`) and reorder `geo_dists` accordingly
     - Use this **re-ranked** ordering to compute Recall@N
- Aggregate over all queries to report Recall@N for the adaptive method

### Output

Prints Recall@N metrics (e.g., Recall@1, R@5, R@10, R@20)

### Note on Recall@100

The original `reranking.py` script uses recall values `[1, 5, 10, 20, 100]`. Including Recall@100 makes the adaptive method directly comparable to existing baselines. Recall@100 shows an upper bound when allowing many candidates and checks that the adaptive strategy does not degrade performance in the long tail compared to always re-ranking.

---

## Comparison Summary

For each dataset, compare:

- **Retrieval-only** (CosPlace, no re-ranking): R@1, R@5, R@10, R@20
- **Always re-ranking** (e.g., `reranking.py` over all queries): R@1, R@5, R@10, R@20
- **Adaptive re-ranking** (our method): R@1, R@5, R@10, R@20, and % of queries where `is_hard` is True

See [Results](RESULTS.md) for complete experimental results.

---

*Last updated: 2025-12-18*

