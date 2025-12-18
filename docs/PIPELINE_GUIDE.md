# Pipeline Guide - Complete Workflow

## Overview

This guide walks through the complete adaptive re-ranking pipeline from VPR evaluation to final results.

---

## Pipeline Stages

### Stage 0: VPR Evaluation (Baseline)

**Goal**: Run Visual Place Recognition to get initial retrieval results.

**Script**: `VPR-methods-evaluation/main.py`

```bash
python VPR-methods-evaluation/main.py \
  --num_workers 4 \
  --batch_size 32 \
  --log_dir log_sf_xs_test \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/gallery \
  --queries_folder data/sf_xs/test/queries \
  --num_preds_to_save 20 \
  --recall_values 1 5 10 20 \
  --save_for_uncertainty \
  --device cuda
```

**Outputs**:
- `log_sf_xs_test/[timestamp]/preds/` - Prediction files (.txt, .jpg)
- `log_sf_xs_test/[timestamp]/z_data.torch` - Descriptors, distances, poses

---

### Stage 1: Feature Extraction (No Inliers)

**Goal**: Extract retrieval-based features available **before** image matching. Build a structured dataset where each row corresponds to one query image, containing features that describe how confident/uncertain the system is about the Top-1 retrieval, and a label telling us whether the Top-1 prediction is actually correct.

**Script**: `extension_6_1/stage_1_extract_features_no_inliers.py` (Note: This script needs to be created or use the temporary extraction scripts)

**Inputs**:
- **Prediction text files (`.txt`)** in `--preds-dir`: One file per query containing retrieval results (indices/IDs of top-K retrieved database images)
- **`z_data` file (`--z-data-path`)**: PyTorch file storing:
  - `database_utms`: poses of database images
  - `predictions`: indices of retrieved neighbors
  - `distances`: descriptor distances between query and database images

**What is Computed Per Query**:

- **Label (`labels[itr]`)**: 
  - Use `get_list_distances_from_preds(txt_file_query)` to obtain geographic distances from predictions to ground truth
  - If first distance `geo_dists[0]` ≤ `positive-dist-threshold` (default 25 m): `labels[itr] = 1.0` (Top-1 correct)
  - Otherwise: `labels[itr] = 0.0` (Top-1 wrong)

**Features Extracted** (8 total - all available before image matching):
1. `top1_distance` - Descriptor distance of Top-1 retrieved image (`dists[itr][0]`)
2. `peakiness` - Ratio of Top-1 to Top-2 distances (`dists[itr][0] / (dists[itr][1] + 1e-8)`)
3. `sue_score` - Spatial Uncertainty Estimate from top-K neighbors (weighted variance of positions)
4. `topk_distance_spread` - Standard deviation of top-5 distances
5. `top1_top2_similarity` - Distance ratio (Top2/Top1) - measures ambiguity
6. `top1_top3_ratio` - Distance ratio (Top1/Top3) - how much better Top-1 is than Top-3
7. `top2_top3_ratio` - Distance ratio (Top2/Top3) - captures second-tier ambiguity
8. `geographic_clustering` - Average pairwise distance of top-K positions

```bash
python -m extension_6_1.stage_1_extract_features_no_inliers \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --z-data-path logs/log_sf_xs_test/[timestamp]/z_data.torch \
  --output-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --positive-dist-threshold 25
```

**Output**: `data/features_and_predictions/features_sf_xs_test_improved.npz` containing:
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

### Stage 2: Feature File Validation & Preparation

**Goal**: Validate and inspect feature files produced by Stage 1 to ensure data quality. Prepare feature files for different dataset splits (train, validation, test). Understand feature distributions before training.

**Note**: In the current implementation, Stage 2 is **implicitly handled** by:
1. The `load_feature_file()` function in training/application scripts
2. The training script which validates inputs before training
3. Manual inspection when needed

**What Stage 2 Involves**:

#### 1. Feature File Inspection
- **Load and verify** structure of `.npz` files:
  - Check that all required arrays are present: `labels`, `top1_distance`, `peakiness`, `sue_score`, and 5 additional features
  - Verify array shapes are consistent (all should have same length = number of queries)
  - Check data types and value ranges

#### 2. Data Quality Checks
- **Label distribution**: Count queries with `label = 1` (correct Top-1) vs `label = 0` (incorrect Top-1)
  - Check for class imbalance (important for logistic regression training)
- **Feature statistics**: Compute basic statistics (mean, std, min, max) for each feature
  - Check for missing values or invalid entries (NaN, Inf)
  - Identify potential outliers

#### 3. Preparation for Training
- Ensure feature files are ready for Stage 3:
  - Training file: `features_svox_train_improved.npz` (required)
  - Validation file: `features_sf_xs_val_improved.npz` (for model validation and threshold selection)
  - All feature arrays should be properly formatted

**Why Stage 2 is Important**:
- **Catch errors early**: Identify issues in feature extraction before training
- **Understand your data**: Know the characteristics of your training set
- **Debug problems**: If training fails, check if the issue is in the data
- **Inform decisions**: Feature statistics can guide hyperparameter choices

**Usage** (in training scripts):
```python
def load_feature_file(path: str) -> dict:
    """Load feature file and return dictionary."""
    data = np.load(path)
    result = {
        "labels": data["labels"].astype("float32"),
        "top1_distance": data["top1_distance"].astype("float32"),
        "peakiness": data["peakiness"].astype("float32"),
        "sue_score": data["sue_score"].astype("float32"),
        # ... 5 additional features
    }
    return result
```

---

### Stage 3: Train Logistic Regression Model

**Goal**: Use per-query feature files to train a logistic regression model that predicts the probability that Top-1 is correct (i.e., the query is "easy"). This model will later be used to decide whether a query is easy (skip re-ranking) or hard (apply re-ranking).

**Script**: `extension_6_1/stage_3_train_logreg_easy_queries.py`

**Inputs**:
- **Training feature file**: `data/features_and_predictions/features_svox_train_improved.npz`
  - Contains arrays: `labels`, `top1_distance`, `peakiness`, `sue_score`, and 5 additional features
- **Validation feature file**: `data/features_and_predictions/features_sf_xs_val_improved.npz`
  - Used to check model generalization and find optimal threshold

**Model and Features**:
- **Features per query** (input to model): 8 improved features (all available before image matching)
  - `top1_distance`, `peakiness`, `sue_score`
  - `topk_distance_spread`, `top1_top2_similarity`, `top1_top3_ratio`, `top2_top3_ratio`, `geographic_clustering`
- **Label per query**: `labels` (1 = correct/easy, 0 = wrong/hard)
- **Target**: `easy_score = labels` (1 = easy/correct, 0 = hard/wrong)
- **Model**: Logistic Regression classifier
  - Implemented with `sklearn.linear_model.LogisticRegression`
  - `class_weight='balanced'` to handle class imbalance
  - Trained to predict probability that `label = 1` (Top-1 correct/easy)

**Process**:
1. Load training and validation features
2. Build feature matrix (8 features)
3. Scale features using `StandardScaler` (mean=0, std=1)
4. Train Logistic Regression with `class_weight='balanced'`
5. **Find optimal threshold** on validation set:
   - Test thresholds from 0.1 to 0.95 in 0.01 steps
   - Select threshold that maximizes F1-score (default) or targets specific recall rate
   - This learned threshold prevents hard thresholding
6. Save model bundle containing:
   - Trained model
   - Scaler (for feature normalization)
   - Optimal threshold
   - Feature names
   - Metadata

```bash
python -m extension_6_1.stage_3_train_logreg_easy_queries \
  --train-features data/features_and_predictions/features_svox_train_improved.npz \
  --val-features data/features_and_predictions/features_sf_xs_val_improved.npz \
  --output-model logreg_easy_queries_optimal.pkl
```

**Output**: `logreg_easy_queries_optimal.pkl` - Serialized model bundle containing:
- `model`: Trained Logistic Regression model
- `scaler`: StandardScaler for feature normalization
- `optimal_threshold`: Learned threshold (e.g., 0.410)
- `feature_names`: List of feature names used
- `threshold_method`: Method used to find threshold (e.g., "f1")
- `target_type`: Target variable type (e.g., "easy_score")

This file will be loaded in Stage 4 to compute probabilities for new queries and define the adaptive re-ranking rule.

---

### Stage 4: Apply Model and Define Adaptive Re-ranking Rule

**Goal**: Use the trained logistic regression model to compute, for each query, the probability that Top-1 is correct. Define the adaptive re-ranking rule using the optimal threshold (learned from validation, not hard-coded) to classify queries as easy (skip re-ranking) or hard (apply re-ranking).

**Script**: `extension_6_1/stage_4_apply_logreg_easy_queries.py`

**Inputs**:
- **Trained model file**: `logreg_easy_queries_optimal.pkl` (from Stage 3)
- **Feature file**: `data/features_and_predictions/features_sf_xs_test_improved.npz` (for test split)

**Process**:
1. Load model bundle (model, scaler, optimal threshold, feature names)
2. Load test features
3. Build feature matrix using expected feature names from model
4. Scale features using saved scaler
5. Predict probabilities: `probs = model.predict_proba(X_scaled)[:, 1]` (probability of being easy)
6. **Apply optimal threshold** (learned, not hard-coded):
   - If `probs >= optimal_threshold` → **easy query** → **skip re-ranking**
   - If `probs < optimal_threshold` → **hard query** → **apply re-ranking**
7. Save outputs

```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --output-path logreg_easy_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_test.txt
```

**Outputs**:
- **`.npz` file** (`logreg_easy_test.npz`) containing:
  - `probs` – predicted probability that Top-1 is correct (probability of being easy) per query
  - `is_easy` – boolean array: `True` if query is easy (skip re-ranking)
  - `is_hard` – boolean array: `True` if query is hard (apply re-ranking)
  - `labels` – ground-truth Top-1 correctness labels (0/1)
- **Hard queries list** (`data/features_and_predictions/hard_queries_test.txt`):
  - Text file with one query index per line (indices of hard queries)
  - Used by adaptive image matching script to process only hard queries

**Key Difference from Original Design**:
- **No hard thresholding**: Uses optimal threshold learned from validation set (e.g., 0.410)
- **Not fixed at 0.5**: Threshold is data-driven and optimal for the validation set
- **Predicts "easy" directly**: More natural than predicting "hard"

---

### Stage 5: Adaptive Image Matching

**Goal**: Run image matching **only** for hard queries.

**Script**: `match_queries_preds_adaptive.py`

```bash
python match_queries_preds_adaptive.py \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --hard-queries-list data/features_and_predictions/hard_queries_test.txt \
  --out-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20
```

**Process**:
- Load hard query indices from file
- For each hard query: Run SuperPoint + LightGlue matching
- For each easy query: Create empty .torch file (for compatibility)
- Save inlier counts to .torch files

**Output**: `preds_superpoint-lg_adaptive/` - .torch files with inlier counts

---

### Stage 6: Evaluate Adaptive Re-ranking

**Goal**: Use the easy/hard decisions from the logistic regression model to evaluate an adaptive pipeline. For easy queries, use retrieval-only ranking. For hard queries, use re-ranked ranking (based on inliers). Compute Recall@N for this adaptive strategy.

**Script**: `extension_6_1/stage_5_adaptive_reranking_eval.py`

**Inputs**:
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

**Behavior**:
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

```bash
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --inliers-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --logreg-output logreg_easy_test.npz \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20
```

**Output**: Prints Recall@N metrics (e.g., Recall@1, R@5, R@10, R@20)

**Note on Recall@100**: The original `reranking.py` script uses recall values `[1, 5, 10, 20, 100]`. Including Recall@100 makes the adaptive method directly comparable to existing baselines. Recall@100 shows an upper bound when allowing many candidates and checks that the adaptive strategy does not degrade performance in the long tail compared to always re-ranking.

---

## Complete Pipeline Example

### For SF-XS Test Dataset:

```bash
# 1. VPR Evaluation
python VPR-methods-evaluation/main.py \
  --log_dir log_sf_xs_test \
  --database_folder data/sf_xs/test/gallery \
  --queries_folder data/sf_xs/test/queries \
  --device cuda

# 2. Extract Features
python -m extension_6_1.stage_1_extract_features_no_inliers \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --z-data-path logs/log_sf_xs_test/[timestamp]/z_data.torch \
  --output-path data/features_and_predictions/features_sf_xs_test_improved.npz

# 3. Apply Model (assuming model already trained)
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_test.txt

# 4. Adaptive Image Matching
python match_queries_preds_adaptive.py \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --hard-queries-list data/features_and_predictions/hard_queries_test.txt \
  --out-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --matcher superpoint-lg --device cuda

# 5. Evaluate
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --inliers-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --logreg-output logreg_easy_test.npz \
  --recall-values 1 5 10 20
```

---

## Comparison with Baseline Methods

### Baseline (Retrieval-only)
- **No image matching**: Use VPR retrieval results directly
- **Fast**: No computation overhead
- **Lower accuracy**: R@1 = 63.1% (SF-XS test)
- **% queries re-ranked**: 0%

### Full Re-ranking
- **Image matching for all queries**: Run SuperPoint + LightGlue on all queries
- **Slow**: ~9.5 seconds per query
- **Higher accuracy**: R@1 = 77.4% (SF-XS test)
- **% queries re-ranked**: 100%

### Adaptive Re-ranking
- **Image matching for hard queries only**: Predict hard queries using logistic regression, skip easy ones
- **Moderate speed**: ~2.4 seconds per query (average)
- **Good accuracy**: R@1 = 69.8% (SF-XS test)
- **Time savings**: 74.6% (only 25.4% queries re-ranked)
- **% queries re-ranked**: 25.4% (varies by dataset and model)

## What to Compare in Your Report

For each dataset (SF-XS test, Tokyo-XS test, SVOX test), you should compare:

- **Retrieval-only** (CosPlace, no re-ranking): R@1, R@5, R@10, R@20
- **Always re-ranking** (e.g., `reranking.py` over all queries): R@1, R@5, R@10, R@20
- **Adaptive re-ranking** (our method): R@1, R@5, R@10, R@20, and % of queries where `is_hard` is True (fraction of queries that were re-ranked)

You can summarize this in a table:

| Dataset      | Method              | R@1  | R@5  | R@10 | % queries re-ranked |
|-------------|---------------------|------|------|------|----------------------|
| SF-XS test  | Retrieval-only      | 63.1%| 74.8%| 78.6%| 0%                   |
| SF-XS test  | Always re-ranking   | 77.4%| 80.3%| 80.9%| 100%                 |
| SF-XS test  | Adaptive (logreg)   | 69.8%| 77.7%| 79.5%| 25.4%                |
| Tokyo-XS    | ...                 | ...  | ...  | ...  | ...                  |
| SVOX test   | ...                 | ...  | ...  | ...  | ...                  |

In the discussion, highlight whether adaptive re-ranking:
- Achieves **similar R@1** to always re-ranking, and
- Reduces the **fraction of queries** for which expensive re-ranking (SuperPoint+LightGlue) is run.

---

## Additional Resources

- **[Stage Details](STAGE_DETAILS.md)**: Comprehensive reference for each pipeline stage
- **[Methodology](METHODOLOGY.md)**: Details on different approaches and strategies
- **[Technical Details](TECHNICAL_DETAILS.md)**: Implementation details and fixes

