## Extension 6.1 – Stage 3: Train Logistic Regression on Per-query Features

### Goal of Stage 3
- Use the per-query feature files from Stage 1/2 (e.g. `features_svox_train.npz`) to **train a logistic regression model** that predicts:
  - \( p(\text{Top‑1 is correct} \mid \text{features}) \)
- This model will later be used to decide whether a query is **easy** (skip re-ranking) or **hard** (apply re-ranking).

### Inputs to Stage 3
- **Training feature file** (recommended): `features_svox_train.npz`
  - Contains arrays:
    - `labels` – 1 if Top‑1 within threshold, else 0.
    - `num_inliers`
    - `top1_distance`
    - `peakiness`
    - `sue_score` (currently placeholder).
- **Validation feature file**: `features_sf_xs_val.npz`
  - Used to:
    - Check that the model generalizes.
    - Optionally tune hyperparameters (e.g., regularization strength `C`).

### Model and features
- **Features per query** (input to the model):
  - `num_inliers`
  - `top1_distance`
  - `peakiness`
  - `sue_score`
- **Label per query**:
  - `labels` (0 or 1).
- **Model**:
  - A simple **logistic regression classifier**:
    - Implemented with `sklearn.linear_model.LogisticRegression`.
    - Trained to predict the probability that `label = 1` (Top‑1 correct).

### Output of Stage 3
- A serialized model file, e.g.:
  - `logreg_svox_train.pkl`
- This file will be loaded in later stages to:
  - Compute probabilities for new queries.
  - Define the adaptive re-ranking rule (Stage 4).


