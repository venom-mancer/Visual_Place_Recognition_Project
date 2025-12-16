## Extension 6.1 – Stage 2: Feature File Validation & Preparation

### Goal of Stage 2
- **Validate and inspect** the feature files produced by Stage 1 to ensure data quality.
- **Prepare feature files** for different dataset splits (train, validation, test).
- **Understand feature distributions** before training the logistic regression model.

### Inputs to Stage 2
- **Feature files (`.npz`)** produced by Stage 1:
  - `features_svox_train.npz` – Training features (from SVOX train split)
  - `features_sf_xs_val.npz` – Validation features (from SF-XS validation split, optional)
  - Test feature files (e.g., `features_sf_xs_test.npz`, `features_tokyo_xs_test.npz`, `features_svox_test.npz`)

### What Stage 2 Involves

#### 1. Feature File Inspection
- **Load and verify** the structure of `.npz` files:
  - Check that all required arrays are present: `labels`, `num_inliers`, `top1_distance`, `peakiness`, `sue_score`
  - Verify array shapes are consistent (all should have the same length = number of queries)
  - Check data types and value ranges

#### 2. Data Quality Checks
- **Label distribution**:
  - Count how many queries have `label = 1` (correct Top-1) vs `label = 0` (incorrect Top-1)
  - Check for class imbalance (important for logistic regression training)
- **Feature statistics**:
  - Compute basic statistics (mean, std, min, max) for each feature
  - Check for missing values or invalid entries (e.g., NaN, Inf)
  - Identify potential outliers

#### 3. Feature Exploration (Optional)
- **Visualize feature distributions**:
  - Histograms or box plots for each feature
  - Compare feature distributions between correct (`label=1`) and incorrect (`label=0`) queries
  - This helps understand which features are most discriminative

#### 4. Preparation for Training
- **Ensure feature files are ready** for Stage 3:
  - Training file: `features_svox_train.npz` (required)
  - Validation file: `features_sf_xs_val.npz` (optional, for model validation)
  - All feature arrays should be properly formatted and normalized if needed

### Tools for Stage 2

The `vpr_uncertainty/feature_io.py` module provides helper functions:

```python
from vpr_uncertainty.feature_io import load_feature_file, describe_feature_file

# Load a feature file
data = load_feature_file("features_svox_train.npz")

# Print summary statistics
describe_feature_file("features_svox_train.npz")
```

### Output of Stage 2
- **Validated feature files** ready for training
- **Summary statistics** about the dataset (optional report)
- **Confidence** that the data is clean and ready for Stage 3

### Why Stage 2 is Important
- **Catch errors early**: Identify issues in feature extraction before training
- **Understand your data**: Know the characteristics of your training set
- **Debug problems**: If training fails, you can check if the issue is in the data
- **Inform decisions**: Feature statistics can guide hyperparameter choices (e.g., regularization strength)

### Note on Implementation
In our implementation, Stage 2 is **implicitly handled** by:
1. The `feature_io.py` helper functions that can load and inspect feature files
2. The training script (`train_logreg.py`) which validates inputs before training
3. Manual inspection when needed using Python scripts or notebooks

For a more formal Stage 2, you could create a dedicated validation script that:
- Loads all feature files
- Performs comprehensive checks
- Generates a validation report
- Flags any issues before proceeding to training

