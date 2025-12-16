# Extension 6.1 - Adaptive Re-ranking with Logistic Regression

This folder contains all code and documentation for Extension 6.1 implementation.

## File Organization

### Code Files (with Stage Numbers)
- `stage_1_extract_features.py` - Stage 1: Extract per-query features and labels
- `stage_2_feature_io.py` - Stage 2: Helper functions for loading/validating feature files
- `stage_3_train_logreg.py` - Stage 3: Train logistic regression model
- `stage_4_apply_logreg.py` - Stage 4: Apply logistic regression to get easy/hard decisions
- `stage_5_adaptive_reranking_eval.py` - Stage 5: Evaluate adaptive re-ranking pipeline
- `baselines.py` - Helper: Uncertainty baseline implementations

### Documentation Files
- `docs_extension_6_1_overview.md` - Overall concept summary and pipeline explanation
- `stage_1.md` - Stage 1 documentation
- `stage_2.md` - Stage 2 documentation
- `stage_3.md` - Stage 3 documentation
- `stage_4.md` - Stage 4 documentation
- `stage_5.md` - Stage 5 documentation
- `stage_6.md` - Stage 6 documentation (experiments and reporting)
- `execution.md` - Step-by-step execution plan with all commands

## Usage

All scripts should be run from the project root directory:

```bash
# Stage 1: Extract features
python -m extension_6_1.stage_1_extract_features --preds-dir ... --inliers-dir ... --z-data-path ... --output-path features.npz

# Stage 3: Train logistic regression
python -m extension_6_1.stage_3_train_logreg --train-features features_svox_train.npz --output-model logreg_model.pkl

# Stage 4: Apply logistic regression
python -m extension_6_1.stage_4_apply_logreg --model-path logreg_model.pkl --feature-path features_test.npz --output-path outputs.npz

# Stage 5: Evaluate adaptive re-ranking
python -m extension_6_1.stage_5_adaptive_reranking_eval --preds-dir ... --inliers-dir ... --logreg-output outputs.npz
```

See `execution.md` for the complete execution plan.

