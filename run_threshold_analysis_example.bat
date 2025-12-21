@echo off
REM Example script to run threshold analysis for all test datasets (Windows)

REM This script demonstrates how to run comprehensive threshold analysis
REM Make sure you have:
REM 1. Trained model: logreg_easy_queries_optimal.pkl
REM 2. Feature files for all test datasets
REM 3. Prediction directories with .txt files
REM 4. Inliers directories with .torch files (from full re-ranking)

python analyze_threshold_impact.py ^
  --model-path logreg_easy_queries_optimal.pkl ^
  --datasets sf_xs_test tokyo_xs_test svox_test ^
  --feature-paths ^
    data/features_and_predictions/features_sf_xs_test_improved.npz ^
    data/features_and_predictions/features_tokyo_xs_test_improved.npz ^
    data/features_and_predictions/features_svox_test_improved.npz ^
  --preds-dirs ^
    logs/log_sf_xs_test/2025-12-17_21-14-10/preds ^
    log_tokyo_xs_test/2025-12-18_14-43-02/preds ^
    log_svox_test/2025-12-18_16-01-59/preds ^
  --inliers-dirs ^
    logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg ^
    log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg ^
    log_svox_test/2025-12-18_16-01-59/preds_superpoint-lg ^
  --output-dir output_stages/threshold_analysis ^
  --threshold-range 0.1 0.95 ^
  --threshold-step 0.02 ^
  --num-preds 20 ^
  --positive-dist-threshold 25

echo.
echo Analysis complete! Check output_stages/threshold_analysis/ for plots and summary.

