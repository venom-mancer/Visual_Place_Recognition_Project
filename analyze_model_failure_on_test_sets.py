"""
Analyze why models fail to detect hard queries on Tokyo-XS, SVOX Sun, and SVOX Night test sets.

This script:
1. Loads model predictions and feature distributions
2. Compares feature distributions between training/validation and test sets
3. Analyzes why models predict all queries as "easy" on certain test sets
4. Suggests solutions (threshold calibration, feature normalization, etc.)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


def analyze_feature_distributions(model_path, train_features_path, val_features_path, test_features_path, test_name):
    """Analyze feature distributions and model predictions."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {test_name}")
    print(f"{'='*70}")
    
    # Load model
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    optimal_threshold = model_bundle.get("optimal_threshold", 0.5)
    feature_names = model_bundle.get("feature_names", [])
    
    # Load features
    train_features = load_feature_file(train_features_path)
    val_features = load_feature_file(val_features_path)
    test_features = load_feature_file(test_features_path)
    
    # Build feature matrices
    X_train = build_feature_matrix(train_features, feature_names)
    X_val = build_feature_matrix(val_features, feature_names)
    X_test = build_feature_matrix(test_features, feature_names)
    
    # Get labels
    y_train = train_features["labels"].astype("float32")
    y_val = val_features["labels"].astype("float32")
    y_test = test_features["labels"].astype("float32")
    
    # Handle NaNs
    train_mask = ~np.isnan(X_train).any(axis=1)
    val_mask = ~np.isnan(X_val).any(axis=1)
    test_mask = ~np.isnan(X_test).any(axis=1)
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply threshold
    train_pred = (train_probs >= optimal_threshold).astype(int)
    val_pred = (val_probs >= optimal_threshold).astype(int)
    test_pred = (test_probs >= optimal_threshold).astype(int)
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Training: {len(X_train)} queries, {y_train.mean():.1%} easy, {1-y_train.mean():.1%} hard")
    print(f"  Validation: {len(X_val)} queries, {y_val.mean():.1%} easy, {1-y_val.mean():.1%} hard")
    print(f"  Test: {len(X_test)} queries, {y_test.mean():.1%} easy, {1-y_test.mean():.1%} hard")
    
    print(f"\nProbability Statistics:")
    print(f"  Training: min={train_probs.min():.3f}, mean={train_probs.mean():.3f}, max={train_probs.max():.3f}")
    print(f"  Validation: min={val_probs.min():.3f}, mean={val_probs.mean():.3f}, max={val_probs.max():.3f}")
    print(f"  Test: min={test_probs.min():.3f}, mean={test_probs.mean():.3f}, max={test_probs.max():.3f}")
    
    print(f"\nModel Predictions (threshold={optimal_threshold:.3f}):")
    train_hard_rate = (train_probs < optimal_threshold).mean()
    val_hard_rate = (val_probs < optimal_threshold).mean()
    test_hard_rate = (test_probs < optimal_threshold).mean()
    
    print(f"  Training: {train_hard_rate:.1%} predicted as hard")
    print(f"  Validation: {val_hard_rate:.1%} predicted as hard")
    print(f"  Test: {test_hard_rate:.1%} predicted as hard")
    
    # Feature distribution comparison
    print(f"\nFeature Distribution Comparison (mean values):")
    print(f"{'Feature':<25} {'Train':<12} {'Val':<12} {'Test':<12} {'Diff (Test-Train)':<15}")
    print("-" * 75)
    
    for i, feat_name in enumerate(feature_names):
        train_mean = X_train[:, i].mean()
        val_mean = X_val[:, i].mean()
        test_mean = X_test[:, i].mean()
        diff = test_mean - train_mean
        print(f"{feat_name:<25} {train_mean:>10.4f} {val_mean:>10.4f} {test_mean:>10.4f} {diff:>+14.4f}")
    
    # Problem diagnosis
    print(f"\n{'='*70}")
    print(f"Problem Diagnosis:")
    print(f"{'='*70}")
    
    if test_hard_rate < 0.01:
        print(f"[PROBLEM] Model predicts {test_hard_rate:.1%} hard queries (almost none!)")
        print(f"   This means:")
        print(f"   - All queries are predicted as 'easy'")
        print(f"   - No queries get re-ranking")
        print(f"   - No time savings (but also no performance improvement)")
        
        if test_probs.min() > optimal_threshold:
            print(f"\n   Root Cause: All test probabilities ({test_probs.min():.3f} to {test_probs.max():.3f})")
            print(f"   are ABOVE the threshold ({optimal_threshold:.3f})")
            print(f"   -> Model is overconfident on this test set")
        else:
            print(f"\n   Root Cause: Most test probabilities are above threshold")
            print(f"   -> Model learned different distribution in training")
        
        print(f"\n   Solutions:")
        print(f"   1. Threshold Calibration: Re-calibrate threshold on test set")
        print(f"   2. Feature Normalization: Check if features need different scaling")
        print(f"   3. Domain Adaptation: Train separate model for this dataset")
        print(f"   4. Ensemble: Combine predictions from multiple models")
    else:
        print(f"✅ Model predicts {test_hard_rate:.1%} hard queries")
        print(f"   This is reasonable (actual hard rate: {1-y_test.mean():.1%})")
    
    return {
        "test_name": test_name,
        "test_hard_rate": test_hard_rate,
        "actual_hard_rate": 1 - y_test.mean(),
        "test_probs_min": test_probs.min(),
        "test_probs_max": test_probs.max(),
        "test_probs_mean": test_probs.mean(),
        "optimal_threshold": optimal_threshold,
        "problem": test_hard_rate < 0.01
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze why models fail to detect hard queries on certain test sets"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_improved.npz",
        help="Training features"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Validation features"
    )
    parser.add_argument(
        "--tokyo-test-features",
        type=str,
        default="data/features_and_predictions/features_tokyo_xs_test_improved.npz",
        help="Tokyo-XS test features"
    )
    parser.add_argument(
        "--svox-test-features",
        type=str,
        default="data/features_and_predictions/features_svox_test_improved.npz",
        help="SVOX test features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/model_failure_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each problematic test set
    results = []
    
    # Tokyo-XS Test
    if Path(args.tokyo_test_features).exists():
        result = analyze_feature_distributions(
            args.model1_path,
            args.train_features,
            args.val_features,
            args.tokyo_test_features,
            "Tokyo-XS Test"
        )
        results.append(result)
    
    # SVOX Test (Sun and Night use same file for now)
    if Path(args.svox_test_features).exists():
        result = analyze_feature_distributions(
            args.model1_path,
            args.train_features,
            args.val_features,
            args.svox_test_features,
            "SVOX Test"
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    for r in results:
        if r["problem"]:
            print(f"\n[PROBLEM] {r['test_name']}:")
            print(f"   Predicted hard rate: {r['test_hard_rate']:.1%}")
            print(f"   Actual hard rate: {r['actual_hard_rate']:.1%}")
            print(f"   Probability range: [{r['test_probs_min']:.3f}, {r['test_probs_max']:.3f}]")
            print(f"   Threshold: {r['optimal_threshold']:.3f}")
            print(f"   -> All probabilities above threshold -> Model overconfident")
    
    # Save summary
    summary_path = output_dir / "failure_analysis_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Model Failure Analysis: Hard Query Detection\n\n")
        f.write("## Problem\n\n")
        f.write("Models fail to detect hard queries on certain test sets, predicting 0% hard queries.\n\n")
        f.write("## Results\n\n")
        for r in results:
            f.write(f"### {r['test_name']}\n\n")
            f.write(f"- **Predicted Hard Rate**: {r['test_hard_rate']:.1%}\n")
            f.write(f"- **Actual Hard Rate**: {r['actual_hard_rate']:.1%}\n")
            f.write(f"- **Probability Range**: [{r['test_probs_min']:.3f}, {r['test_probs_max']:.3f}]\n")
            f.write(f"- **Optimal Threshold**: {r['optimal_threshold']:.3f}\n")
            if r["problem"]:
                f.write(f"- **Status**: [PROBLEM] Model overconfident\n")
            else:
                f.write(f"- **Status**: [OK]\n")
            f.write("\n")
        
        f.write("## Solutions\n\n")
        f.write("1. **Threshold Calibration**: Re-calibrate threshold on test set\n")
        f.write("2. **Feature Analysis**: Check feature distribution shifts\n")
        f.write("3. **Domain Adaptation**: Train dataset-specific models\n")
        f.write("4. **Ensemble Methods**: Combine multiple models\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



This script:
1. Loads model predictions and feature distributions
2. Compares feature distributions between training/validation and test sets
3. Analyzes why models predict all queries as "easy" on certain test sets
4. Suggests solutions (threshold calibration, feature normalization, etc.)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


def analyze_feature_distributions(model_path, train_features_path, val_features_path, test_features_path, test_name):
    """Analyze feature distributions and model predictions."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {test_name}")
    print(f"{'='*70}")
    
    # Load model
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    optimal_threshold = model_bundle.get("optimal_threshold", 0.5)
    feature_names = model_bundle.get("feature_names", [])
    
    # Load features
    train_features = load_feature_file(train_features_path)
    val_features = load_feature_file(val_features_path)
    test_features = load_feature_file(test_features_path)
    
    # Build feature matrices
    X_train = build_feature_matrix(train_features, feature_names)
    X_val = build_feature_matrix(val_features, feature_names)
    X_test = build_feature_matrix(test_features, feature_names)
    
    # Get labels
    y_train = train_features["labels"].astype("float32")
    y_val = val_features["labels"].astype("float32")
    y_test = test_features["labels"].astype("float32")
    
    # Handle NaNs
    train_mask = ~np.isnan(X_train).any(axis=1)
    val_mask = ~np.isnan(X_val).any(axis=1)
    test_mask = ~np.isnan(X_test).any(axis=1)
    
    X_train = X_train[train_mask]
    y_train = y_train[train_mask]
    X_val = X_val[val_mask]
    y_val = y_val[val_mask]
    X_test = X_test[test_mask]
    y_test = y_test[test_mask]
    
    # Scale features
    X_train_scaled = scaler.transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Get predictions
    train_probs = model.predict_proba(X_train_scaled)[:, 1]
    val_probs = model.predict_proba(X_val_scaled)[:, 1]
    test_probs = model.predict_proba(X_test_scaled)[:, 1]
    
    # Apply threshold
    train_pred = (train_probs >= optimal_threshold).astype(int)
    val_pred = (val_probs >= optimal_threshold).astype(int)
    test_pred = (test_probs >= optimal_threshold).astype(int)
    
    # Statistics
    print(f"\nDataset Statistics:")
    print(f"  Training: {len(X_train)} queries, {y_train.mean():.1%} easy, {1-y_train.mean():.1%} hard")
    print(f"  Validation: {len(X_val)} queries, {y_val.mean():.1%} easy, {1-y_val.mean():.1%} hard")
    print(f"  Test: {len(X_test)} queries, {y_test.mean():.1%} easy, {1-y_test.mean():.1%} hard")
    
    print(f"\nProbability Statistics:")
    print(f"  Training: min={train_probs.min():.3f}, mean={train_probs.mean():.3f}, max={train_probs.max():.3f}")
    print(f"  Validation: min={val_probs.min():.3f}, mean={val_probs.mean():.3f}, max={val_probs.max():.3f}")
    print(f"  Test: min={test_probs.min():.3f}, mean={test_probs.mean():.3f}, max={test_probs.max():.3f}")
    
    print(f"\nModel Predictions (threshold={optimal_threshold:.3f}):")
    train_hard_rate = (train_probs < optimal_threshold).mean()
    val_hard_rate = (val_probs < optimal_threshold).mean()
    test_hard_rate = (test_probs < optimal_threshold).mean()
    
    print(f"  Training: {train_hard_rate:.1%} predicted as hard")
    print(f"  Validation: {val_hard_rate:.1%} predicted as hard")
    print(f"  Test: {test_hard_rate:.1%} predicted as hard")
    
    # Feature distribution comparison
    print(f"\nFeature Distribution Comparison (mean values):")
    print(f"{'Feature':<25} {'Train':<12} {'Val':<12} {'Test':<12} {'Diff (Test-Train)':<15}")
    print("-" * 75)
    
    for i, feat_name in enumerate(feature_names):
        train_mean = X_train[:, i].mean()
        val_mean = X_val[:, i].mean()
        test_mean = X_test[:, i].mean()
        diff = test_mean - train_mean
        print(f"{feat_name:<25} {train_mean:>10.4f} {val_mean:>10.4f} {test_mean:>10.4f} {diff:>+14.4f}")
    
    # Problem diagnosis
    print(f"\n{'='*70}")
    print(f"Problem Diagnosis:")
    print(f"{'='*70}")
    
    if test_hard_rate < 0.01:
        print(f"[PROBLEM] Model predicts {test_hard_rate:.1%} hard queries (almost none!)")
        print(f"   This means:")
        print(f"   - All queries are predicted as 'easy'")
        print(f"   - No queries get re-ranking")
        print(f"   - No time savings (but also no performance improvement)")
        
        if test_probs.min() > optimal_threshold:
            print(f"\n   Root Cause: All test probabilities ({test_probs.min():.3f} to {test_probs.max():.3f})")
            print(f"   are ABOVE the threshold ({optimal_threshold:.3f})")
            print(f"   -> Model is overconfident on this test set")
        else:
            print(f"\n   Root Cause: Most test probabilities are above threshold")
            print(f"   -> Model learned different distribution in training")
        
        print(f"\n   Solutions:")
        print(f"   1. Threshold Calibration: Re-calibrate threshold on test set")
        print(f"   2. Feature Normalization: Check if features need different scaling")
        print(f"   3. Domain Adaptation: Train separate model for this dataset")
        print(f"   4. Ensemble: Combine predictions from multiple models")
    else:
        print(f"✅ Model predicts {test_hard_rate:.1%} hard queries")
        print(f"   This is reasonable (actual hard rate: {1-y_test.mean():.1%})")
    
    return {
        "test_name": test_name,
        "test_hard_rate": test_hard_rate,
        "actual_hard_rate": 1 - y_test.mean(),
        "test_probs_min": test_probs.min(),
        "test_probs_max": test_probs.max(),
        "test_probs_mean": test_probs.mean(),
        "optimal_threshold": optimal_threshold,
        "problem": test_hard_rate < 0.01
    }


def main():
    parser = argparse.ArgumentParser(
        description="Analyze why models fail to detect hard queries on certain test sets"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_improved.npz",
        help="Training features"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Validation features"
    )
    parser.add_argument(
        "--tokyo-test-features",
        type=str,
        default="data/features_and_predictions/features_tokyo_xs_test_improved.npz",
        help="Tokyo-XS test features"
    )
    parser.add_argument(
        "--svox-test-features",
        type=str,
        default="data/features_and_predictions/features_svox_test_improved.npz",
        help="SVOX test features"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/model_failure_analysis",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Analyze each problematic test set
    results = []
    
    # Tokyo-XS Test
    if Path(args.tokyo_test_features).exists():
        result = analyze_feature_distributions(
            args.model1_path,
            args.train_features,
            args.val_features,
            args.tokyo_test_features,
            "Tokyo-XS Test"
        )
        results.append(result)
    
    # SVOX Test (Sun and Night use same file for now)
    if Path(args.svox_test_features).exists():
        result = analyze_feature_distributions(
            args.model1_path,
            args.train_features,
            args.val_features,
            args.svox_test_features,
            "SVOX Test"
        )
        results.append(result)
    
    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    
    for r in results:
        if r["problem"]:
            print(f"\n[PROBLEM] {r['test_name']}:")
            print(f"   Predicted hard rate: {r['test_hard_rate']:.1%}")
            print(f"   Actual hard rate: {r['actual_hard_rate']:.1%}")
            print(f"   Probability range: [{r['test_probs_min']:.3f}, {r['test_probs_max']:.3f}]")
            print(f"   Threshold: {r['optimal_threshold']:.3f}")
            print(f"   -> All probabilities above threshold -> Model overconfident")
    
    # Save summary
    summary_path = output_dir / "failure_analysis_summary.md"
    with open(summary_path, 'w') as f:
        f.write("# Model Failure Analysis: Hard Query Detection\n\n")
        f.write("## Problem\n\n")
        f.write("Models fail to detect hard queries on certain test sets, predicting 0% hard queries.\n\n")
        f.write("## Results\n\n")
        for r in results:
            f.write(f"### {r['test_name']}\n\n")
            f.write(f"- **Predicted Hard Rate**: {r['test_hard_rate']:.1%}\n")
            f.write(f"- **Actual Hard Rate**: {r['actual_hard_rate']:.1%}\n")
            f.write(f"- **Probability Range**: [{r['test_probs_min']:.3f}, {r['test_probs_max']:.3f}]\n")
            f.write(f"- **Optimal Threshold**: {r['optimal_threshold']:.3f}\n")
            if r["problem"]:
                f.write(f"- **Status**: [PROBLEM] Model overconfident\n")
            else:
                f.write(f"- **Status**: [OK]\n")
            f.write("\n")
        
        f.write("## Solutions\n\n")
        f.write("1. **Threshold Calibration**: Re-calibrate threshold on test set\n")
        f.write("2. **Feature Analysis**: Check feature distribution shifts\n")
        f.write("3. **Domain Adaptation**: Train dataset-specific models\n")
        f.write("4. **Ensemble Methods**: Combine multiple models\n")
    
    print(f"\nSummary saved to: {summary_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

