"""
Compare model performance with and without inliers feature.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_3_train_logreg_easy_queries import load_feature_file, build_feature_matrix


def load_model_results(model_path: Path, val_features_path: Path):
    """Load model and evaluate on validation set."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    optimal_threshold = bundle.get("optimal_threshold", 0.5)
    feature_names = bundle["feature_names"]
    
    # Load validation features
    val_features = load_feature_file(str(val_features_path))
    X_val, y_val, _ = build_feature_matrix(val_features)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[valid_mask]
    y_val = y_val[valid_mask]
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    predictions = (probs >= optimal_threshold).astype(int)
    
    # Compute metrics
    accuracy = (predictions == y_val).mean()
    hard_query_rate = (1 - predictions.mean()) * 100
    actual_hard_rate = (1 - y_val.mean()) * 100
    
    return {
        "accuracy": accuracy,
        "hard_query_rate": hard_query_rate,
        "actual_hard_rate": actual_hard_rate,
        "optimal_threshold": optimal_threshold,
        "num_features": len(feature_names),
        "feature_names": feature_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare models with and without inliers"
    )
    parser.add_argument(
        "--model-without-inliers",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model trained without inliers (8 features)"
    )
    parser.add_argument(
        "--model-with-inliers",
        type=str,
        default="models/logreg_easy_with_inliers.pkl",
        help="Model trained with inliers (9 features)"
    )
    parser.add_argument(
        "--val-features-without-inliers",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Validation features without inliers"
    )
    parser.add_argument(
        "--val-features-with-inliers",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_with_inliers.npz",
        help="Validation features with inliers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/inliers_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Comparing Models: With vs Without Inliers")
    print(f"{'='*70}\n")
    
    # Load results
    print("Loading model without inliers (8 features)...")
    results_without = load_model_results(
        Path(args.model_without_inliers),
        Path(args.val_features_without_inliers)
    )
    
    print("Loading model with inliers (9 features)...")
    results_with = load_model_results(
        Path(args.model_with_inliers),
        Path(args.val_features_with_inliers)
    )
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"Comparison Results")
    print(f"{'='*70}\n")
    
    print(f"Model WITHOUT Inliers (8 features):")
    print(f"  Accuracy: {results_without['accuracy']*100:.2f}%")
    print(f"  Hard Query Rate: {results_without['hard_query_rate']:.2f}% (actual: {results_without['actual_hard_rate']:.2f}%)")
    print(f"  Optimal Threshold: {results_without['optimal_threshold']:.3f}")
    print(f"  Features: {', '.join(results_without['feature_names'])}")
    
    print(f"\nModel WITH Inliers (9 features):")
    print(f"  Accuracy: {results_with['accuracy']*100:.2f}%")
    print(f"  Hard Query Rate: {results_with['hard_query_rate']:.2f}% (actual: {results_with['actual_hard_rate']:.2f}%)")
    print(f"  Optimal Threshold: {results_with['optimal_threshold']:.3f}")
    print(f"  Features: {', '.join(results_with['feature_names'])}")
    
    print(f"\n{'='*70}")
    print(f"Improvement Analysis")
    print(f"{'='*70}\n")
    
    accuracy_improvement = (results_with['accuracy'] - results_without['accuracy']) * 100
    hard_rate_diff = results_with['hard_query_rate'] - results_without['hard_query_rate']
    
    print(f"Accuracy: {results_without['accuracy']*100:.2f}% -> {results_with['accuracy']*100:.2f}% "
          f"({accuracy_improvement:+.2f}%)")
    print(f"Hard Query Rate: {results_without['hard_query_rate']:.2f}% -> {results_with['hard_query_rate']:.2f}% "
          f"({hard_rate_diff:+.2f}%)")
    
    if accuracy_improvement > 0:
        print(f"\n[SUCCESS] Adding inliers improved accuracy by {accuracy_improvement:.2f}%")
    elif accuracy_improvement < 0:
        print(f"\n[WARNING] Adding inliers decreased accuracy by {abs(accuracy_improvement):.2f}%")
    else:
        print(f"\n[NEUTRAL] Adding inliers did not change accuracy")
    
    # Save comparison report
    report_path = output_dir / "inliers_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("# Model Comparison: With vs Without Inliers Feature\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Metric | Without Inliers (8 features) | With Inliers (9 features) | Improvement |\n")
        f.write("|--------|------------------------------|--------------------------|-------------|\n")
        f.write(f"| Accuracy | {results_without['accuracy']*100:.2f}% | {results_with['accuracy']*100:.2f}% | "
               f"{accuracy_improvement:+.2f}% |\n")
        f.write(f"| Hard Query Rate | {results_without['hard_query_rate']:.2f}% | {results_with['hard_query_rate']:.2f}% | "
               f"{hard_rate_diff:+.2f}% |\n")
        f.write(f"| Optimal Threshold | {results_without['optimal_threshold']:.3f} | {results_with['optimal_threshold']:.3f} | "
               f"{results_with['optimal_threshold'] - results_without['optimal_threshold']:+.3f} |\n")
        
        f.write(f"\n## Feature Comparison\n\n")
        f.write(f"### Without Inliers (8 features):\n")
        for feat in results_without['feature_names']:
            f.write(f"- {feat}\n")
        
        f.write(f"\n### With Inliers (9 features):\n")
        for feat in results_with['feature_names']:
            f.write(f"- {feat}\n")
        
        f.write(f"\n## Conclusion\n\n")
        if accuracy_improvement > 0.1:
            f.write(f"[SUCCESS] **Adding inliers as a feature improves model performance**\n")
            f.write(f"- Accuracy improved by {accuracy_improvement:.2f}%\n")
            f.write(f"- This follows the original task's approach of using inliers to identify hard queries\n")
        elif accuracy_improvement < -0.1:
            f.write(f"[WARNING] **Adding inliers slightly decreases model performance**\n")
            f.write(f"- Accuracy decreased by {abs(accuracy_improvement):.2f}%\n")
            f.write(f"- The 8 retrieval features are already sufficient\n")
        else:
            f.write(f"[NEUTRAL] **Adding inliers has minimal impact on model performance**\n")
            f.write(f"- Accuracy change: {accuracy_improvement:+.2f}%\n")
            f.write(f"- Both models perform similarly\n")
        
        f.write(f"\n## Important Note\n\n")
        f.write(f"[WARNING] **Using inliers requires running image matching on ALL queries first.**\n")
        f.write(f"This is acceptable for training/validation, but for new test sets, you would need to:\n")
        f.write(f"1. Run image matching on all queries (expensive!)\n")
        f.write(f"2. Extract inliers\n")
        f.write(f"3. Then predict hard queries\n")
        f.write(f"\nThis defeats the purpose of adaptive re-ranking (predicting BEFORE matching).\n")
        f.write(f"However, if you already have inliers from previous runs, using them can improve predictions.\n")
    
    print(f"\nSaved comparison report: {report_path}")
    print(f"\n{'='*70}")
    print(f"Comparison Complete!")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""

import argparse
import numpy as np
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_3_train_logreg_easy_queries import load_feature_file, build_feature_matrix


def load_model_results(model_path: Path, val_features_path: Path):
    """Load model and evaluate on validation set."""
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    optimal_threshold = bundle.get("optimal_threshold", 0.5)
    feature_names = bundle["feature_names"]
    
    # Load validation features
    val_features = load_feature_file(str(val_features_path))
    X_val, y_val, _ = build_feature_matrix(val_features)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[valid_mask]
    y_val = y_val[valid_mask]
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    probs = model.predict_proba(X_val_scaled)[:, 1]
    predictions = (probs >= optimal_threshold).astype(int)
    
    # Compute metrics
    accuracy = (predictions == y_val).mean()
    hard_query_rate = (1 - predictions.mean()) * 100
    actual_hard_rate = (1 - y_val.mean()) * 100
    
    return {
        "accuracy": accuracy,
        "hard_query_rate": hard_query_rate,
        "actual_hard_rate": actual_hard_rate,
        "optimal_threshold": optimal_threshold,
        "num_features": len(feature_names),
        "feature_names": feature_names
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare models with and without inliers"
    )
    parser.add_argument(
        "--model-without-inliers",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model trained without inliers (8 features)"
    )
    parser.add_argument(
        "--model-with-inliers",
        type=str,
        default="models/logreg_easy_with_inliers.pkl",
        help="Model trained with inliers (9 features)"
    )
    parser.add_argument(
        "--val-features-without-inliers",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Validation features without inliers"
    )
    parser.add_argument(
        "--val-features-with-inliers",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_with_inliers.npz",
        help="Validation features with inliers"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/inliers_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Comparing Models: With vs Without Inliers")
    print(f"{'='*70}\n")
    
    # Load results
    print("Loading model without inliers (8 features)...")
    results_without = load_model_results(
        Path(args.model_without_inliers),
        Path(args.val_features_without_inliers)
    )
    
    print("Loading model with inliers (9 features)...")
    results_with = load_model_results(
        Path(args.model_with_inliers),
        Path(args.val_features_with_inliers)
    )
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"Comparison Results")
    print(f"{'='*70}\n")
    
    print(f"Model WITHOUT Inliers (8 features):")
    print(f"  Accuracy: {results_without['accuracy']*100:.2f}%")
    print(f"  Hard Query Rate: {results_without['hard_query_rate']:.2f}% (actual: {results_without['actual_hard_rate']:.2f}%)")
    print(f"  Optimal Threshold: {results_without['optimal_threshold']:.3f}")
    print(f"  Features: {', '.join(results_without['feature_names'])}")
    
    print(f"\nModel WITH Inliers (9 features):")
    print(f"  Accuracy: {results_with['accuracy']*100:.2f}%")
    print(f"  Hard Query Rate: {results_with['hard_query_rate']:.2f}% (actual: {results_with['actual_hard_rate']:.2f}%)")
    print(f"  Optimal Threshold: {results_with['optimal_threshold']:.3f}")
    print(f"  Features: {', '.join(results_with['feature_names'])}")
    
    print(f"\n{'='*70}")
    print(f"Improvement Analysis")
    print(f"{'='*70}\n")
    
    accuracy_improvement = (results_with['accuracy'] - results_without['accuracy']) * 100
    hard_rate_diff = results_with['hard_query_rate'] - results_without['hard_query_rate']
    
    print(f"Accuracy: {results_without['accuracy']*100:.2f}% -> {results_with['accuracy']*100:.2f}% "
          f"({accuracy_improvement:+.2f}%)")
    print(f"Hard Query Rate: {results_without['hard_query_rate']:.2f}% -> {results_with['hard_query_rate']:.2f}% "
          f"({hard_rate_diff:+.2f}%)")
    
    if accuracy_improvement > 0:
        print(f"\n[SUCCESS] Adding inliers improved accuracy by {accuracy_improvement:.2f}%")
    elif accuracy_improvement < 0:
        print(f"\n[WARNING] Adding inliers decreased accuracy by {abs(accuracy_improvement):.2f}%")
    else:
        print(f"\n[NEUTRAL] Adding inliers did not change accuracy")
    
    # Save comparison report
    report_path = output_dir / "inliers_comparison_report.md"
    with open(report_path, 'w') as f:
        f.write("# Model Comparison: With vs Without Inliers Feature\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Metric | Without Inliers (8 features) | With Inliers (9 features) | Improvement |\n")
        f.write("|--------|------------------------------|--------------------------|-------------|\n")
        f.write(f"| Accuracy | {results_without['accuracy']*100:.2f}% | {results_with['accuracy']*100:.2f}% | "
               f"{accuracy_improvement:+.2f}% |\n")
        f.write(f"| Hard Query Rate | {results_without['hard_query_rate']:.2f}% | {results_with['hard_query_rate']:.2f}% | "
               f"{hard_rate_diff:+.2f}% |\n")
        f.write(f"| Optimal Threshold | {results_without['optimal_threshold']:.3f} | {results_with['optimal_threshold']:.3f} | "
               f"{results_with['optimal_threshold'] - results_without['optimal_threshold']:+.3f} |\n")
        
        f.write(f"\n## Feature Comparison\n\n")
        f.write(f"### Without Inliers (8 features):\n")
        for feat in results_without['feature_names']:
            f.write(f"- {feat}\n")
        
        f.write(f"\n### With Inliers (9 features):\n")
        for feat in results_with['feature_names']:
            f.write(f"- {feat}\n")
        
        f.write(f"\n## Conclusion\n\n")
        if accuracy_improvement > 0.1:
            f.write(f"[SUCCESS] **Adding inliers as a feature improves model performance**\n")
            f.write(f"- Accuracy improved by {accuracy_improvement:.2f}%\n")
            f.write(f"- This follows the original task's approach of using inliers to identify hard queries\n")
        elif accuracy_improvement < -0.1:
            f.write(f"[WARNING] **Adding inliers slightly decreases model performance**\n")
            f.write(f"- Accuracy decreased by {abs(accuracy_improvement):.2f}%\n")
            f.write(f"- The 8 retrieval features are already sufficient\n")
        else:
            f.write(f"[NEUTRAL] **Adding inliers has minimal impact on model performance**\n")
            f.write(f"- Accuracy change: {accuracy_improvement:+.2f}%\n")
            f.write(f"- Both models perform similarly\n")
        
        f.write(f"\n## Important Note\n\n")
        f.write(f"[WARNING] **Using inliers requires running image matching on ALL queries first.**\n")
        f.write(f"This is acceptable for training/validation, but for new test sets, you would need to:\n")
        f.write(f"1. Run image matching on all queries (expensive!)\n")
        f.write(f"2. Extract inliers\n")
        f.write(f"3. Then predict hard queries\n")
        f.write(f"\nThis defeats the purpose of adaptive re-ranking (predicting BEFORE matching).\n")
        f.write(f"However, if you already have inliers from previous runs, using them can improve predictions.\n")
    
    print(f"\nSaved comparison report: {report_path}")
    print(f"\n{'='*70}")
    print(f"Comparison Complete!")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

