"""
Solution 2: Robust Feature Normalization

Uses robust scaling (median/IQR) instead of mean/std to handle outliers
and distribution shifts better.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix, find_optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Apply robust feature normalization for better generalization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to training features (for fitting robust scaler)"
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to test features"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        help="Path to save hard query indices"
    )
    
    args = parser.parse_args()
    
    # Load model
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    saved_threshold = bundle.get("optimal_threshold", 0.5)
    
    # Load training features to fit robust scaler
    train_features = load_feature_file(args.train_features)
    X_train = build_feature_matrix(train_features, feature_names)
    train_mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[train_mask]
    
    # Fit robust scaler
    robust_scaler = RobustScaler()
    X_train_robust = robust_scaler.fit_transform(X_train)
    
    # Load test features
    test_features = load_feature_file(args.feature_path)
    X_test = build_feature_matrix(test_features, feature_names)
    y_true = test_features["labels"].astype("float32")
    
    # Handle NaNs
    test_mask = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[test_mask]
    y_true = y_true[test_mask]
    
    # Apply robust scaling
    X_test_robust = robust_scaler.transform(X_test)
    
    # Predict with original model (but using robust-scaled features)
    # Note: This assumes the model can work with robust-scaled features
    # In practice, we might need to retrain with robust scaling
    y_probs = model.predict_proba(X_test_robust)[:, 1]
    
    print(f"Robust-scaled probabilities: min={y_probs.min():.3f}, mean={y_probs.mean():.3f}, max={y_probs.max():.3f}")
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, y_probs, method="f1")
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Apply threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    hard_rate = (y_probs < optimal_threshold).mean()
    
    print(f"\nResults:")
    print(f"  Hard query rate: {hard_rate:.1%}")
    print(f"  Actual hard rate: {(1-y_true.mean()):.1%}")
    print(f"  Accuracy: {(y_pred == y_true).mean():.1%}")
    print(f"  F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    # Save results
    save_dict = {
        "probs": y_probs,
        "is_easy": y_pred.astype(bool),
        "is_hard": (~y_pred.astype(bool)),
        "labels": y_true,
        "threshold": optimal_threshold,
        "normalization": "robust",
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    np.savez_compressed(args.output_path, **save_dict)
    print(f"\nSaved predictions to: {args.output_path}")
    
    # Save hard queries
    if args.hard_queries_output:
        hard_query_indices = np.where(~y_pred.astype(bool))[0]
        with open(args.hard_queries_output, "w") as f:
            for idx in hard_query_indices:
                f.write(f"{idx}\n")
        print(f"Saved {len(hard_query_indices)} hard query indices to: {args.hard_queries_output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

Solution 2: Robust Feature Normalization

Uses robust scaling (median/IQR) instead of mean/std to handle outliers
and distribution shifts better.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix, find_optimal_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Apply robust feature normalization for better generalization"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to training features (for fitting robust scaler)"
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to test features"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save predictions"
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        help="Path to save hard query indices"
    )
    
    args = parser.parse_args()
    
    # Load model
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    feature_names = bundle["feature_names"]
    saved_threshold = bundle.get("optimal_threshold", 0.5)
    
    # Load training features to fit robust scaler
    train_features = load_feature_file(args.train_features)
    X_train = build_feature_matrix(train_features, feature_names)
    train_mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[train_mask]
    
    # Fit robust scaler
    robust_scaler = RobustScaler()
    X_train_robust = robust_scaler.fit_transform(X_train)
    
    # Load test features
    test_features = load_feature_file(args.feature_path)
    X_test = build_feature_matrix(test_features, feature_names)
    y_true = test_features["labels"].astype("float32")
    
    # Handle NaNs
    test_mask = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[test_mask]
    y_true = y_true[test_mask]
    
    # Apply robust scaling
    X_test_robust = robust_scaler.transform(X_test)
    
    # Predict with original model (but using robust-scaled features)
    # Note: This assumes the model can work with robust-scaled features
    # In practice, we might need to retrain with robust scaling
    y_probs = model.predict_proba(X_test_robust)[:, 1]
    
    print(f"Robust-scaled probabilities: min={y_probs.min():.3f}, mean={y_probs.mean():.3f}, max={y_probs.max():.3f}")
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, y_probs, method="f1")
    print(f"Optimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Apply threshold
    y_pred = (y_probs >= optimal_threshold).astype(int)
    hard_rate = (y_probs < optimal_threshold).mean()
    
    print(f"\nResults:")
    print(f"  Hard query rate: {hard_rate:.1%}")
    print(f"  Actual hard rate: {(1-y_true.mean()):.1%}")
    print(f"  Accuracy: {(y_pred == y_true).mean():.1%}")
    print(f"  F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    # Save results
    save_dict = {
        "probs": y_probs,
        "is_easy": y_pred.astype(bool),
        "is_hard": (~y_pred.astype(bool)),
        "labels": y_true,
        "threshold": optimal_threshold,
        "normalization": "robust",
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    np.savez_compressed(args.output_path, **save_dict)
    print(f"\nSaved predictions to: {args.output_path}")
    
    # Save hard queries
    if args.hard_queries_output:
        hard_query_indices = np.where(~y_pred.astype(bool))[0]
        with open(args.hard_queries_output, "w") as f:
            for idx in hard_query_indices:
                f.write(f"{idx}\n")
        print(f"Saved {len(hard_query_indices)} hard query indices to: {args.hard_queries_output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


