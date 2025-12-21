"""
Solution 3: Ensemble Method - Combine predictions from multiple models

This combines predictions from all 3 models (Night+Sun, Night Only, Sun Only)
using weighted averaging or voting to improve robustness.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix, find_optimal_threshold


def ensemble_predictions(model_paths, feature_path, method="average"):
    """
    Combine predictions from multiple models.
    
    Args:
        model_paths: List of paths to model files
        feature_path: Path to feature file
        method: "average" (weighted average) or "vote" (majority voting)
    
    Returns:
        ensemble_probs: Combined probabilities
        individual_probs: List of probabilities from each model
    """
    # Load features
    features = load_feature_file(feature_path)
    
    individual_probs = []
    individual_models = []
    
    for model_path in model_paths:
        # Load model
        bundle = joblib.load(Path(model_path))
        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_names = bundle["feature_names"]
        
        # Build feature matrix
        X = build_feature_matrix(features, feature_names)
        
        # Handle NaNs
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        individual_probs.append(probs)
        individual_models.append({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "valid_mask": valid_mask
        })
    
    # Combine predictions
    if method == "average":
        # Weighted average (equal weights for now)
        ensemble_probs = np.mean(individual_probs, axis=0)
    elif method == "vote":
        # Majority voting (convert to binary first, then vote)
        thresholds = [0.390, 0.110, 0.500]  # From the 3 models
        votes = []
        for i, probs in enumerate(individual_probs):
            votes.append((probs >= thresholds[i]).astype(int))
        votes = np.array(votes)
        ensemble_votes = (votes.sum(axis=0) >= 2).astype(int)  # Majority
        # Convert back to probabilities (approximate)
        ensemble_probs = ensemble_votes.astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_probs, individual_probs, individual_models[0]["valid_mask"]


def main():
    parser = argparse.ArgumentParser(
        description="Apply ensemble method combining multiple models"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_only.pkl",
        help="Model 2 (Night Only)"
    )
    parser.add_argument(
        "--model3-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_sun_only.pkl",
        help="Model 3 (Sun Only)"
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
        help="Path to save ensemble predictions"
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        help="Path to save hard query indices"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["average", "vote"],
        default="average",
        help="Ensemble method: average or vote"
    )
    
    args = parser.parse_args()
    
    model_paths = [args.model1_path, args.model2_path, args.model3_path]
    
    print(f"Ensemble Method: {args.method}")
    print(f"Combining {len(model_paths)} models...")
    
    # Get ensemble predictions
    ensemble_probs, individual_probs, valid_mask = ensemble_predictions(
        model_paths, args.feature_path, method=args.method
    )
    
    # Load labels
    features = load_feature_file(args.feature_path)
    y_true = features["labels"].astype("float32")[valid_mask]
    
    print(f"\nEnsemble probability statistics:")
    print(f"  Min: {ensemble_probs.min():.3f}, Max: {ensemble_probs.max():.3f}")
    print(f"  Mean: {ensemble_probs.mean():.3f}, Median: {np.median(ensemble_probs):.3f}")
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, ensemble_probs, method="f1")
    print(f"\nOptimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Apply threshold
    y_pred = (ensemble_probs >= optimal_threshold).astype(int)
    hard_rate = (ensemble_probs < optimal_threshold).mean()
    
    print(f"\nResults:")
    print(f"  Hard query rate: {hard_rate:.1%}")
    print(f"  Actual hard rate: {(1-y_true.mean()):.1%}")
    print(f"  Accuracy: {(y_pred == y_true).mean():.1%}")
    print(f"  F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    # Save results
    save_dict = {
        "probs": ensemble_probs,
        "is_easy": y_pred.astype(bool),
        "is_hard": (~y_pred.astype(bool)),
        "labels": y_true,
        "threshold": optimal_threshold,
        "method": args.method,
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Add individual model probabilities
    for i, probs in enumerate(individual_probs):
        save_dict[f"model_{i+1}_probs"] = probs
    
    np.savez_compressed(args.output_path, **save_dict)
    print(f"\nSaved ensemble predictions to: {args.output_path}")
    
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

Solution 3: Ensemble Method - Combine predictions from multiple models

This combines predictions from all 3 models (Night+Sun, Night Only, Sun Only)
using weighted averaging or voting to improve robustness.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix, find_optimal_threshold


def ensemble_predictions(model_paths, feature_path, method="average"):
    """
    Combine predictions from multiple models.
    
    Args:
        model_paths: List of paths to model files
        feature_path: Path to feature file
        method: "average" (weighted average) or "vote" (majority voting)
    
    Returns:
        ensemble_probs: Combined probabilities
        individual_probs: List of probabilities from each model
    """
    # Load features
    features = load_feature_file(feature_path)
    
    individual_probs = []
    individual_models = []
    
    for model_path in model_paths:
        # Load model
        bundle = joblib.load(Path(model_path))
        model = bundle["model"]
        scaler = bundle["scaler"]
        feature_names = bundle["feature_names"]
        
        # Build feature matrix
        X = build_feature_matrix(features, feature_names)
        
        # Handle NaNs
        valid_mask = ~np.isnan(X).any(axis=1)
        X = X[valid_mask]
        
        # Scale and predict
        X_scaled = scaler.transform(X)
        probs = model.predict_proba(X_scaled)[:, 1]
        
        individual_probs.append(probs)
        individual_models.append({
            "model": model,
            "scaler": scaler,
            "feature_names": feature_names,
            "valid_mask": valid_mask
        })
    
    # Combine predictions
    if method == "average":
        # Weighted average (equal weights for now)
        ensemble_probs = np.mean(individual_probs, axis=0)
    elif method == "vote":
        # Majority voting (convert to binary first, then vote)
        thresholds = [0.390, 0.110, 0.500]  # From the 3 models
        votes = []
        for i, probs in enumerate(individual_probs):
            votes.append((probs >= thresholds[i]).astype(int))
        votes = np.array(votes)
        ensemble_votes = (votes.sum(axis=0) >= 2).astype(int)  # Majority
        # Convert back to probabilities (approximate)
        ensemble_probs = ensemble_votes.astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return ensemble_probs, individual_probs, individual_models[0]["valid_mask"]


def main():
    parser = argparse.ArgumentParser(
        description="Apply ensemble method combining multiple models"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_only.pkl",
        help="Model 2 (Night Only)"
    )
    parser.add_argument(
        "--model3-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_sun_only.pkl",
        help="Model 3 (Sun Only)"
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
        help="Path to save ensemble predictions"
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        help="Path to save hard query indices"
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["average", "vote"],
        default="average",
        help="Ensemble method: average or vote"
    )
    
    args = parser.parse_args()
    
    model_paths = [args.model1_path, args.model2_path, args.model3_path]
    
    print(f"Ensemble Method: {args.method}")
    print(f"Combining {len(model_paths)} models...")
    
    # Get ensemble predictions
    ensemble_probs, individual_probs, valid_mask = ensemble_predictions(
        model_paths, args.feature_path, method=args.method
    )
    
    # Load labels
    features = load_feature_file(args.feature_path)
    y_true = features["labels"].astype("float32")[valid_mask]
    
    print(f"\nEnsemble probability statistics:")
    print(f"  Min: {ensemble_probs.min():.3f}, Max: {ensemble_probs.max():.3f}")
    print(f"  Mean: {ensemble_probs.mean():.3f}, Median: {np.median(ensemble_probs):.3f}")
    
    # Find optimal threshold
    optimal_threshold, best_f1 = find_optimal_threshold(y_true, ensemble_probs, method="f1")
    print(f"\nOptimal threshold: {optimal_threshold:.3f} (F1: {best_f1:.4f})")
    
    # Apply threshold
    y_pred = (ensemble_probs >= optimal_threshold).astype(int)
    hard_rate = (ensemble_probs < optimal_threshold).mean()
    
    print(f"\nResults:")
    print(f"  Hard query rate: {hard_rate:.1%}")
    print(f"  Actual hard rate: {(1-y_true.mean()):.1%}")
    print(f"  Accuracy: {(y_pred == y_true).mean():.1%}")
    print(f"  F1-Score: {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    # Save results
    save_dict = {
        "probs": ensemble_probs,
        "is_easy": y_pred.astype(bool),
        "is_hard": (~y_pred.astype(bool)),
        "labels": y_true,
        "threshold": optimal_threshold,
        "method": args.method,
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    # Add individual model probabilities
    for i, probs in enumerate(individual_probs):
        save_dict[f"model_{i+1}_probs"] = probs
    
    np.savez_compressed(args.output_path, **save_dict)
    print(f"\nSaved ensemble predictions to: {args.output_path}")
    
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


