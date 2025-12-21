"""
Stage 4 (Improved): Apply Logistic Regression to predict HARD queries.

This version:
- Predicts "hard" queries (probability of being hard/wrong)
- Supports dataset-specific threshold calibration
- Can use saved threshold OR calibrate on test set
- Applies re-ranking for hard queries
"""

import argparse
from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


def load_feature_file(path: str) -> dict:
    """Load feature file and return dictionary."""
    data = np.load(path)
    result = {
        "labels": data["labels"].astype("float32"),
        "top1_distance": data["top1_distance"].astype("float32"),
        "peakiness": data["peakiness"].astype("float32"),
        "sue_score": data["sue_score"].astype("float32"),
    }
    # Add new features if available
    if "topk_distance_spread" in data:
        result["topk_distance_spread"] = data["topk_distance_spread"].astype("float32")
        result["top1_top2_similarity"] = data["top1_top2_similarity"].astype("float32")
        result["top1_top3_ratio"] = data["top1_top3_ratio"].astype("float32")
        result["top2_top3_ratio"] = data["top2_top3_ratio"].astype("float32")
        result["geographic_clustering"] = data["geographic_clustering"].astype("float32")
    
    # Add inliers if available (9th feature)
    if "num_inliers_top1" in data:
        result["num_inliers_top1"] = data["num_inliers_top1"].astype("float32")
    
    return result


def build_feature_matrix(features_dict: dict, expected_feature_names: list) -> np.ndarray:
    """
    Build feature matrix based on expected feature names from model.
    """
    feature_arrays = []
    for name in expected_feature_names:
        if name not in features_dict:
            raise ValueError(f"Feature '{name}' not found in feature file")
        feature_arrays.append(features_dict[name])
    
    X = np.stack(feature_arrays, axis=1).astype("float32")
    return X


def find_optimal_threshold(y_true, y_probs, method="f1", target_rate=None):
    """
    Find optimal threshold on test set.
    
    Args:
        y_true: True labels (1 = hard, 0 = easy)
        y_probs: Predicted probabilities (probability of being hard)
        method: "f1" (maximize F1), "recall" (target recall), or "rate" (target hard query rate)
        target_rate: Target hard query rate (0.0-1.0) if method="rate"
    
    Returns:
        optimal_threshold: Best threshold value
        best_score: Best F1, recall, or achieved rate
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)  # 1 = hard, 0 = easy
        
        if method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        elif method == "rate":
            # Target hard query rate (percentage of queries predicted as hard)
            hard_rate = y_pred.mean()  # 1 = hard
            score = 1.0 - abs(hard_rate - target_rate)  # Closer to target = better
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def main(args):
    # Load model
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    saved_threshold = bundle.get("optimal_threshold", 0.5)
    threshold_method = bundle.get("threshold_method", "f1")
    # This project uses "hard_score" models by default (1 = hard/wrong, 0 = easy/correct).
    target_type = bundle.get("target_type", "hard_score")
    
    print(f"Loaded model: {args.model_path}")
    print(f"  Features: {', '.join(feature_names)} ({len(feature_names)} features)")
    print(f"  Target: {target_type} (1 = hard/wrong, 0 = easy/correct)")
    print(f"  Saved threshold: {saved_threshold:.3f} (method: {threshold_method})")
    
    # Load features
    features = load_feature_file(args.feature_path)
    X = build_feature_matrix(features, feature_names)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    
    if (~valid_mask).sum() > 0:
        print(f"Removed {(~valid_mask).sum()} queries with NaN features")
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being hard
    
    # Determine which threshold to use
    if args.calibrate_threshold and "labels" in features:
        # Calibrate threshold on test set (if labels available)
        labels = features["labels"][valid_mask].astype("float32")
        hard_score = (1 - labels).astype("float32")  # 1 = hard/wrong, 0 = easy/correct
        
        print(f"\n{'='*70}")
        print(f"Calibrating threshold on test set...")
        print(f"{'='*70}")
        
        if args.target_hard_rate is not None:
            # Target-based calibration: achieve specific hard query rate
            target_rate = args.target_hard_rate
            optimal_threshold, achieved_rate = find_optimal_threshold(
                hard_score, probs, method="rate", target_rate=target_rate
            )
            print(f"  Method: Target hard query rate ({target_rate*100:.1f}%)")
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
            print(f"  Achieved hard query rate: {achieved_rate*100:.1f}%")
        else:
            # F1-based calibration: maximize F1-score
            optimal_threshold, best_f1 = find_optimal_threshold(
                hard_score, probs, method="f1"
            )
            y_pred = (probs >= optimal_threshold).astype(int)
            precision = precision_score(hard_score, y_pred, zero_division=0)
            recall = recall_score(hard_score, y_pred, zero_division=0)
            
            print(f"  Method: Maximize F1-score")
            print(f"  Optimal threshold: {optimal_threshold:.3f}")
            print(f"  F1-score: {best_f1:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
        
        print(f"  (Saved threshold was {saved_threshold:.3f})")
        threshold_source = "calibrated_on_test"
    else:
        # Use saved threshold from validation
        optimal_threshold = saved_threshold
        threshold_source = "saved_from_validation"
        if args.calibrate_threshold:
            print(f"\n⚠️  Warning: Cannot calibrate threshold - labels not available in feature file")
            print(f"  Using saved threshold: {optimal_threshold:.3f}")
    
    # Apply threshold (probs is now probability of being hard)
    is_hard = probs >= optimal_threshold  # If P(hard) >= threshold → Hard
    is_easy = ~is_hard                    # If P(hard) < threshold → Easy
    
    num_queries = len(probs)
    num_easy = is_easy.sum()
    num_hard = is_hard.sum()
    
    # Get hard query indices (for image matching)
    hard_query_indices = np.where(is_hard)[0]
    
    # Save outputs
    save_dict = {
        "probs": probs.astype("float32"),
        "is_easy": is_easy,
        "is_hard": is_hard,
        "hard_query_indices": hard_query_indices.astype("int32"),
        "optimal_threshold": optimal_threshold,
        "threshold_source": threshold_source,
        "valid_mask": valid_mask,
        "target_type": target_type,
    }
    
    if "labels" in features:
        save_dict["labels"] = features["labels"][valid_mask].astype("float32")
        # Compute accuracy if labels available
        # labels: 1 = correct, 0 = wrong
        # is_easy: True = easy, False = hard
        # So: is_easy should match (labels == 1)
        accuracy = (is_easy.astype(int) == save_dict["labels"]).mean()
        save_dict["accuracy"] = accuracy
    
    output_path = Path(args.output_path)
    np.savez_compressed(output_path, **save_dict)
    print(f"\nSaved results to {output_path}")
    
    # Save hard query indices to text file
    if args.hard_queries_output:
        hard_queries_path = Path(args.hard_queries_output)
        with open(hard_queries_path, "w") as f:
            for idx in hard_query_indices:
                f.write(f"{idx}\n")
        print(f"Saved hard query indices to {hard_queries_path}")
        print(f"  -> Use this file to run image matching ONLY on hard queries")
        print(f"  -> Expected time savings: {100*num_easy/num_queries:.1f}% of image matching time")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Query Detection (BEFORE Image Matching) - LOGISTIC REGRESSION (HARD)")
    print(f"{'='*70}")
    print(f"Processed {num_queries} queries using {len(feature_names)} retrieval features:")
    print(f"  - {', '.join(feature_names)}")
    print(f"  - Model predicts: hard_score (probability of being hard/wrong)")
    print(f"  - Threshold: {optimal_threshold:.3f} ({threshold_source})")
    print(f"\nPredictions:")
    print(f"  Hard (predicted P(hard) >= {optimal_threshold:.3f}, apply image matching): {num_hard} ({100*num_hard/num_queries:.1f}%)")
    print(f"  Easy (predicted P(hard) < {optimal_threshold:.3f}, skip image matching): {num_easy} ({100*num_easy/num_queries:.1f}%)")
    print(f"\nPredicted probability statistics:")
    print(f"  Min: {probs.min():.3f}, Max: {probs.max():.3f}")
    print(f"  Mean: {probs.mean():.3f}, Median: {np.median(probs):.3f}")
    
    if "labels" in features:
        actually_wrong = (save_dict["labels"] == 0).sum()
        actually_correct = (save_dict["labels"] == 1).sum()
        print(f"\nGround truth (if available):")
        print(f"  Actually wrong: {actually_wrong} ({100*actually_wrong/num_queries:.1f}%)")
        print(f"  Actually correct: {actually_correct} ({100*actually_correct/num_queries:.1f}%)")
        if "accuracy" in save_dict:
            print(f"  Detection accuracy: {save_dict['accuracy']*100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply logistic regression to predict HARD queries with optional threshold calibration"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained logistic regression model (.pkl)",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to feature file (.npz)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save predictions (.npz)",
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        default=None,
        help="Path to save hard query indices (.txt)",
    )
    parser.add_argument(
        "--calibrate-threshold",
        action="store_true",
        help="Calibrate threshold on test set (requires labels in feature file)",
    )
    parser.add_argument(
        "--target-hard-rate",
        type=float,
        default=None,
        help="Target hard query rate (0.0-1.0) for threshold calibration. E.g., 0.30 for 30%% hard queries",
    )
    
    args = parser.parse_args()
    main(args)
