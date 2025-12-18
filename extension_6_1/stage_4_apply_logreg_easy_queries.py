"""
Stage 4 (Improved): Apply Logistic Regression to predict EASY queries.

This version:
- Predicts "easy" queries (probability of being easy/correct)
- Uses optimal threshold saved with model (no hard thresholding)
- Skips re-ranking for easy queries
"""

import argparse
from pathlib import Path
import joblib
import numpy as np


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


def main(args):
    # Load model
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    optimal_threshold = bundle.get("optimal_threshold", 0.5)
    threshold_method = bundle.get("threshold_method", "f1")
    target_type = bundle.get("target_type", "easy_score")
    
    print(f"Loaded model: {args.model_path}")
    print(f"  Features: {', '.join(feature_names)} ({len(feature_names)} features)")
    print(f"  Target: {target_type} (1 = easy/correct, 0 = hard/wrong)")
    print(f"  Optimal threshold: {optimal_threshold:.3f} (method: {threshold_method})")
    
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
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
    
    # Apply optimal threshold (no hard thresholding - threshold is learned)
    is_easy = probs >= optimal_threshold
    is_hard = ~is_easy
    
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
        "valid_mask": valid_mask,
        "target_type": target_type,
    }
    
    if "labels" in features:
        save_dict["labels"] = features["labels"][valid_mask].astype("float32")
    
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
    print(f"Query Detection (BEFORE Image Matching) - LOGISTIC REGRESSION (EASY)")
    print(f"{'='*70}")
    print(f"Processed {num_queries} queries using {len(feature_names)} retrieval features:")
    print(f"  - {', '.join(feature_names)}")
    print(f"  - Model predicts: easy_score (probability of being easy/correct)")
    print(f"  - Optimal threshold: {optimal_threshold:.3f} (learned from validation)")
    print(f"\nPredictions:")
    print(f"  Easy (predicted prob >= {optimal_threshold:.3f}, skip image matching): {num_easy} ({100*num_easy/num_queries:.1f}%)")
    print(f"  Hard (predicted prob < {optimal_threshold:.3f}, apply image matching): {num_hard} ({100*num_hard/num_queries:.1f}%)")
    print(f"\nPredicted probability statistics:")
    print(f"  Min: {probs.min():.3f}, Max: {probs.max():.3f}")
    print(f"  Mean: {probs.mean():.3f}, Median: {np.median(probs):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Apply logistic regression to predict EASY queries"
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
    
    args = parser.parse_args()
    main(args)

