"""
Find threshold that achieves target performance ratio of full re-ranking.

This script:
1. Tests different thresholds
2. Evaluates adaptive re-ranking performance for each threshold
3. Finds threshold that achieves target performance ratio (e.g., 90% of full re-ranking)
4. Balances performance vs efficiency
"""

import argparse
from pathlib import Path
import numpy as np
import joblib
import torch
from glob import glob
import os
from tqdm import tqdm

from util import get_list_distances_from_preds


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
    """Build feature matrix based on expected feature names from model."""
    feature_arrays = []
    for name in expected_feature_names:
        if name not in features_dict:
            raise ValueError(f"Feature '{name}' not found in feature file")
        feature_arrays.append(features_dict[name])
    
    X = np.stack(feature_arrays, axis=1).astype("float32")
    return X


def compute_recall_at_1_adaptive(
    preds_dir: Path,
    inliers_dir: Path,
    is_hard: np.ndarray,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> float:
    """
    Compute Recall@1 for adaptive re-ranking strategy.
    
    Args:
        preds_dir: Directory with prediction .txt files
        inliers_dir: Directory with .torch files (image matching results)
        is_hard: Boolean array indicating which queries are hard
        num_preds: Number of predictions to consider
        positive_dist_threshold: Distance threshold in meters
    
    Returns:
        Recall@1 as percentage
    """
    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    if total_queries != len(is_hard):
        raise ValueError(f"Mismatch: {total_queries} txt files vs {len(is_hard)} hard flags")
    
    correct_at_1 = 0
    
    for idx, txt_file_query in enumerate(txt_files):
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
            txt_filename = Path(txt_file_query).name
            torch_filename = txt_filename.replace(".txt", ".torch")
            torch_file_query = inliers_dir / torch_filename
            
            if is_hard[idx] and torch_file_query.exists():
                # Hard query: re-rank by inliers
                query_results = torch.load(str(torch_file_query), weights_only=False)
                actual_num_preds = min(len(query_results), num_preds, len(geo_dists))
                if actual_num_preds == 0:
                    continue
                
                query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
                for i in range(actual_num_preds):
                    query_db_inliers[i] = query_results[i]["num_inliers"]
                query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
                query_geo_dists = geo_dists[:actual_num_preds][indices]
            else:
                # Easy query: use retrieval-only ordering
                query_geo_dists = geo_dists
            
            # Check if Top-1 is correct
            if len(query_geo_dists) > 0 and query_geo_dists[0] <= positive_dist_threshold:
                correct_at_1 += 1
                
        except Exception as e:
            continue
    
    recall_at_1 = (correct_at_1 / total_queries) * 100.0 if total_queries > 0 else 0.0
    return recall_at_1


def find_performance_targeted_threshold(
    model_path: Path,
    feature_path: Path,
    preds_dir: Path,
    inliers_dir: Path,
    full_reranking_recall_at_1: float,
    target_performance_ratio: float = 0.90,
    threshold_start: float = 0.5,
    threshold_end: float = 0.99,
    threshold_step: float = 0.01,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> tuple:
    """
    Find threshold that achieves target performance ratio of full re-ranking.
    
    Args:
        model_path: Path to trained model
        feature_path: Path to feature file
        preds_dir: Directory with prediction files
        inliers_dir: Directory with inlier files (from full re-ranking)
        full_reranking_recall_at_1: Full re-ranking R@1 (ground truth)
        target_performance_ratio: Target performance ratio (0.90 = 90% of full re-ranking)
        threshold_start: Start threshold for search
        threshold_end: End threshold for search
        threshold_step: Step size for threshold search
        num_preds: Number of predictions to consider
        positive_dist_threshold: Distance threshold in meters
    
    Returns:
        optimal_threshold: Threshold that achieves target performance
        achieved_recall_at_1: Achieved R@1
        hard_query_rate: Percentage of hard queries
        performance_ratio: Achieved performance ratio
    """
    print(f"\n{'='*70}")
    print(f"Performance-Targeted Threshold Finding")
    print(f"{'='*70}")
    print(f"Full re-ranking R@1: {full_reranking_recall_at_1:.2f}%")
    print(f"Target performance ratio: {target_performance_ratio:.1%}")
    print(f"Target R@1: {full_reranking_recall_at_1 * target_performance_ratio:.2f}%")
    print(f"{'='*70}\n")
    
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    
    # Load features
    features = load_feature_file(str(feature_path))
    X = build_feature_matrix(features, feature_names)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    
    if (~valid_mask).sum() > 0:
        print(f"Removed {(~valid_mask).sum()} queries with NaN features")
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
    
    num_queries = len(probs)
    print(f"Total queries: {num_queries}")
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Mean probability: {probs.mean():.3f}\n")
    
    # Try thresholds (conservative: higher threshold = more hard queries)
    thresholds = np.arange(threshold_start, threshold_end + threshold_step, threshold_step)
    print(f"Testing {len(thresholds)} thresholds from {threshold_start:.2f} to {threshold_end:.2f}...\n")
    
    best_threshold = threshold_start
    best_performance_ratio = 0.0
    best_recall_at_1 = 0.0
    best_hard_query_rate = 0.0
    
    results = []
    
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        # Classify queries
        is_easy = probs >= threshold
        is_hard = ~is_easy
        
        hard_query_rate = is_hard.mean()
        
        # Compute adaptive re-ranking performance
        recall_at_1 = compute_recall_at_1_adaptive(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        
        performance_ratio = recall_at_1 / full_reranking_recall_at_1 if full_reranking_recall_at_1 > 0 else 0.0
        
        results.append({
            "threshold": threshold,
            "recall_at_1": recall_at_1,
            "hard_query_rate": hard_query_rate,
            "performance_ratio": performance_ratio
        })
        
        # Check if this threshold achieves target
        if performance_ratio >= target_performance_ratio:
            if performance_ratio > best_performance_ratio:
                best_threshold = threshold
                best_performance_ratio = performance_ratio
                best_recall_at_1 = recall_at_1
                best_hard_query_rate = hard_query_rate
    
    # If no threshold achieves target, use best one
    if best_performance_ratio < target_performance_ratio:
        # Find threshold with highest performance ratio
        best_result = max(results, key=lambda x: x["performance_ratio"])
        best_threshold = best_result["threshold"]
        best_performance_ratio = best_result["performance_ratio"]
        best_recall_at_1 = best_result["recall_at_1"]
        best_hard_query_rate = best_result["hard_query_rate"]
        print(f"\nWARNING: No threshold achieved target {target_performance_ratio:.1%}")
        print(f"   Using best threshold: {best_threshold:.3f} (performance ratio: {best_performance_ratio:.1%})")
    
    print(f"\n{'='*70}")
    print(f"Optimal Threshold Found")
    print(f"{'='*70}")
    print(f"Threshold: {best_threshold:.3f}")
    print(f"Hard query rate: {best_hard_query_rate*100:.1f}%")
    print(f"Adaptive R@1: {best_recall_at_1:.2f}%")
    print(f"Full re-ranking R@1: {full_reranking_recall_at_1:.2f}%")
    print(f"Performance ratio: {best_performance_ratio:.1%}")
    print(f"Time savings: {(1 - best_hard_query_rate)*100:.1f}%")
    print(f"{'='*70}\n")
    
    return best_threshold, best_recall_at_1, best_hard_query_rate, best_performance_ratio


def main():
    parser = argparse.ArgumentParser(
        description="Find threshold that achieves target performance ratio of full re-ranking"
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
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with prediction .txt files",
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="Directory with .torch files (from full re-ranking)",
    )
    parser.add_argument(
        "--full-reranking-r1",
        type=float,
        required=True,
        help="Full re-ranking Recall@1 (ground truth)",
    )
    parser.add_argument(
        "--target-performance-ratio",
        type=float,
        default=0.90,
        help="Target performance ratio (0.90 = 90%% of full re-ranking)",
    )
    parser.add_argument(
        "--threshold-start",
        type=float,
        default=0.5,
        help="Start threshold for search (default: 0.5)",
    )
    parser.add_argument(
        "--threshold-end",
        type=float,
        default=0.99,
        help="End threshold for search (default: 0.99)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size for threshold search (default: 0.01)",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions to consider (default: 20)",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold in meters (default: 25)",
    )
    parser.add_argument(
        "--output-threshold",
        type=str,
        default=None,
        help="Path to save optimal threshold (optional)",
    )
    
    args = parser.parse_args()
    
    # Find optimal threshold
    optimal_threshold, achieved_r1, hard_query_rate, performance_ratio = find_performance_targeted_threshold(
        model_path=Path(args.model_path),
        feature_path=Path(args.feature_path),
        preds_dir=Path(args.preds_dir),
        inliers_dir=Path(args.inliers_dir),
        full_reranking_recall_at_1=args.full_reranking_r1,
        target_performance_ratio=args.target_performance_ratio,
        threshold_start=args.threshold_start,
        threshold_end=args.threshold_end,
        threshold_step=args.threshold_step,
        num_preds=args.num_preds,
        positive_dist_threshold=args.positive_dist_threshold
    )
    
    # Save threshold if requested
    if args.output_threshold:
        with open(args.output_threshold, 'w') as f:
            f.write(f"{optimal_threshold:.6f}\n")
        print(f"Saved optimal threshold to {args.output_threshold}")
    
    print(f"\nDone! Use threshold {optimal_threshold:.3f} for this dataset.")


if __name__ == "__main__":
    main()

