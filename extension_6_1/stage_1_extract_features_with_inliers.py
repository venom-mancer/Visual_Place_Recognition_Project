"""
Stage 1 (Enhanced): Extract features INCLUDING inliers for top-1 match.

This version extracts:
- 8 retrieval-based features (available before matching)
- num_inliers_top1 (requires image matching results)

This follows the original task's approach of using inliers as a feature.
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent.parent))

from extension_6_1.stage_1_extract_features_no_inliers import extract_retrieval_features


def extract_inliers_from_matching(preds_dir: Path, inliers_dir: Path, num_queries: int) -> np.ndarray:
    """
    Extract num_inliers for top-1 match from image matching results.
    
    Args:
        preds_dir: Directory with prediction .txt files
        inliers_dir: Directory with .torch files (image matching results)
        num_queries: Number of queries
    
    Returns:
        Array of num_inliers for each query (top-1 match only)
    """
    inliers = np.zeros(num_queries, dtype=np.float32)
    
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    print(f"Extracting inliers from {len(txt_files)} queries...")
    
    for txt_file in tqdm(txt_files, desc="Loading inliers"):
        try:
            query_idx = int(txt_file.stem)
            torch_file = inliers_dir / f"{txt_file.stem}.torch"
            
            if not torch_file.exists():
                # No matching done for this query
                inliers[query_idx] = 0.0
                continue
            
            # Load matching results
            query_results = torch.load(str(torch_file), weights_only=False)
            
            if len(query_results) > 0:
                # Get num_inliers for top-1 match (first result)
                inliers[query_idx] = float(query_results[0].get("num_inliers", 0))
            else:
                inliers[query_idx] = 0.0
                
        except Exception as e:
            print(f"Warning: Error loading inliers for {txt_file.name}: {e}")
            inliers[query_idx] = 0.0
    
    return inliers


def main():
    parser = argparse.ArgumentParser(
        description="Extract features including inliers for top-1 match"
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with prediction .txt files"
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="Directory with .torch files (image matching results)"
    )
    parser.add_argument(
        "--z-data",
        type=str,
        required=True,
        help="Path to z_data.torch file with retrieval distances"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Output .npz file path"
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold for positive matches (meters)"
    )
    
    args = parser.parse_args()
    
    preds_dir = Path(args.preds_dir)
    inliers_dir = Path(args.inliers_dir)
    z_data_path = Path(args.z_data)
    output_path = Path(args.output_path)
    
    # Load retrieval data (from VPR-methods-evaluation)
    print("Loading retrieval data...")
    z_data = torch.load(str(z_data_path), weights_only=False)
    num_queries = int(np.asarray(z_data["predictions"]).shape[0])
    print(f"Found {num_queries} queries")
    
    # Extract retrieval-based features (8 features)
    print("\nExtracting retrieval-based features...")
    retrieval_features = extract_retrieval_features(
        preds_dir=preds_dir,
        z_data=z_data,
        positive_dist_threshold=args.positive_dist_threshold,
        num_geo_nn=10,
    )
    
    # Extract inliers for top-1 match
    print("\nExtracting inliers for top-1 match...")
    inliers_top1 = extract_inliers_from_matching(preds_dir, inliers_dir, num_queries)
    
    print(f"\nInliers statistics:")
    print(f"  Range: {inliers_top1.min():.0f} - {inliers_top1.max():.0f}")
    print(f"  Mean: {inliers_top1.mean():.2f}")
    print(f"  Median: {np.median(inliers_top1):.2f}")
    print(f"  Queries with inliers > 0: {(inliers_top1 > 0).sum()} ({(inliers_top1 > 0).mean() * 100:.1f}%)")
    
    # Combine all features
    print("\nSaving features...")
    np.savez_compressed(
        output_path,
        labels=retrieval_features["labels"],
        top1_distance=retrieval_features["top1_distance"],
        peakiness=retrieval_features["peakiness"],
        sue_score=retrieval_features["sue_score"],
        topk_distance_spread=retrieval_features.get("topk_distance_spread", np.zeros(num_queries)),
        top1_top2_similarity=retrieval_features.get("top1_top2_similarity", np.zeros(num_queries)),
        top1_top3_ratio=retrieval_features.get("top1_top3_ratio", np.zeros(num_queries)),
        top2_top3_ratio=retrieval_features.get("top2_top3_ratio", np.zeros(num_queries)),
        geographic_clustering=retrieval_features.get("geographic_clustering", np.zeros(num_queries)),
        num_inliers_top1=inliers_top1,  # NEW: 9th feature
    )
    
    print(f"Saved features to: {output_path}")
    print(f"  Features: 8 retrieval + 1 inliers = 9 total features")
    print(f"  Total queries: {num_queries}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


