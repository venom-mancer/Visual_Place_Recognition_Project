"""
Add inliers (top-1) as 9th feature to existing feature files.

This script:
1. Loads existing feature files (8 features)
2. Extracts inliers from image matching results
3. Adds inliers as 9th feature
4. Saves updated feature files
"""

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm


def extract_inliers_from_matching(preds_dir: Path, inliers_dir: Path, num_queries: int) -> np.ndarray:
    """Extract num_inliers for top-1 match from image matching results."""
    inliers = np.zeros(num_queries, dtype=np.float32)
    
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    print(f"Extracting inliers from {len(txt_files)} queries...")
    
    for txt_file in tqdm(txt_files, desc="Loading inliers"):
        try:
            query_idx = int(txt_file.stem)
            
            # Skip if query index is out of bounds
            if query_idx >= num_queries:
                continue
            
            torch_file = inliers_dir / f"{txt_file.stem}.torch"
            
            if not torch_file.exists():
                inliers[query_idx] = 0.0
                continue
            
            query_results = torch.load(str(torch_file), weights_only=False)
            
            if len(query_results) > 0:
                inliers[query_idx] = float(query_results[0].get("num_inliers", 0))
            else:
                inliers[query_idx] = 0.0
                
        except Exception as e:
            print(f"Warning: Error loading inliers for {txt_file.name}: {e}")
            if query_idx < num_queries:
                inliers[query_idx] = 0.0
    
    return inliers


def main():
    parser = argparse.ArgumentParser(
        description="Add inliers as 9th feature to existing feature files"
    )
    parser.add_argument(
        "--feature-file",
        type=str,
        required=True,
        help="Existing feature .npz file (8 features)"
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
        "--output-file",
        type=str,
        required=True,
        help="Output .npz file path (9 features)"
    )
    
    args = parser.parse_args()
    
    # Load existing features
    print(f"Loading features from: {args.feature_file}")
    data = np.load(args.feature_file)
    num_queries = len(data["labels"])
    print(f"  Found {num_queries} queries with {len([k for k in data.keys() if k != 'labels'])} features")
    
    # Extract inliers
    preds_dir = Path(args.preds_dir)
    inliers_dir = Path(args.inliers_dir)
    inliers_top1 = extract_inliers_from_matching(preds_dir, inliers_dir, num_queries)
    
    print(f"\nInliers statistics:")
    print(f"  Range: {inliers_top1.min():.0f} - {inliers_top1.max():.0f}")
    print(f"  Mean: {inliers_top1.mean():.2f}")
    print(f"  Median: {np.median(inliers_top1):.2f}")
    print(f"  Queries with inliers > 0: {(inliers_top1 > 0).sum()} ({(inliers_top1 > 0).mean() * 100:.1f}%)")
    
    # Save updated features
    print(f"\nSaving features with inliers to: {args.output_file}")
    save_dict = {key: data[key] for key in data.keys()}
    save_dict["num_inliers_top1"] = inliers_top1
    
    np.savez_compressed(args.output_file, **save_dict)
    
    print(f"  Saved {len(save_dict) - 1} features (8 retrieval + 1 inliers = 9 total)")
    print(f"  Total queries: {num_queries}")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())

