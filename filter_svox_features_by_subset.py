"""
Filter SVOX train features by subset (night, sun, or both).

This script identifies which queries in the feature file correspond to night vs sun
by checking the query image paths in the VPR evaluation logs.
"""

import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm


def find_query_indices_by_subset(preds_dir: Path, subset: str, queries_night_dir: Path = None, queries_sun_dir: Path = None) -> set:
    """
    Find query indices that belong to a specific subset (night or sun).
    
    Args:
        preds_dir: Directory containing prediction .txt files
        subset: 'night' or 'sun'
        queries_night_dir: Path to queries_night directory (for checking file existence)
        queries_sun_dir: Path to queries_sun directory (for checking file existence)
    
    Returns:
        Set of query indices (as integers)
    """
    query_indices = set()
    
    # Get all .txt files (prediction files)
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    print(f"Scanning {len(txt_files)} prediction files to find {subset} queries...")
    
    # Set up directory paths for checking
    if queries_night_dir is None:
        queries_night_dir = Path("data/svox/images/train/queries_night")
    if queries_sun_dir is None:
        queries_sun_dir = Path("data/svox/images/train/queries_sun")
    
    target_dir = queries_night_dir if subset == "night" else queries_sun_dir
    
    # Get all image filenames in the target directory
    target_filenames = {f.name for f in target_dir.glob("*.jpg")} if target_dir.exists() else set()
    print(f"  Found {len(target_filenames)} images in {target_dir}")
    
    for txt_file in tqdm(txt_files, desc=f"Finding {subset} queries"):
        # Read the query image path (second line after "Query path:")
        with open(txt_file, 'r') as f:
            lines = f.readlines()
            if len(lines) >= 2:
                query_path = lines[1].strip()
                # Extract filename from path (handle both forward and backslash)
                # Path format: data/svox/images/test/queries\@0618721.30@5735121.64@201407@09zrs2RO7bc9O3ihU6OQAw@B@.jpg
                query_filename = Path(query_path.replace('\\', '/')).name
                
                # Check if this filename exists in the target directory
                # The queries are in queries/ folder, but we match by filename to queries_sun/queries_night
                if query_filename in target_filenames:
                    query_idx = int(txt_file.stem)
                    query_indices.add(query_idx)
    
    return query_indices


def filter_feature_file(input_path: str, output_path: str, query_indices: set):
    """
    Filter a feature file to keep only queries with indices in query_indices.
    
    Args:
        input_path: Path to input .npz file
        output_path: Path to output .npz file
        query_indices: Set of query indices to keep
    """
    print(f"\nLoading features from: {input_path}")
    data = np.load(input_path)
    
    # Get all keys
    keys = list(data.keys())
    print(f"  Features found: {keys}")
    
    # Get total number of queries
    num_queries = len(data[keys[0]])
    print(f"  Total queries: {num_queries}")
    
    # Create boolean mask for queries to keep
    all_indices = set(range(num_queries))
    keep_mask = np.array([i in query_indices for i in range(num_queries)])
    
    num_kept = keep_mask.sum()
    print(f"  Queries to keep: {num_kept} (indices: {sorted(query_indices)})")
    
    # Filter all features
    filtered_data = {}
    for key in keys:
        filtered_data[key] = data[key][keep_mask]
        print(f"  {key}: {data[key].shape} -> {filtered_data[key].shape}")
    
    # Save filtered features
    print(f"\nSaving filtered features to: {output_path}")
    np.savez_compressed(output_path, **filtered_data)
    print(f"  Saved {num_kept} queries")


def main():
    parser = argparse.ArgumentParser(
        description="Filter SVOX train features by subset (night, sun, or both)"
    )
    parser.add_argument(
        "--input-features",
        type=str,
        required=True,
        help="Path to input feature file (e.g., features_svox_train_improved.npz)"
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Path to VPR prediction directory (contains .txt files with query paths)"
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["night", "sun", "both"],
        required=True,
        help="Subset to filter: 'night', 'sun', or 'both'"
    )
    parser.add_argument(
        "--output-features",
        type=str,
        required=True,
        help="Path to output feature file"
    )
    parser.add_argument(
        "--queries-night-dir",
        type=str,
        default="data/svox/images/train/queries_night",
        help="Path to queries_night directory"
    )
    parser.add_argument(
        "--queries-sun-dir",
        type=str,
        default="data/svox/images/train/queries_sun",
        help="Path to queries_sun directory"
    )
    
    args = parser.parse_args()
    
    preds_dir = Path(args.preds_dir)
    if not preds_dir.exists():
        raise FileNotFoundError(f"Prediction directory not found: {preds_dir}")
    
    # Find query indices for the subset
    if args.subset == "both":
        # Keep all queries (no filtering needed, but we'll verify)
        print("Subset 'both': Keeping all queries")
        # Load to get total count
        data = np.load(args.input_features)
        num_queries = len(data[list(data.keys())[0]])
        query_indices = set(range(num_queries))
    elif args.subset == "night":
        query_indices = find_query_indices_by_subset(
            preds_dir, "night", 
            Path(args.queries_night_dir), 
            Path(args.queries_sun_dir)
        )
    elif args.subset == "sun":
        query_indices = find_query_indices_by_subset(
            preds_dir, "sun",
            Path(args.queries_night_dir),
            Path(args.queries_sun_dir)
        )
    else:
        raise ValueError(f"Unknown subset: {args.subset}")
    
    print(f"\nFound {len(query_indices)} queries for subset '{args.subset}'")
    
    # Filter feature file
    filter_feature_file(args.input_features, args.output_features, query_indices)
    
    print(f"\n{'='*70}")
    print(f"Filtering complete!")
    print(f"  Input: {args.input_features}")
    print(f"  Output: {args.output_features}")
    print(f"  Subset: {args.subset}")
    print(f"  Queries: {len(query_indices)}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()



