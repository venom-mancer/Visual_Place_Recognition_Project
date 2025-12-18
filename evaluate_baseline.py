"""
Evaluate baseline (retrieval-only) recall without re-ranking.
"""
import argparse
from pathlib import Path
import torch
from glob import glob
import os
from tqdm import tqdm

from util import get_list_distances_from_preds


def main(args):
    preds_folder = Path(args.preds_dir).resolve()
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values
    
    txt_files = glob(os.path.join(str(preds_folder), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    recalls = [0.0] * len(recall_values)
    processed_queries = 0
    
    print(f"Found {total_queries} .txt files to process")
    print("Evaluating baseline (retrieval-only, no re-ranking)")
    
    for txt_file_query in tqdm(txt_files):
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:args.num_preds]
            
            # Compute recall contributions for this query (no re-ranking)
            for i, n in enumerate(recall_values):
                if n <= len(geo_dists) and torch.any(geo_dists[:n] <= threshold):
                    recalls[i:] = [r + 1 for r in recalls[i:]]
                    break
            
            processed_queries += 1
            
        except Exception as e:
            print(f"Warning: Error processing {Path(txt_file_query).name}: {e}")
            continue
    
    if processed_queries > 0:
        recalls = [r / processed_queries * 100.0 for r in recalls]
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
        print(f"\nProcessed {processed_queries} out of {total_queries} queries")
        print("Baseline (retrieval-only) results:")
        print(recalls_str)
    else:
        print(f"\nERROR: No queries were successfully processed out of {total_queries} total queries.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preds-dir", type=str, required=True, help="Directory with prediction .txt files")
    parser.add_argument("--num-preds", type=int, default=100, help="Number of predictions to consider")
    parser.add_argument("--positive-dist-threshold", type=int, default=25, help="Distance threshold in meters")
    parser.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20, 100], help="Recall values")
    
    args = parser.parse_args()
    main(args)


