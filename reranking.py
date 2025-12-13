import numpy as np
from tqdm import tqdm
import os, argparse
from glob import glob
from pathlib import Path
import torch
from datetime import datetime
import wandb

from util import get_list_distances_from_preds

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--num-preds", type=int, default=100, help="number of predictions to re-rank")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 100],
        help="values for recall (e.g. recall@1, recall@5)",
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default=None,
        help="name of the matcher used (e.g., superpoint-lg). If not provided, will try to infer from inliers-dir path",
    )
    parser.add_argument(
        "--vpr-method",
        type=str,
        default=None,
        help="VPR method name (e.g., cosplace, netvlad) for wandb tracking",
    )

    return parser.parse_args()

def main(args):
    start_time = datetime.now()
    
    # Resolve paths to absolute paths to avoid path issues
    preds_folder = Path(args.preds_dir).resolve()
    inliers_folder = Path(args.inliers_dir).resolve()
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values
    
    # Infer matcher name from inliers-dir if not provided
    matcher_name = args.matcher
    if matcher_name is None:
        # Try to extract matcher name from directory path (e.g., preds_superpoint-lg -> superpoint-lg)
        dir_name = inliers_folder.name
        if dir_name.startswith('preds_'):
            matcher_name = dir_name.replace('preds_', '')
        else:
            matcher_name = "unknown"
    
    # Check if directories exist
    if not preds_folder.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {preds_folder}")
    if not inliers_folder.exists():
        raise FileNotFoundError(f"Inliers directory does not exist: {inliers_folder}")

    # Initialize wandb
    run_name = f"reranking_{matcher_name}_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    if args.vpr_method:
        run_name = f"{args.vpr_method}_{run_name}"
    
    wandb.init(
        project="visual-place-recognition-reranking",
        name=run_name,
        config={
            "matcher": matcher_name,
            "vpr_method": args.vpr_method,
            "num_preds": num_preds,
            "positive_dist_threshold": threshold,
            "recall_values": recall_values,
            "preds_dir": str(preds_folder),
            "inliers_dir": str(inliers_folder),
        }
    )

    txt_files = glob(str(preds_folder / "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    recalls = np.zeros(len(recall_values))
    processed_queries = 0
    skipped_files = []

    print(f"Found {total_queries} .txt files to process")
    print(f"Inliers directory: {inliers_folder}")
    print(f"Matcher: {matcher_name}")

    for txt_file_query in tqdm(txt_files):
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
            txt_filename = Path(txt_file_query).name
            torch_filename = txt_filename.replace('.txt', '.torch')
            torch_file_query = inliers_folder / torch_filename
            
            # Check if torch file exists before trying to load it
            if not torch_file_query.exists():
                skipped_files.append(txt_filename)
                continue
            
            query_results = torch.load(str(torch_file_query), weights_only=False)
            
            # Handle case where query_results has fewer items than num_preds
            actual_num_preds = min(len(query_results), num_preds, len(geo_dists))
            if actual_num_preds == 0:
                skipped_files.append(txt_filename)
                continue
            
            query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
            for i in range(actual_num_preds):
                query_db_inliers[i] = query_results[i]['num_inliers']
            query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
            geo_dists = geo_dists[indices]
            
            for i, n in enumerate(recall_values):
                if n <= len(geo_dists) and torch.any(geo_dists[:n] <= threshold):
                    recalls[i:] += 1
                    break
            
            processed_queries += 1
            
        except Exception as e:
            print(f"Warning: Error processing {Path(txt_file_query).name}: {e}")
            skipped_files.append(Path(txt_file_query).name)
            continue

    # Report skipped files if any
    if skipped_files:
        print(f"\nWarning: Skipped {len(skipped_files)} files (missing .torch files or errors)")
        if len(skipped_files) <= 10:
            print(f"Skipped files: {', '.join(skipped_files)}")
        else:
            print(f"First 10 skipped files: {', '.join(skipped_files[:10])}...")
            print(f"Total skipped: {len(skipped_files)} files")

    # Calculate recalls based on actually processed queries
    if processed_queries > 0:
        recalls = recalls / processed_queries * 100
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
        print(f"\nProcessed {processed_queries} out of {total_queries} queries")
        print(recalls_str)
        
        # Log metrics to wandb
        metrics = {
            "processed_queries": processed_queries,
            "total_queries": total_queries,
            "skipped_queries": len(skipped_files),
        }
        
        # Add recall metrics
        for val, rec in zip(recall_values, recalls):
            metrics[f"recall@{val}"] = float(rec)
        
        wandb.log(metrics)
        print(f"\nMetrics logged to wandb: {metrics}")
        
    else:
        print(f"\nERROR: No queries were successfully processed out of {total_queries} total queries.")
        print("Please check that:")
        print("  1. The matching step has been run for all queries")
        print("  2. The --inliers-dir contains .torch files")
        print("  3. File names match between .txt and .torch files")
    
    wandb.finish()

if __name__ == "__main__":
    args = parse_arguments()
    main(args)