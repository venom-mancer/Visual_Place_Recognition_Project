import argparse
import os
from glob import glob
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from util import get_list_distances_from_preds


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with prediction .txt files (retrieval outputs, one per query).",
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        required=True,
        help="Directory with .torch files (image matching outputs, one per query).",
    )
    parser.add_argument(
        "--logreg-output",
        type=str,
        required=True,
        help="Path to .npz file produced by apply_logreg.py (contains is_easy / is_hard).",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=100,
        help="Number of predictions to consider for re-ranking.",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance (in meters) for a prediction to be considered a positive.",
    )
    parser.add_argument(
        "--recall-values",
        type=int,
        nargs="+",
        default=[1, 5, 10, 20, 100],
        help="Values for recall (e.g. recall@1, recall@5).",
    )

    return parser.parse_args()


def main(args):
    preds_folder = Path(args.preds_dir).resolve()
    inliers_folder = Path(args.inliers_dir).resolve()
    num_preds = args.num_preds
    threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    # Load logistic regression outputs
    logreg_data = np.load(args.logreg_output)
    is_easy = logreg_data["is_easy"]
    is_hard = logreg_data["is_hard"]

    # Collect and sort prediction files
    txt_files = glob(os.path.join(str(preds_folder), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    if total_queries != is_easy.shape[0]:
        raise ValueError(
            f"Number of .txt files ({total_queries}) does not match size of is_easy ({is_easy.shape[0]}). "
            "Make sure logreg_output was computed on the same preds-dir."
        )

    recalls = np.zeros(len(recall_values), dtype="float32")
    processed_queries = 0
    skipped_files = []

    print(f"Found {total_queries} .txt files to process")
    print(f"Using adaptive strategy with logistic regression decisions from: {args.logreg_output}")

    for idx, txt_file_query in enumerate(tqdm(txt_files)):
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
            txt_filename = Path(txt_file_query).name
            torch_filename = txt_filename.replace(".txt", ".torch")
            torch_file_query = inliers_folder / torch_filename

            # If no inliers file, we cannot apply re-ranking; fall back to retrieval-only
            if not torch_file_query.exists():
                # Even if the query is marked hard, we skip re-ranking if we lack data
                query_geo_dists = geo_dists
            else:
                if is_hard[idx]:
                    # Load inliers and re-rank as in reranking.py
                    query_results = torch.load(str(torch_file_query), weights_only=False)
                    actual_num_preds = min(len(query_results), num_preds, len(geo_dists))
                    if actual_num_preds == 0:
                        skipped_files.append(txt_filename)
                        continue

                    query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
                    for i in range(actual_num_preds):
                        query_db_inliers[i] = query_results[i]["num_inliers"]
                    query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
                    query_geo_dists = geo_dists[:actual_num_preds][indices]
                else:
                    # Easy query: use retrieval-only ordering
                    query_geo_dists = geo_dists

            # Compute recall contributions for this query
            for i, n in enumerate(recall_values):
                if n <= len(query_geo_dists) and torch.any(query_geo_dists[:n] <= threshold):
                    recalls[i:] += 1
                    break

            processed_queries += 1

        except Exception as e:
            print(f"Warning: Error processing {Path(txt_file_query).name}: {e}")
            skipped_files.append(Path(txt_file_query).name)
            continue

    if skipped_files:
        print(f"\nWarning: Skipped {len(skipped_files)} files (missing data or errors)")
        if len(skipped_files) <= 10:
            print(f"Skipped files: {', '.join(skipped_files)}")
        else:
            print(f"First 10 skipped files: {', '.join(skipped_files[:10])}...")

    if processed_queries > 0:
        recalls = recalls / processed_queries * 100.0
        recalls_str = ", ".join([f"R@{val}: {rec:.1f}" for val, rec in zip(recall_values, recalls)])
        print(f"\nProcessed {processed_queries} out of {total_queries} queries")
        print("Adaptive re-ranking results:")
        print(recalls_str)
    else:
        print(f"\nERROR: No queries were successfully processed out of {total_queries} total queries.")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)


