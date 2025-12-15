import numpy as np
import os
import argparse
import sys
from glob import glob
from pathlib import Path
import torch

sys.path.append(str(
    Path(__file__).parent.parent
))

from util import get_list_distances_from_preds
from vpr_uncertainty.baselines import compute_sue


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--preds-dir", type=str, help="directory with predictions of a VPR model")
    parser.add_argument("--inliers-dir", type=str, help="directory with image matching results")
    parser.add_argument("--z-data-path", type=str, help="path to the 'z_data' file (output part of VPR evaluation)")
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="distance (in meters) for a prediction to be considered a positive",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="features.npz",
        help="where to save the per-query features and labels",
    )

    return parser.parse_args()


def main(args):
    preds_folder = args.preds_dir
    inliers_folder = Path(args.inliers_dir)
    threshold = args.positive_dist_threshold
    z_data_path = args.z_data_path

    txt_files = glob(os.path.join(preds_folder, "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    dict_results = torch.load(z_data_path, weights_only=False)
    ref_poses = dict_results["database_utms"]
    preds = dict_results["predictions"]
    dists = dict_results["distances"]

    total_queries = len(txt_files)

    # Labels: 1 if Top-1 is within threshold, 0 otherwise
    labels = np.zeros(total_queries, dtype="float32")

    # Features
    num_inliers = np.zeros(total_queries, dtype="float32")
    top1_distance = np.zeros(total_queries, dtype="float32")
    peakiness = np.zeros(total_queries, dtype="float32")

    # For SUE we reuse the existing implementation from baselines.py
    # It returns an AUC-PR when called in eval.py, but we can adapt the
    # logic here to extract per-query SUE scores if needed later.
    # For now, we compute a placeholder vector to keep the interface ready.
    sue_score = np.zeros(total_queries, dtype="float32")

    for itr, txt_file_query in enumerate(txt_files):
        geo_dists = get_list_distances_from_preds(txt_file_query)
        labels[itr] = 1.0 if geo_dists[0] <= threshold else 0.0

        torch_file_query = inliers_folder.joinpath(Path(txt_file_query).name.replace("txt", "torch"))
        query_inliers_results = torch.load(torch_file_query, weights_only=False)
        num_inliers[itr] = query_inliers_results[0]["num_inliers"]

        # distances and peakiness from z_data
        top1_distance[itr] = dists[itr][0]
        if dists.shape[1] > 1:
            peakiness[itr] = dists[itr][0] / (dists[itr][1] + 1e-8)
        else:
            peakiness[itr] = 1.0

    # Pack everything into a single file for later use (training / validation / testing)
    np.savez_compressed(
        args.output_path,
        labels=labels,
        num_inliers=num_inliers,
        top1_distance=top1_distance,
        peakiness=peakiness,
        sue_score=sue_score,
    )

    print(f"Saved features and labels for {total_queries} queries to {args.output_path}")


if __name__ == "__main__":
    args = parse_arguments()
    main(args)


