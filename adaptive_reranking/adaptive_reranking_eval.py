"""
Adaptive re-ranking with *adaptive matching* (Option B).

For each query (test time):
  1) Use existing LoFTR top-1 inliers + trained LR model to decide if the query is EASY or HARD.
  2) EASY  → trust retrieval-only ranking (no extra LoFTR, no re-ranking).
  3) HARD  → on-the-fly run LoFTR for top-K predictions, re-rank by inliers, then evaluate Recall@N.

This script:
  - Never overwrites your existing logs (top-1 inliers are read-only).
  - Computes Recall@N for the adaptive strategy.
  - Reports how many queries are EASY/HARD and how many extra LoFTR pairs were evaluated.
"""

import argparse
import os
import sys
from copy import deepcopy
from glob import glob
from pathlib import Path

import joblib
import numpy as np
import torch
from tqdm import tqdm
import wandb
import time

# Add parent directory to path to import util and setup_temp_dir
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import get_list_distances_from_preds, read_file_preds
from setup_temp_dir import setup_project_temp_directory


def _setup_matching_module():
    """
    Ensure that the local `image-matching-models` package is used,
    just like in `match_queries_preds.py`.
    """
    project_root = Path(__file__).parent
    image_matching_path = str(project_root.joinpath("image-matching-models"))
    if image_matching_path not in sys.path:
        sys.path.insert(0, image_matching_path)

    from matching import get_matcher, available_models  # type: ignore
    from matching.utils import get_default_device  # type: ignore

    return get_matcher, available_models, get_default_device


def parse_arguments():
    get_matcher, available_models, get_default_device = _setup_matching_module()

    parser = argparse.ArgumentParser(
        description="Evaluate adaptive re-ranking with LR model (adaptive matching, Option B)."
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        required=True,
        help="Directory with prediction .txt files (retrieval outputs).",
    )
    parser.add_argument(
        "--top1-inliers-dir",
        type=str,
        required=True,
        help="Directory with .torch files containing LoFTR results for TOP-1 only.",
    )
    parser.add_argument(
        "--lr-model",
        type=str,
        required=True,
        help="Path to trained LR model (.pkl file from tuning).",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions to consider for potential re-ranking (K).",
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
        default=[1, 5, 10, 20],
        help="Values for recall (e.g. recall@1, recall@5).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Probability threshold for easy/hard classification. "
            "If None, uses threshold saved inside the LR model bundle."
        ),
    )
    parser.add_argument(
        "--matcher",
        type=str,
        default="loftr",
        choices=available_models,
        help="Image matcher to use for on-the-fly matching (default: loftr).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=get_default_device(),
        choices=["cpu", "cuda"],
        help="Device for the matcher (cpu or cuda).",
    )
    parser.add_argument(
        "--im-size",
        type=int,
        default=512,
        help="Resize images to im_size x im_size before matching.",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default=None,
        help="If set, log metrics to this Weights & Biases project.",
    )
    parser.add_argument(
        "--wandb-run-name",
        type=str,
        default=None,
        help="Optional W&B run name (if not set, a default name is used).",
    )

    args = parser.parse_args()

    # We need get_matcher for main(); stash it on the args namespace.
    args._get_matcher = get_matcher
    return args


def main(args):
    # Measure wall-clock runtime
    start_time = time.time()

    # Optionally initialize Weights & Biases
    wandb_run = None
    if args.wandb_project is not None:
        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"adaptive_reranking_{Path(args.preds_dir).name}_{args.matcher}"
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config={
                "preds_dir": str(args.preds_dir),
                "top1_inliers_dir": str(args.top1_inliers_dir),
                "lr_model": str(args.lr_model),
                "num_preds": args.num_preds,
                "positive_dist_threshold": args.positive_dist_threshold,
                "matcher": args.matcher,
                "device": args.device,
                "im_size": args.im_size,
            },
        )

    # Set up temporary directory for matchers that need it (no side effects on logs).
    temp_dir = setup_project_temp_directory()
    print(f"Temporary files will be stored in: {temp_dir}")

    # Load LR model
    print(f"\nLoading LR model from {args.lr_model}...")
    model_bundle = joblib.load(args.lr_model)
    scaler = model_bundle["scaler"]
    model = model_bundle["model"]
    model_threshold = model_bundle.get("best_threshold", 0.5)

    # Use provided threshold or model's threshold
    threshold = args.threshold if args.threshold is not None else model_threshold
    print(f"Using LR decision threshold: {threshold:.2f}")

    preds_folder = Path(args.preds_dir)
    inliers_top1_folder = Path(args.top1_inliers_dir)
    num_preds = args.num_preds
    dist_threshold = args.positive_dist_threshold
    recall_values = args.recall_values

    if not preds_folder.exists():
        raise FileNotFoundError(f"Predictions directory does not exist: {preds_folder}")
    if not inliers_top1_folder.exists():
        raise FileNotFoundError(f"Top-1 inliers directory does not exist: {inliers_top1_folder}")

    # Prepare matcher
    print(f"\nInitializing matcher '{args.matcher}' on device '{args.device}'...")
    matcher = args._get_matcher(args.matcher, device=args.device)
    img_size = args.im_size

    txt_files = glob(os.path.join(str(preds_folder), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))

    total_queries = len(txt_files)
    recalls = np.zeros(len(recall_values), dtype=np.float64)
    processed_queries = 0
    easy_queries = 0
    hard_queries = 0
    skipped_files = []

    # Cost metrics (extra LoFTR evaluations beyond the already-computed top-1)
    extra_loftr_pairs = 0  # pairs run inside this script (for hard queries)

    print(f"\nProcessing {total_queries} queries...")
    print("Adaptive strategy:")
    print("  - Always use existing top-1 inliers for LR decision.")
    print("  - EASY  → retrieval-only, no extra LoFTR.")
    print("  - HARD  → run LoFTR for top-K, then re-rank by inliers.\n")

    for txt_file_query in tqdm(txt_files):
        try:
            txt_path = Path(txt_file_query)

            # ----------------------------------------------------------------------------------
            # 1) Read retrieval predictions & distances (CosPlace output)
            # ----------------------------------------------------------------------------------
            geo_dists = torch.tensor(get_list_distances_from_preds(str(txt_path)))[:num_preds]
            if len(geo_dists) == 0:
                skipped_files.append(txt_path.name)
                continue

            # ----------------------------------------------------------------------------------
            # 2) Load existing top-1 inliers for LR decision
            # ----------------------------------------------------------------------------------
            torch_filename = txt_path.with_suffix(".torch").name
            torch_file_query = inliers_top1_folder / torch_filename

            if not torch_file_query.exists():
                skipped_files.append(txt_path.name)
                continue

            top1_results = torch.load(str(torch_file_query), weights_only=False)
            if len(top1_results) == 0:
                skipped_files.append(txt_path.name)
                continue

            # We assume index 0 corresponds to the retrieval top-1 prediction
            inliers_top1 = float(top1_results[0]["num_inliers"])
            inliers_array = np.array([[inliers_top1]], dtype=np.float32)
            inliers_scaled = scaler.transform(inliers_array)
            prob_correct = float(model.predict_proba(inliers_scaled)[0, 1])

            # ----------------------------------------------------------------------------------
            # 3) EASY vs HARD decision
            # ----------------------------------------------------------------------------------
            is_easy = prob_correct >= threshold

            if is_easy:
                # EASY: retrieval-only ranking
                easy_queries += 1
                ranking_dists = geo_dists

            else:
                # HARD: on-the-fly LoFTR for top-K predictions, then re-rank
                hard_queries += 1

                # Read query + prediction paths
                query_path, pred_paths = read_file_preds(str(txt_path))
                pred_paths = pred_paths[:num_preds]

                if len(pred_paths) == 0:
                    skipped_files.append(txt_path.name)
                    continue

                # Ensure we don't exceed available distances
                actual_num_preds = min(len(pred_paths), len(geo_dists))
                if actual_num_preds == 0:
                    skipped_files.append(txt_path.name)
                    continue

                # Prepare inlier array; index 0 uses existing top-1 inliers
                query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
                query_db_inliers[0] = inliers_top1

                # Load query image once
                img_q = matcher.load_image(query_path, resize=img_size)

                # Run LoFTR for remaining predictions [1:K]
                for idx in range(1, actual_num_preds):
                    img_db = matcher.load_image(pred_paths[idx], resize=img_size)
                    result = matcher(deepcopy(img_q), img_db)
                    # We expect matcher to return a dict with 'num_inliers' (same as in match_queries_preds)
                    num_inliers = float(result.get("num_inliers", 0.0))
                    query_db_inliers[idx] = num_inliers
                    extra_loftr_pairs += 1

                # Sort by inliers (descending) and reorder distances
                _, indices = torch.sort(query_db_inliers, descending=True)
                ranking_dists = geo_dists[indices]

            # ----------------------------------------------------------------------------------
            # 4) Evaluate Recall@N on the chosen ranking
            # ----------------------------------------------------------------------------------
            for i, n in enumerate(recall_values):
                if n <= len(ranking_dists) and torch.any(ranking_dists[:n] <= dist_threshold):
                    recalls[i:] += 1
                    break

            processed_queries += 1

        except Exception as e:
            print(f"Warning: error processing {txt_path.name}: {e}")
            skipped_files.append(txt_path.name)
            continue

    # --------------------------------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------------------------------
    if processed_queries > 0:
        recalls = recalls / processed_queries * 100.0
        total_time_sec = time.time() - start_time
        avg_time_per_query = total_time_sec / processed_queries

        print(f"\n{'=' * 80}")
        print("ADAPTIVE RE-RANKING WITH ADAPTIVE MATCHING (OPTION B)")
        print(f"{'=' * 80}")
        print(f"Processed queries: {processed_queries}/{total_queries}")
        print(
            f"Easy queries (retrieval-only): {easy_queries} "
            f"({easy_queries / processed_queries * 100.0:.1f}%)"
        )
        print(
            f"Hard queries (full LoFTR + re-ranking): {hard_queries} "
            f"({hard_queries / processed_queries * 100.0:.1f}%)"
        )
        print("\nRecall@N (adaptive strategy):")
        for val, rec in zip(recall_values, recalls):
            print(f"  Recall@{val}: {rec:.2f}%")

        # Cost metrics
        avg_extra_pairs = extra_loftr_pairs / processed_queries
        # If we include the already-computed top-1 per query, this is the total LoFTR cost:
        avg_total_pairs_incl_top1 = avg_extra_pairs + 1.0

        print(f"\nTiming:")
        print(f"  Total runtime: {total_time_sec:.1f} s")
        print(f"  Avg runtime per processed query: {avg_time_per_query:.4f} s/query")

        print(f"\nCost metrics (LoFTR pairs):")
        print(f"  Extra pairs run in this script (beyond precomputed top-1): {extra_loftr_pairs}")
        print(f"  Avg extra pairs per processed query: {avg_extra_pairs:.2f}")
        print(
            "  Approx. avg TOTAL pairs per query including top-1: "
            f"{avg_total_pairs_incl_top1:.2f} (baseline full re-ranking would be ≈ {args.num_preds})"
        )

        # Log to W&B if enabled
        if wandb_run is not None:
            metrics = {
                "processed_queries": processed_queries,
                "total_queries": total_queries,
                "easy_queries": easy_queries,
                "hard_queries": hard_queries,
                "pct_easy": easy_queries / processed_queries * 100.0,
                "pct_hard": hard_queries / processed_queries * 100.0,
                "avg_extra_loftr_pairs_per_query": avg_extra_pairs,
                "avg_total_loftr_pairs_incl_top1": avg_total_pairs_incl_top1,
                "extra_loftr_pairs_total": extra_loftr_pairs,
                "total_runtime_sec": total_time_sec,
                "avg_runtime_per_query_sec": avg_time_per_query,
            }
            for val, rec in zip(recall_values, recalls):
                metrics[f"recall@{val}"] = float(rec)
            wandb.log(metrics)

        if skipped_files:
            print(f"\nWarning: skipped {len(skipped_files)} queries due to missing/invalid files.")
            if len(skipped_files) <= 10:
                print("  Skipped:", ", ".join(skipped_files))
            else:
                print("  First 10 skipped:", ", ".join(skipped_files[:10]))

        print(f"{'=' * 80}\n")

        if wandb_run is not None:
            wandb.finish()

    else:
        print("\nERROR: No queries were successfully processed.")
        print("Please check that:")
        print("  1. CosPlace prediction .txt files exist in --preds-dir;")
        print("  2. Top-1 LoFTR .torch files exist in --top1-inliers-dir;")
        print("  3. File names match between .txt and .torch;")
        print("  4. The matcher can load images from the paths in the .txt files.")

        if wandb_run is not None:
            wandb.finish()


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
