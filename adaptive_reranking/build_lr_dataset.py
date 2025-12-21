import argparse
import csv
import sys
from pathlib import Path

import torch

# Add parent directory to path to import util
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import read_file_preds, get_list_distances_from_preds


def build_rows(
    preds_dir: str,
    inliers_dir: str,
    dataset_name: str,
    vpr_method: str,
    matcher_method: str,
):
    """
    Build LR rows for a single (VPR, matcher, dataset) combination.

    For each query:
      - feature: inliers_top1  (matcher inliers between query and retrieval top-1)
      - label : is_top1_correct (1 if top-1 is in the positives list, else 0)
    """
    preds_path = Path(preds_dir)
    inliers_path = Path(inliers_dir)

    if not preds_path.exists():
        raise FileNotFoundError(f"Preds directory does not exist: {preds_path}")
    if not inliers_path.exists():
        raise FileNotFoundError(f"Inliers directory does not exist: {inliers_path}")

    rows = []
    txt_files = sorted(preds_path.glob("*.txt"), key=lambda p: int(p.stem))

    for txt_file in txt_files:
        query_id = txt_file.stem
        torch_file = inliers_path / f"{query_id}.torch"
        if not torch_file.exists():
            # Skip queries without matching inliers results
            continue

        # Read query path and predictions (top-K paths, retrieval order)
        query_path, pred_paths = read_file_preds(str(txt_file))
        query_path = query_path.strip()
        pred_paths = [p.strip() for p in pred_paths]
        if len(pred_paths) == 0:
            continue
        top1_path = pred_paths[0]

        # Load matcher results and take inliers for top-1 only
        results = torch.load(torch_file, weights_only=False)
        if len(results) == 0:
            continue
        inliers_top1 = int(results[0]["num_inliers"])

        # Label: 1 if top-1 distance <= 25m (consistent with evaluation metric)
        # Use distance-based computation instead of path matching for consistency
        try:
            geo_dists = get_list_distances_from_preds(str(txt_file))
            if len(geo_dists) > 0:
                top1_distance = geo_dists[0]
                is_top1_correct = 1 if top1_distance <= 25.0 else 0
            else:
                # Fallback to path matching if distance computation fails
                with open(txt_file, "r") as f:
                    lines = [line.strip() for line in f.readlines()]
                positives = []
                if "Positives paths:" in lines:
                    start_idx = lines.index("Positives paths:") + 1
                    positives = [l for l in lines[start_idx:] if l]
                is_top1_correct = 1 if top1_path in positives else 0
        except Exception:
            # Fallback to path matching if distance computation fails
            with open(txt_file, "r") as f:
                lines = [line.strip() for line in f.readlines()]
            positives = []
            if "Positives paths:" in lines:
                start_idx = lines.index("Positives paths:") + 1
                positives = [l for l in lines[start_idx:] if l]
            is_top1_correct = 1 if top1_path in positives else 0

        rows.append(
            {
                "query_id": query_id,
                "dataset_name": dataset_name,
                "vpr_method": vpr_method,
                "matcher_method": matcher_method,
                "inliers_top1": inliers_top1,
                "is_top1_correct": is_top1_correct,
                "query_path": query_path,
                "top1_path": top1_path,
            }
        )

    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Build LR training dataset from VPR predictions and matcher inliers."
    )
    parser.add_argument(
        "--preds-dirs",
        nargs="+",
        required=True,
        help="One or more preds folders (retrieval outputs).",
    )
    parser.add_argument(
        "--inliers-dirs",
        nargs="+",
        required=True,
        help="One or more inliers folders (e.g., preds_loftr, preds_superpoint-lg).",
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=True,
        help="Dataset names for each pair of dirs (same length as preds-dirs).",
    )
    parser.add_argument("--out-csv", required=True, help="Output CSV path.")
    parser.add_argument("--vpr-method", required=True, help="Name of the VPR model, e.g. cosplace, mixvpr.")
    parser.add_argument(
        "--matcher-method",
        required=True,
        help="Name of the matcher, e.g. loftr, superpoint-lg.",
    )

    args = parser.parse_args()

    if not (
        len(args.preds_dirs) == len(args.inliers_dirs) == len(args.dataset_names)
    ):
        raise ValueError(
            "preds-dirs, inliers-dirs, and dataset-names must all have the same length."
        )

    all_rows = []
    for preds_dir, inliers_dir, ds_name in zip(
        args.preds_dirs, args.inliers_dirs, args.dataset_names
    ):
        all_rows.extend(
            build_rows(
                preds_dir=preds_dir,
                inliers_dir=inliers_dir,
                dataset_name=ds_name,
                vpr_method=args.vpr_method,
                matcher_method=args.matcher_method,
            )
        )

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "query_id",
                "dataset_name",
                "vpr_method",
                "matcher_method",
                "inliers_top1",
                "is_top1_correct",
                "query_path",
                "top1_path",
            ],
        )
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"Wrote {len(all_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()


