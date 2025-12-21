"""
Evaluate adaptive re-ranking vs full re-ranking (Recall@1) on multiple test datasets
for the 3 inliers-based logistic models (night+sun, night only, sun only).

Outputs:
- Markdown table with Adaptive R@1 vs Full Re-ranking R@1 per (dataset, model)
- CSV with the same information

Notes:
- Models are expected to be "hard_score" by default (probs = P(hard)).
- Threshold used is the model's saved validation threshold (optimal_threshold).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch
from glob import glob

from util import get_list_distances_from_preds
from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    feature_file: Path
    preds_dir: Path
    inliers_dir: Path


def _sorted_query_txts(preds_dir: Path) -> list[Path]:
    txts = [Path(p) for p in glob(str(preds_dir / "*.txt"))]
    txts.sort(key=lambda x: int(x.stem))
    return txts


def compute_baseline_r1(
    preds_dir: Path,
    positive_dist_threshold: int = 25,
    num_preds: int = 20,
) -> Optional[float]:
    """
    Retrieval-only baseline Recall@1:
    Uses the original retrieval order (no re-ranking).
    """
    txt_files = _sorted_query_txts(preds_dir)
    total = len(txt_files)
    if total == 0:
        return None
    correct = 0
    for txt in txt_files:
        geo_dists = get_list_distances_from_preds(str(txt))[:num_preds]
        if len(geo_dists) > 0 and geo_dists[0] <= positive_dist_threshold:
            correct += 1
    return (correct / total) * 100.0


def compute_full_reranking_r1(
    preds_dir: Path,
    inliers_dir: Path,
    num_preds: int = 20,
    positive_dist_threshold: int = 25,
) -> tuple[Optional[float], str]:
    """
    Compute full re-ranking R@1. Requires that inliers_dir contains a .torch file per query.

    Returns:
        (r1, note) where r1 is None if full coverage is not available.
    """
    txt_files = _sorted_query_txts(preds_dir)
    total = len(txt_files)
    if total == 0:
        return None, "no queries found"

    torch_files = list(inliers_dir.glob("*.torch"))
    if len(torch_files) != total:
        return None, f"missing torch files ({len(torch_files)}/{total})"

    # Detect how many predictions were actually matched per query (k) by inspecting one file
    try:
        sample_obj = torch.load(str(inliers_dir / f"{txt_files[0].stem}.torch"), weights_only=False)
        matched_k = len(sample_obj) if hasattr(sample_obj, "__len__") else None
    except Exception:
        matched_k = None

    correct = 0
    for txt in txt_files:
        geo_dists = torch.tensor(get_list_distances_from_preds(str(txt)))[:num_preds]
        tfile = inliers_dir / f"{txt.stem}.torch"
        query_results = torch.load(str(tfile), weights_only=False)
        actual_num = min(len(query_results), num_preds, len(geo_dists))
        if actual_num == 0:
            continue
        inliers = torch.tensor([query_results[i]["num_inliers"] for i in range(actual_num)], dtype=torch.float32)
        _, idxs = torch.sort(inliers, descending=True)
        reranked_geo = geo_dists[:actual_num][idxs]
        if len(reranked_geo) > 0 and reranked_geo[0] <= positive_dist_threshold:
            correct += 1

    if matched_k is not None and matched_k != num_preds:
        return (correct / total) * 100.0, f"ok (matched_k={matched_k}, eval_num_preds={num_preds})"
    return (correct / total) * 100.0, "ok"


def compute_adaptive_r1(
    preds_dir: Path,
    inliers_dir: Path,
    is_hard: np.ndarray,
    num_preds: int = 20,
    positive_dist_threshold: int = 25,
) -> float:
    """
    Compute adaptive R@1:
    - For hard queries: re-rank using inliers (.torch)
    - For easy queries: keep retrieval ordering
    If a hard query is missing a .torch file, we fall back to retrieval ordering.
    """
    txt_files = _sorted_query_txts(preds_dir)
    total = len(txt_files)
    if total != len(is_hard):
        raise ValueError(f"Mismatch: {total} txt files vs {len(is_hard)} hard flags")

    correct = 0
    for idx, txt in enumerate(txt_files):
        geo_dists = torch.tensor(get_list_distances_from_preds(str(txt)))[:num_preds]
        tfile = inliers_dir / f"{txt.stem}.torch"

        if bool(is_hard[idx]) and tfile.exists():
            query_results = torch.load(str(tfile), weights_only=False)
            actual_num = min(len(query_results), num_preds, len(geo_dists))
            if actual_num == 0:
                continue
            inliers = torch.tensor([query_results[i]["num_inliers"] for i in range(actual_num)], dtype=torch.float32)
            _, idxs = torch.sort(inliers, descending=True)
            geo = geo_dists[:actual_num][idxs]
        else:
            geo = geo_dists

        if len(geo) > 0 and geo[0] <= positive_dist_threshold:
            correct += 1

    return (correct / total) * 100.0


def predict_is_hard_for_dataset(model_path: Path, feature_path: Path) -> dict:
    """
    Load model + features and return:
      - is_hard array aligned with query indices
      - threshold used
      - target_type
      - threshold_method
    """
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    threshold = float(bundle.get("optimal_threshold", 0.5))
    threshold_method = bundle.get("threshold_method", "f1")
    target_type = bundle.get("target_type", "hard_score")

    feats = load_feature_file(str(feature_path))
    X = build_feature_matrix(feats, feature_names)

    valid_mask = ~np.isnan(X).any(axis=1)
    if (~valid_mask).sum() > 0:
        # If this ever happens, alignment with preds txt indices becomes ambiguous.
        raise ValueError(
            f"NaNs found in {feature_path} ({(~valid_mask).sum()} rows). "
            f"Please regenerate features to avoid NaNs."
        )

    Xs = scaler.transform(X)
    probs = model.predict_proba(Xs)[:, 1]

    if target_type == "hard_score":
        # P(hard) >= threshold => hard
        is_hard = probs >= threshold
    else:
        # P(easy) < threshold => hard
        is_hard = probs < threshold

    return {
        "is_hard": is_hard.astype(bool),
        "threshold": threshold,
        "threshold_method": threshold_method,
        "target_type": target_type,
        "hard_rate": float(is_hard.mean()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Evaluate adaptive re-ranking vs full re-ranking (R@1) on test datasets."
    )
    parser.add_argument("--model-night-sun", type=str, required=True)
    parser.add_argument("--model-night-only", type=str, required=True)
    parser.add_argument("--model-sun-only", type=str, required=True)

    parser.add_argument("--sf-xs-feature", type=str, default="data/features_and_predictions/features_sf_xs_test_with_inliers.npz")
    parser.add_argument("--sf-xs-preds", type=str, default="logs/log_sf_xs_test/2025-12-17_21-14-10/preds")
    parser.add_argument("--sf-xs-inliers", type=str, default="logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg")

    parser.add_argument("--tokyo-feature", type=str, default="data/features_and_predictions/features_tokyo_xs_test_with_inliers.npz")
    parser.add_argument("--tokyo-preds", type=str, default="log_tokyo_xs_test/2025-12-18_14-43-02/preds")
    parser.add_argument("--tokyo-inliers", type=str, default="log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg")

    parser.add_argument("--svox-night-feature", type=str, default="data/features_and_predictions/features_svox_night_test_with_inliers.npz")
    parser.add_argument("--svox-night-preds", type=str, default="logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds")
    parser.add_argument("--svox-night-inliers", type=str, default="logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds_superpoint-lg_top2_256")

    parser.add_argument("--svox-sun-feature", type=str, default="data/features_and_predictions/features_svox_sun_test_with_inliers.npz")
    parser.add_argument("--svox-sun-preds", type=str, default="logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds")
    parser.add_argument("--svox-sun-inliers", type=str, default="logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds_superpoint-lg_top2_256")

    parser.add_argument("--output-dir", type=str, default="output_stages/models_test_comparison_inliers_f1")
    parser.add_argument("--num-preds", type=int, default=20)
    parser.add_argument("--positive-dist-threshold", type=int, default=25)

    args = parser.parse_args()

    models = [
        ("Night + Sun", Path(args.model_night_sun)),
        ("Night Only", Path(args.model_night_only)),
        ("Sun Only", Path(args.model_sun_only)),
    ]

    datasets = [
        DatasetCfg("SF-XS test", Path(args.sf_xs_feature), Path(args.sf_xs_preds), Path(args.sf_xs_inliers)),
        DatasetCfg("Tokyo-XS test", Path(args.tokyo_feature), Path(args.tokyo_preds), Path(args.tokyo_inliers)),
        DatasetCfg("SVOX Night test", Path(args.svox_night_feature), Path(args.svox_night_preds), Path(args.svox_night_inliers)),
        DatasetCfg("SVOX Sun test", Path(args.svox_sun_feature), Path(args.svox_sun_preds), Path(args.svox_sun_inliers)),
    ]

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for d in datasets:
        # Skip datasets not ready yet (e.g., SVOX night/sun while matching is still running)
        if not d.feature_file.exists() or not d.preds_dir.exists():
            continue

        baseline_r1 = compute_baseline_r1(
            preds_dir=d.preds_dir,
            num_preds=args.num_preds,
            positive_dist_threshold=args.positive_dist_threshold,
        )

        full_r1, full_note = compute_full_reranking_r1(
            preds_dir=d.preds_dir,
            inliers_dir=d.inliers_dir,
            num_preds=args.num_preds,
            positive_dist_threshold=args.positive_dist_threshold,
        )

        for model_name, model_path in models:
            pred = predict_is_hard_for_dataset(model_path, d.feature_file)
            adaptive_r1 = compute_adaptive_r1(
                preds_dir=d.preds_dir,
                inliers_dir=d.inliers_dir,
                is_hard=pred["is_hard"],
                num_preds=args.num_preds,
                positive_dist_threshold=args.positive_dist_threshold,
            )

            ratio = None
            delta = None
            if full_r1 is not None and full_r1 > 0:
                ratio = (adaptive_r1 / full_r1) * 100.0
                delta = adaptive_r1 - full_r1

            rows.append(
                {
                    "dataset": d.name,
                    "model": model_name,
                    "threshold_used": pred["threshold"],
                    "threshold_method": pred["threshold_method"],
                    "target_type": pred["target_type"],
                    "hard_rate_pct": pred["hard_rate"] * 100.0,
                    "baseline_r1": baseline_r1,
                    "adaptive_r1": adaptive_r1,
                    "full_r1": full_r1,
                    "full_note": full_note,
                    "performance_ratio_pct": ratio,
                    "delta_vs_full": delta,
                }
            )

    if len(rows) == 0:
        raise SystemExit("No datasets ready for evaluation (missing feature files or preds dirs).")

    # Write CSV
    csv_path = out_dir / "adaptive_vs_full_r1_results.csv"
    import csv
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Write Markdown
    md_path = out_dir / "adaptive_vs_full_r1_results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# Adaptive vs Full Re-ranking (Recall@1) - Test Sets\n\n")
        f.write("Each row uses the model's **saved validation threshold** (`optimal_threshold`).\n\n")
        f.write("| Dataset | Model | Threshold Used | Hard Queries Detected | Baseline R@1 | Adaptive R@1 | Full Re-ranking R@1 | Ratio | Note |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---|\n")
        for r in rows:
            thr = f"{r['threshold_used']:.3f}"
            hard_pct = f"{r['hard_rate_pct']:.1f}%"
            base = f"{r['baseline_r1']:.2f}%" if r["baseline_r1"] is not None else "-"
            ad = f"{r['adaptive_r1']:.2f}%"
            if r["full_r1"] is None:
                full = "-"
                ratio = "-"
            else:
                full = f"{r['full_r1']:.2f}%"
                ratio = f"{r['performance_ratio_pct']:.1f}%" if r["performance_ratio_pct"] is not None else "-"
            note = r["full_note"] if r["dataset"].startswith("SVOX") else "ok"
            if r["full_r1"] is None:
                note = f"full rerank unavailable: {r['full_note']}"
            f.write(f"| {r['dataset']} | {r['model']} | {thr} | {hard_pct} | {base} | {ad} | {full} | {ratio} | {note} |\n")

        f.write("\n## Details\n\n")
        f.write("- **Threshold used**: model bundle field `optimal_threshold` (selected on SF-XS val)\n")
        f.write("- **Hard queries detected**: percentage of queries predicted hard (i.e., re-ranked)\n")
        f.write("- **Adaptive R@1**: re-rank only when query is predicted hard\n")
        f.write("- **Full re-ranking R@1**: re-rank all queries (requires a `.torch` file per query)\n")

    print(f"Saved CSV: {csv_path}")
    print(f"Saved MD:  {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


