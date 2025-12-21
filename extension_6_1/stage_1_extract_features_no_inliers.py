"""
Stage 1: Extract retrieval-only features (NO inliers).

This file was referenced throughout the project docs/scripts. It computes the 8 retrieval features
described in `docs/08_VALIDATION_FEATURES.md` from:
- VPR retrieval outputs: `preds/` + `z_data.torch`

Outputs a `.npz` with:
  labels,
  top1_distance, peakiness, sue_score,
  topk_distance_spread, top1_top2_similarity, top1_top3_ratio, top2_top3_ratio,
  geographic_clustering

Notes:
- `z_data.torch` comes from `VPR-methods-evaluation/main.py` and contains:
    - predictions: (N, K) database indices
    - distances:   (N, K) squared L2 distances between normalized descriptors
    - database_utms: (Db,2) UTM coords (meters)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

from util import get_list_distances_from_preds


def _avg_pairwise_distance(points_xy: np.ndarray) -> float:
    """Average pairwise Euclidean distance for shape (M,2)."""
    m = points_xy.shape[0]
    if m <= 1:
        return 0.0
    # Compute condensed pairwise distances efficiently for small M (M<=20)
    dsum = 0.0
    cnt = 0
    for i in range(m):
        diffs = points_xy[i + 1 :] - points_xy[i]
        dsum += float(np.sqrt((diffs * diffs).sum(axis=1)).sum())
        cnt += diffs.shape[0]
    return dsum / max(cnt, 1)


def compute_sue(
    preds: np.ndarray,
    ref_poses: np.ndarray,
    num_NN: int = 10,
) -> np.ndarray:
    """
    Spatial Uncertainty Estimate (SUE) proxy.

    We define SUE as the mean distance of the top-N retrieved database UTMs from their centroid.
    Lower SUE => more spatially clustered (more confident).
    """
    n = preds.shape[0]
    sue = np.zeros(n, dtype=np.float32)
    for i in range(n):
        idxs = preds[i, :num_NN]
        pts = ref_poses[idxs]
        c = pts.mean(axis=0, keepdims=True)
        sue[i] = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean().astype(np.float32)
    return sue


def extract_retrieval_features(
    preds_dir: Path,
    z_data: dict,
    positive_dist_threshold: int = 25,
    num_geo_nn: int = 10,
) -> dict:
    """
    Compute the 8 retrieval features for each query, aligned with query indices in preds_dir.
    """
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    num_queries = len(txt_files)
    if num_queries == 0:
        raise ValueError(f"No .txt files found in {preds_dir}")

    preds = np.asarray(z_data["predictions"])
    dists = np.asarray(z_data["distances"])
    ref_poses = np.asarray(z_data["database_utms"])

    if preds.shape[0] != num_queries or dists.shape[0] != num_queries:
        raise ValueError(
            f"Mismatch between preds_dir queries ({num_queries}) and z_data arrays "
            f"(predictions={preds.shape}, distances={dists.shape})"
        )

    # Labels: Top-1 geographic correctness within threshold
    labels = np.zeros(num_queries, dtype=np.float32)
    for i, txt in enumerate(txt_files):
        geo_dists = get_list_distances_from_preds(str(txt))
        labels[i] = 1.0 if len(geo_dists) > 0 and geo_dists[0] <= positive_dist_threshold else 0.0

    eps = 1e-9
    top1 = dists[:, 0].astype(np.float32)
    top2 = dists[:, 1].astype(np.float32) if dists.shape[1] > 1 else (top1 + 1.0)
    top3 = dists[:, 2].astype(np.float32) if dists.shape[1] > 2 else (top2 + 1.0)

    peakiness = (top1 / (top2 + eps)).astype(np.float32)
    topk_distance_spread = np.var(dists[:, :5], axis=1).astype(np.float32) if dists.shape[1] >= 5 else np.var(dists, axis=1).astype(np.float32)
    top1_top2_similarity = (top2 / (top1 + eps)).astype(np.float32)
    top1_top3_ratio = (top1 / (top3 + eps)).astype(np.float32)
    top2_top3_ratio = (top2 / (top3 + eps)).astype(np.float32)

    sue_score = compute_sue(preds=preds, ref_poses=ref_poses, num_NN=num_geo_nn).astype(np.float32)

    geographic_clustering = np.zeros(num_queries, dtype=np.float32)
    for i in tqdm(range(num_queries), desc="Geographic clustering"):
        idxs = preds[i, :num_geo_nn]
        pts = ref_poses[idxs]
        geographic_clustering[i] = np.float32(_avg_pairwise_distance(pts))

    return {
        "labels": labels,
        "top1_distance": top1,
        "peakiness": peakiness,
        "sue_score": sue_score,
        "topk_distance_spread": topk_distance_spread,
        "top1_top2_similarity": top1_top2_similarity,
        "top1_top3_ratio": top1_top3_ratio,
        "top2_top3_ratio": top2_top3_ratio,
        "geographic_clustering": geographic_clustering,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract 8 retrieval-only features (no inliers)")
    parser.add_argument("--preds-dir", type=str, required=True)
    parser.add_argument("--z-data", type=str, required=True, help="Path to z_data.torch from VPR-methods-evaluation")
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--positive-dist-threshold", type=int, default=25)
    parser.add_argument("--num-geo-nn", type=int, default=10)
    args = parser.parse_args()

    preds_dir = Path(args.preds_dir)
    z_data = torch.load(args.z_data, weights_only=False)
    feats = extract_retrieval_features(
        preds_dir=preds_dir,
        z_data=z_data,
        positive_dist_threshold=args.positive_dist_threshold,
        num_geo_nn=args.num_geo_nn,
    )

    out = Path(args.output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **feats)
    print(f"Saved features to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


