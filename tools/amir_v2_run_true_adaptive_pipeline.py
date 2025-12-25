"""
Run the *true* Amir_V2 adaptive pipeline for a single dataset split:

1) Retrieval already exists (preds_dir with *.txt)
2) Compute top-1 matching for ALL queries -> inliers_top1_dir (num_preds=1)
3) Apply Amir_V2 inliers_top1 gate -> hard query list + gate npz
4) Compute top-K matching ONLY for hard queries -> adaptive_inliers_dir (creates empty .torch for easy)
5) Evaluate adaptive reranking (Recall@1/5/10/20)

This matches the intended adaptive method:
  cheap gating (top-1) + expensive matching only for predicted hard queries.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch


def count_torch(dirp: Path) -> int:
    return len(list(dirp.glob("*.torch"))) if dirp.exists() else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-dir", required=True, type=str)
    ap.add_argument("--gate-model", required=True, type=str, help="trained_models/amir_v2_gate_*.pkl")
    ap.add_argument("--out-root", required=True, type=str, help="where to write outputs")
    ap.add_argument("--matcher", type=str, default="superpoint-lg")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--im-size", type=int, default=512)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--positive-dist-threshold", type=int, default=25)
    ap.add_argument("--recall-values", type=int, nargs="+", default=[1, 5, 10, 20])
    args = ap.parse_args()

    preds_dir = Path(args.preds_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # (2) Top-1 matching for all queries
    inliers_top1_dir = out_root / f"preds_{args.matcher}_top1"
    cmd_top1 = [
        sys.executable,
        "match_queries_preds.py",
        "--preds-dir",
        str(preds_dir),
        "--out-dir",
        str(inliers_top1_dir),
        "--matcher",
        args.matcher,
        "--device",
        args.device,
        "--im-size",
        str(args.im_size),
        "--num-preds",
        "1",
    ]
    print("\n=== Step 2: Top-1 matching for all queries ===")
    print("Out:", inliers_top1_dir)
    subprocess.run(cmd_top1, check=True)
    print("Top-1 .torch files:", count_torch(inliers_top1_dir))

    # Build a tiny features npz containing only num_inliers_top1 (for gate apply)
    sample = next(iter(inliers_top1_dir.glob("*.torch")))
    pad = len(sample.stem)
    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    n = len(txts)
    inl = np.zeros((n,), dtype="float32")
    for i, pred_txt in enumerate(txts):
        qid = int(pred_txt.stem)
        tpath = inliers_top1_dir / f"{str(qid).zfill(pad)}.torch"
        data = torch.load(str(tpath), weights_only=False)
        inl[i] = float(data[0].get("num_inliers", 0)) if data else 0.0

    features_npz = out_root / "features_inliers_top1_only.npz"
    np.savez_compressed(features_npz, num_inliers_top1=inl)

    # (3) Apply gate -> decisions + hard list
    gate_npz = out_root / "amir_v2_gate_output.npz"
    hard_txt = out_root / "hard_queries.txt"
    cmd_apply = [
        sys.executable,
        "tools/amir_v2_inliers_top1_apply.py",
        "--model",
        args.gate_model,
        "--features",
        str(features_npz),
        "--out-npz",
        str(gate_npz),
        "--out-hard-txt",
        str(hard_txt),
    ]
    print("\n=== Step 3: Apply Amir_V2 gate ===")
    subprocess.run(cmd_apply, check=True)

    # (4) Top-K matching only for hard queries
    adaptive_inliers_dir = out_root / f"preds_{args.matcher}_adaptive_top{args.topk}"
    cmd_topk = [
        sys.executable,
        "match_queries_preds_adaptive.py",
        "--preds-dir",
        str(preds_dir),
        "--hard-queries-list",
        str(hard_txt),
        "--out-dir",
        str(adaptive_inliers_dir),
        "--matcher",
        args.matcher,
        "--device",
        args.device,
        "--im-size",
        str(args.im_size),
        "--num-preds",
        str(args.topk),
    ]
    print("\n=== Step 4: Top-K matching only for hard queries ===")
    subprocess.run(cmd_topk, check=True)
    print("Adaptive .torch files:", count_torch(adaptive_inliers_dir))

    # (5) Evaluate adaptive reranking
    cmd_eval = [
        sys.executable,
        "-m",
        "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir",
        str(preds_dir),
        "--inliers-dir",
        str(adaptive_inliers_dir),
        "--logreg-output",
        str(gate_npz),
        "--num-preds",
        str(args.topk),
        "--positive-dist-threshold",
        str(args.positive_dist_threshold),
        "--recall-values",
        *[str(v) for v in args.recall_values],
    ]
    print("\n=== Step 5: Evaluate adaptive reranking ===")
    subprocess.run(cmd_eval, check=True)

    print("\nDone. Outputs in:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


