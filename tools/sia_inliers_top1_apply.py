"""
Apply a trained SIA-style inliers_top1 gate (tools/sia_inliers_top1_train.py output)
to a target dataset split.

Outputs:
  - .npz with probs/is_easy/is_hard/hard_query_indices
  - .txt hard query list (query ids)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import torch


def inliers_top1_from_torch(torch_path: Path) -> int:
    data = torch.load(str(torch_path), weights_only=False)
    if not data:
        return 0
    return int(data[0].get("num_inliers", 0))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str, help="Path to trained gate .pkl")
    ap.add_argument("--preds-dir", required=True, type=str)
    ap.add_argument("--inliers-dir", required=True, type=str, help="Directory with .torch (top-20 or top-1 ok)")
    ap.add_argument("--out-npz", required=True, type=str)
    ap.add_argument("--out-hard-txt", required=True, type=str)
    args = ap.parse_args()

    bundle = joblib.load(Path(args.model))
    clf = bundle["model"]
    threshold_easy = float(bundle["optimal_threshold"])

    preds_dir = Path(args.preds_dir)
    inliers_dir = Path(args.inliers_dir)

    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not txts:
        raise FileNotFoundError(f"No preds .txt files found in {preds_dir}")

    sample = next(iter(inliers_dir.glob("*.torch")), None)
    if sample is None:
        raise FileNotFoundError(f"No .torch files found in {inliers_dir}")
    pad = len(sample.stem)

    X = np.zeros((len(txts), 1), dtype=np.float32)
    for i, pred_txt in enumerate(txts):
        qid = int(pred_txt.stem)
        torch_path = inliers_dir / f"{str(qid).zfill(pad)}.torch"
        X[i, 0] = float(inliers_top1_from_torch(torch_path))

    probs_easy = clf.predict_proba(X)[:, 1].astype("float32")
    is_easy = probs_easy >= threshold_easy
    is_hard = ~is_easy
    hard_ids = np.where(is_hard)[0].astype("int32")

    out_npz = Path(args.out_npz)
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_npz,
        probs=probs_easy,
        is_easy=is_easy,
        is_hard=is_hard,
        hard_query_indices=hard_ids,
        optimal_threshold=np.float32(threshold_easy),
        feature_name="inliers_top1",
        target_type="easy_score",
    )

    out_txt = Path(args.out_hard_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(str(int(i)) for i in hard_ids) + ("\n" if len(hard_ids) else ""), encoding="utf-8")

    print("Saved:", out_npz)
    print("Saved hard queries:", out_txt, "(count:", int(is_hard.sum()), ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


