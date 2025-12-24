"""
Apply Amir_V2 inliers_top1 gate to a dataset feature file.

Input:
  - trained gate .pkl from tools/amir_v2_inliers_top1_train.py
  - dataset features .npz with num_inliers_top1

Output:
  - .npz with probs/is_easy/is_hard/hard_query_indices (compatible with stage_5 evaluator)
  - .txt hard query list (query ids)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, type=str)
    ap.add_argument("--features", required=True, type=str)
    ap.add_argument("--out-npz", required=True, type=str)
    ap.add_argument("--out-hard-txt", required=True, type=str)
    args = ap.parse_args()

    bundle = joblib.load(Path(args.model))
    clf = bundle["model"]
    threshold_easy = float(bundle["optimal_threshold"])
    invert = bool(bundle.get("invert_probs", False))

    d = np.load(Path(args.features))
    if "num_inliers_top1" not in d.files:
        raise KeyError(f"{args.features} missing num_inliers_top1")
    X = d["num_inliers_top1"].astype("float32").reshape(-1, 1)

    probs_easy = clf.predict_proba(X)[:, 1].astype("float32")
    if invert:
        probs_easy = (1.0 - probs_easy).astype("float32")
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
        feature_name="num_inliers_top1",
        target_type="easy_score",
        invert_probs=np.bool_(invert),
    )

    out_txt = Path(args.out_hard_txt)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(str(int(i)) for i in hard_ids) + ("\n" if len(hard_ids) else ""), encoding="utf-8")

    print("Saved:", out_npz)
    print("Saved hard queries:", out_txt, "(count:", int(is_hard.sum()), ")")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


