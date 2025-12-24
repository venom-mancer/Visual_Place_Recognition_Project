"""
Amir_V2 gate training (adaptive reranking) using a single robust feature:
  num_inliers_top1  ("inliers_top1")

We train a Logistic Regression model to predict:
  P(top1_correct | inliers_top1)  == P(easy)

We tune:
  - C by ROC-AUC on validation labels
  - decision threshold t by maximizing *adaptive Recall@1* on validation

This mirrors the "SIA-style" approach, but is branded as Amir_V2.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# Ensure project root importability for util.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_inliers_top1_and_labels(features_npz: Path) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(features_npz)
    if "num_inliers_top1" not in d.files:
        raise KeyError(f"{features_npz} missing num_inliers_top1")
    if "labels" not in d.files:
        raise KeyError(f"{features_npz} missing labels")
    X = d["num_inliers_top1"].astype("float32").reshape(-1, 1)
    y = d["labels"].astype("int32")  # 1=top1 correct (easy), 0=wrong (hard)
    return X, y


def adaptive_r1_from_gate(
    preds_dir: Path,
    inliers_dir: Path,
    probs_easy: np.ndarray,
    threshold_easy: float,
    positive_dist_threshold: float,
    num_preds: int,
) -> float:
    """
    Simulate adaptive reranking:
      easy if P(easy) >= t -> retrieval-only
      hard otherwise -> rerank by inliers (descending)
    Returns Recall@1 in percent.
    """
    from util import get_list_distances_from_preds

    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    sample = next(iter(inliers_dir.glob("*.torch")))
    pad = len(sample.stem)

    ok = 0
    for i, pred_txt in enumerate(txts):
        qid = int(pred_txt.stem)
        geo = np.array(get_list_distances_from_preds(str(pred_txt)), dtype=np.float32)[:num_preds]
        if geo.size == 0:
            continue

        is_easy = probs_easy[i] >= threshold_easy
        if is_easy:
            top1_dist = float(geo[0])
        else:
            torch_path = inliers_dir / f"{str(qid).zfill(pad)}.torch"
            matches = torch.load(str(torch_path), weights_only=False)
            k = min(len(matches), geo.size)
            if k == 0:
                top1_dist = float(geo[0])
            else:
                inl = np.array([m.get("num_inliers", 0) for m in matches[:k]], dtype=np.float32)
                order = np.argsort(-inl, kind="stable")
                top1_dist = float(geo[:k][order][0])

        if top1_dist <= positive_dist_threshold:
            ok += 1

    return ok / len(txts) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-features", required=True, type=str, help="*.npz with labels + num_inliers_top1")
    ap.add_argument("--val-features", required=True, type=str, help="*.npz with labels + num_inliers_top1")
    ap.add_argument("--val-preds", required=True, type=str, help="preds/*.txt for validation split")
    ap.add_argument("--val-inliers", required=True, type=str, help="inliers/*.torch for validation split (top20)")
    ap.add_argument("--out-model", required=True, type=str, help="output .pkl")
    ap.add_argument("--positive-dist-threshold", type=float, default=25.0)
    ap.add_argument("--num-preds", type=int, default=20)
    args = ap.parse_args()

    train_features = Path(args.train_features)
    val_features = Path(args.val_features)
    val_preds = Path(args.val_preds)
    val_inliers = Path(args.val_inliers)

    X_train, y_train = load_inliers_top1_and_labels(train_features)
    X_val, y_val = load_inliers_top1_and_labels(val_features)

    # Tune C by ROC-AUC on validation.
    # If AUC < 0.5, the model is effectively inverted; we handle that by storing an invert flag
    # and using (1 - probs) at apply time.
    C_values = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best_C = 1.0
    best_auc = -1.0
    best_invert = False
    best_model: LogisticRegression | None = None

    for C in C_values:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        probs_val_easy = clf.predict_proba(X_val)[:, 1]
        auc = float(roc_auc_score(y_val, probs_val_easy))
        eff_auc = max(auc, 1.0 - auc)
        invert = auc < 0.5
        if eff_auc > best_auc:
            best_auc = eff_auc
            best_invert = invert
            best_C = float(C)
            best_model = clf

    assert best_model is not None
    probs_val_easy = best_model.predict_proba(X_val)[:, 1]
    if best_invert:
        probs_val_easy = 1.0 - probs_val_easy

    # Choose threshold by maximizing adaptive R@1 on validation
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_r1 = -1.0
    for t in thresholds:
        r1 = adaptive_r1_from_gate(
            preds_dir=val_preds,
            inliers_dir=val_inliers,
            probs_easy=probs_val_easy,
            threshold_easy=float(t),
            positive_dist_threshold=float(args.positive_dist_threshold),
            num_preds=int(args.num_preds),
        )
        if r1 > best_r1:
            best_r1 = r1
            best_t = float(t)

    bundle = {
        "model": best_model,
        "feature_name": "num_inliers_top1",
        "target_type": "easy_score",
        "optimal_C": best_C,
        "val_roc_auc": best_auc,
        "invert_probs": bool(best_invert),
        "optimal_threshold": best_t,
        "val_best_adaptive_r1": best_r1,
        "meta": {
            "train_features": str(train_features),
            "val_features": str(val_features),
            "val_preds": str(val_preds),
            "val_inliers": str(val_inliers),
            "positive_dist_threshold": float(args.positive_dist_threshold),
            "num_preds": int(args.num_preds),
        },
    }

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out_path)

    print("Saved Amir_V2 inliers_top1 gate to:", out_path)
    print("  C:", best_C, "val_auc:", best_auc)
    print("  invert_probs:", bool(best_invert))
    print("  threshold(P(easy)):", best_t, "val_best_adaptive_r1:", best_r1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


