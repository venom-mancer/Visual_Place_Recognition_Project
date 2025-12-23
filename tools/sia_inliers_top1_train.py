"""
SIA-style gate training for adaptive reranking using only:
  feature: inliers_top1 (num inliers between query and retrieval top-1)

Trains a Logistic Regression model to predict:
  P(top1_correct | inliers_top1)

Training split: SVOX train
Validation split: SF-XS val

It selects:
  - C (regularization) by ROC-AUC on validation
  - decision threshold t by maximizing adaptive Recall@1 on validation

Outputs a joblib bundle containing:
  - model, C, threshold, feature_name, metadata
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# Ensure project root is importable so util.py can be imported when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POS_THRESH = 25
NUM_PREDS = 20


def top1_is_correct_from_preds_txt(pred_txt: Path, positive_dist_threshold: float) -> int:
    """Return 1 if top1 is correct (distance <= threshold) else 0."""
    from util import get_list_distances_from_preds

    dists = get_list_distances_from_preds(str(pred_txt))
    if not dists:
        return 0
    return int(dists[0] <= positive_dist_threshold)


def inliers_top1_from_torch(torch_path: Path) -> int:
    """Extract num_inliers for the retrieval top-1 (first element) from a .torch file."""
    data = torch.load(str(torch_path), weights_only=False)
    if not data:
        return 0
    # first candidate corresponds to retrieval top-1
    return int(data[0].get("num_inliers", 0))


@dataclass(frozen=True)
class Split:
    name: str
    preds_dir: Path
    inliers_dir: Path


def load_xy(split: Split, positive_dist_threshold: float) -> tuple[np.ndarray, np.ndarray]:
    txts = sorted(split.preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    if not txts:
        raise FileNotFoundError(f"No preds .txt files found in {split.preds_dir}")

    # detect padding from inliers files
    sample = next(iter(split.inliers_dir.glob("*.torch")), None)
    if sample is None:
        raise FileNotFoundError(f"No .torch files found in {split.inliers_dir}")
    pad = len(sample.stem)

    X = np.zeros((len(txts), 1), dtype=np.float32)
    y = np.zeros((len(txts),), dtype=np.int32)

    for i, pred_txt in enumerate(txts):
        qid = int(pred_txt.stem)
        stem = str(qid).zfill(pad)
        torch_path = split.inliers_dir / f"{stem}.torch"
        X[i, 0] = float(inliers_top1_from_torch(torch_path))
        y[i] = top1_is_correct_from_preds_txt(pred_txt, positive_dist_threshold)

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
            # retrieval-only
            top1_dist = geo[0]
        else:
            torch_path = inliers_dir / f"{str(qid).zfill(pad)}.torch"
            matches = torch.load(str(torch_path), weights_only=False)
            k = min(len(matches), geo.size)
            if k == 0:
                top1_dist = geo[0]
            else:
                inl = np.array([m.get("num_inliers", 0) for m in matches[:k]], dtype=np.float32)
                order = np.argsort(-inl, kind="stable")
                top1_dist = geo[:k][order][0]

        if top1_dist <= positive_dist_threshold:
            ok += 1

    return ok / len(txts) * 100.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-preds", required=True, type=str)
    ap.add_argument("--train-inliers", required=True, type=str)
    ap.add_argument("--val-preds", required=True, type=str)
    ap.add_argument("--val-inliers", required=True, type=str)
    ap.add_argument("--out-model", required=True, type=str)
    ap.add_argument("--positive-dist-threshold", type=float, default=POS_THRESH)
    ap.add_argument("--num-preds", type=int, default=NUM_PREDS)
    args = ap.parse_args()

    train = Split("train", Path(args.train_preds), Path(args.train_inliers))
    val = Split("val", Path(args.val_preds), Path(args.val_inliers))

    X_train, y_train = load_xy(train, args.positive_dist_threshold)
    X_val, y_val = load_xy(val, args.positive_dist_threshold)

    # Tune C by ROC-AUC on validation
    C_values = [0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0]
    best = {"C": None, "auc": -1.0, "model": None}
    for C in C_values:
        clf = LogisticRegression(C=C, solver="lbfgs", max_iter=1000, class_weight="balanced")
        clf.fit(X_train, y_train)
        probs_val_easy = clf.predict_proba(X_val)[:, 1]  # P(top1_correct) == P(easy)
        try:
            auc = float(roc_auc_score(y_val, probs_val_easy))
        except Exception:
            auc = float("nan")
        if np.isfinite(auc) and auc > best["auc"]:
            best = {"C": C, "auc": auc, "model": clf}

    clf = best["model"]
    probs_val_easy = clf.predict_proba(X_val)[:, 1]

    # Choose threshold t by maximizing adaptive R@1 on validation
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t = 0.5
    best_r1 = -1.0
    for t in thresholds:
        r1 = adaptive_r1_from_gate(
            preds_dir=val.preds_dir,
            inliers_dir=val.inliers_dir,
            probs_easy=probs_val_easy,
            threshold_easy=float(t),
            positive_dist_threshold=args.positive_dist_threshold,
            num_preds=args.num_preds,
        )
        if r1 > best_r1:
            best_r1 = r1
            best_t = float(t)

    out = {
        "model": clf,
        "feature_name": "inliers_top1",
        "target_type": "easy_score",
        "optimal_C": float(best["C"]),
        "val_roc_auc": float(best["auc"]),
        "optimal_threshold": float(best_t),
        "val_best_adaptive_r1": float(best_r1),
        "meta": {
            "train_preds": str(train.preds_dir),
            "train_inliers": str(train.inliers_dir),
            "val_preds": str(val.preds_dir),
            "val_inliers": str(val.inliers_dir),
            "positive_dist_threshold": float(args.positive_dist_threshold),
            "num_preds": int(args.num_preds),
        },
    }

    out_path = Path(args.out_model)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(out, out_path)

    print("Saved SIA-style inliers_top1 gate to:", out_path)
    print("  C:", best["C"], "val_auc:", best["auc"])
    print("  threshold(P(easy)):", best_t, "val_best_adaptive_r1:", best_r1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


