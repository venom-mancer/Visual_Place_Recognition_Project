"""
Generate a 12-row table (4 datasets x 3 Amir_V2 gates) comparing:
- Baseline retrieval R@1
- Adaptive R@1 (rerank only hard queries; decision from Amir_V2 inliers_top1 gate)
- Full re-ranking R@1 (rerank all queries using full inliers)

This script is intended to be run on Amir_V2 branch where the gate models exist:
  trained_models/amir_v2_gate_{night_sun,night_only,sun_only}.pkl
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np


# Ensure project root is importable for util.py
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POS_THRESH = 25
NUM_PREDS = 20


def parse_r1(text: str) -> float:
    m = re.search(r"R@1:\s*(\d+\.\d+)", text)
    if not m:
        raise RuntimeError("Could not parse R@1 from output.")
    return float(m.group(1))


def baseline_r1(preds_dir: Path) -> float:
    # Prefer info.log if present in parent
    info_log = preds_dir.parent / "info.log"
    if info_log.exists():
        txt = info_log.read_text(encoding="utf-8", errors="ignore")
        m = re.search(r"R@1:\s*(\d+\.\d+)", txt)
        if m:
            return float(m.group(1))

    # Fallback: compute from preds top-1 distance
    from util import get_list_distances_from_preds

    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    ok = 0
    for f in txts:
        dists = get_list_distances_from_preds(str(f))
        if dists and dists[0] <= POS_THRESH:
            ok += 1
    return ok / len(txts) * 100.0


def full_rerank_r1(preds_dir: Path, inliers_dir: Path) -> float:
    cmd = [
        sys.executable,
        "reranking.py",
        "--preds-dir",
        str(preds_dir),
        "--inliers-dir",
        str(inliers_dir),
        "--num-preds",
        str(NUM_PREDS),
        "--positive-dist-threshold",
        str(POS_THRESH),
        "--recall-values",
        "1",
        "--matcher",
        "superpoint-lg",
        "--vpr-method",
        "cosplace",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return parse_r1(out.stdout)


def adaptive_r1(preds_dir: Path, inliers_dir: Path, logreg_npz: Path) -> float:
    cmd = [
        sys.executable,
        "-m",
        "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir",
        str(preds_dir),
        "--inliers-dir",
        str(inliers_dir),
        "--logreg-output",
        str(logreg_npz),
        "--num-preds",
        str(NUM_PREDS),
        "--positive-dist-threshold",
        str(POS_THRESH),
        "--recall-values",
        "1",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return parse_r1(out.stdout)


def apply_gate_to_features(gate_pkl: Path, features_npz: Path, out_npz: Path) -> dict:
    bundle = joblib.load(gate_pkl)
    clf = bundle["model"]
    thr = float(bundle["optimal_threshold"])
    invert = bool(bundle.get("invert_probs", False))

    d = np.load(features_npz)
    X = d["num_inliers_top1"].astype("float32").reshape(-1, 1)
    probs_easy = clf.predict_proba(X)[:, 1].astype("float32")
    if invert:
        probs_easy = (1.0 - probs_easy).astype("float32")
    is_easy = probs_easy >= thr
    is_hard = ~is_easy

    np.savez_compressed(
        out_npz,
        probs=probs_easy,
        is_easy=is_easy,
        is_hard=is_hard,
        hard_query_indices=np.where(is_hard)[0].astype("int32"),
        optimal_threshold=np.float32(thr),
        feature_name="num_inliers_top1",
        target_type="easy_score",
        invert_probs=np.bool_(invert),
    )

    return {
        "threshold": thr,
        "hard_pct": float(is_hard.mean() * 100.0),
        "invert_probs": invert,
    }


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    preds_dir: Path
    inliers_dir: Path
    features_npz: Path


def main() -> int:
    out_dir = Path("temp") / "amir_v2_gate_eval"
    out_dir.mkdir(parents=True, exist_ok=True)

    models = [
        ("Night + Sun", Path("trained_models/amir_v2_gate_night_sun.pkl")),
        ("Night Only", Path("trained_models/amir_v2_gate_night_only.pkl")),
        ("Sun Only", Path("trained_models/amir_v2_gate_sun_only.pkl")),
    ]

    datasets = [
        DatasetCfg(
            "SF-XS test",
            Path("logs/log_sf_xs_test/2025-12-17_21-14-10/preds"),
            Path("logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg"),
            Path("data/features_and_predictions/features_sf_xs_test_with_inliers.npz"),
        ),
        DatasetCfg(
            "Tokyo-XS test",
            Path("log_tokyo_xs_test/2025-12-18_14-43-02/preds"),
            Path("log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg"),
            Path("data/features_and_predictions/features_tokyo_xs_test_with_inliers.npz"),
        ),
        DatasetCfg(
            "SVOX Night test",
            Path("logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds"),
            Path("logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds_superpoint-lg_top20_256"),
            Path("data/features_and_predictions/features_svox_night_test_with_inliers.npz"),
        ),
        DatasetCfg(
            "SVOX Sun test",
            Path("logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds"),
            Path("logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds_superpoint-lg_top20_256"),
            Path("data/features_and_predictions/features_svox_sun_test_with_inliers.npz"),
        ),
    ]

    rows = []
    full_cache = {}

    for ds in datasets:
        base = baseline_r1(ds.preds_dir)
        # cache full rerank per dataset (independent of model)
        if ds.name not in full_cache:
            full_cache[ds.name] = full_rerank_r1(ds.preds_dir, ds.inliers_dir)
        full = full_cache[ds.name]

        for model_name, gate_pkl in models:
            out_npz = out_dir / f"logreg_{ds.name.replace(' ','_').replace('-','_')}_{model_name.replace(' ','_').replace('+','plus')}.npz"
            meta = apply_gate_to_features(gate_pkl, ds.features_npz, out_npz)
            adapt = adaptive_r1(ds.preds_dir, ds.inliers_dir, out_npz)
            ratio = adapt / full * 100.0 if full else float("nan")

            rows.append(
                {
                    "Dataset": ds.name,
                    "Model": model_name,
                    "Threshold Used": meta["threshold"],
                    "Hard Queries Detected": meta["hard_pct"],
                    "Time Savings": 100.0 - meta["hard_pct"],
                    "Baseline R@1": base,
                    "Adaptive R@1": adapt,
                    "Full Re-ranking R@1": full,
                    "Ratio": ratio,
                    "Note": "ok",
                }
            )

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


