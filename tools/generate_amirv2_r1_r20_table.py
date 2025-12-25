"""
Generate a markdown table comparing Recall@{1,5,10,20} for:
- Baseline (retrieval-only)
- Amir_V2 adaptive (inliers_top1 gate, rerank only hard)
- Full pipeline (rerank all)

Across 4 datasets x 3 models.
Prints a markdown table to stdout.
"""

from __future__ import annotations

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
TOPK = 20
RECALLS = [1, 5, 10, 20]


def parse_recalls(stdout: str) -> dict[int, float]:
    # Accept formats like:
    # "R@1: 83.2, R@5: 87.0, R@10: 88.3, R@20: 89.5"
    out: dict[int, float] = {}
    for n in RECALLS:
        m = re.search(rf"R@{n}:\s*(\d+\.\d+)", stdout)
        if m:
            out[n] = float(m.group(1))
    if len(out) != len(RECALLS):
        raise RuntimeError(f"Could not parse all recalls from output. Parsed={out}")
    return out


def baseline_recalls(preds_dir: Path) -> dict[int, float]:
    from util import get_list_distances_from_preds

    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    hits = {n: 0 for n in RECALLS}
    for f in txts:
        dists = np.array(get_list_distances_from_preds(str(f)), dtype=np.float32)
        for n in RECALLS:
            if dists.size >= n and np.any(dists[:n] <= POS_THRESH):
                hits[n] += 1
    total = len(txts)
    return {n: hits[n] / total * 100.0 for n in RECALLS}


def full_recalls(preds_dir: Path, inliers_dir: Path) -> dict[int, float]:
    cmd = [
        sys.executable,
        "reranking.py",
        "--preds-dir",
        str(preds_dir),
        "--inliers-dir",
        str(inliers_dir),
        "--num-preds",
        str(TOPK),
        "--positive-dist-threshold",
        str(POS_THRESH),
        "--recall-values",
        *[str(r) for r in RECALLS],
        "--matcher",
        "superpoint-lg",
        "--vpr-method",
        "cosplace",
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return parse_recalls(out.stdout)


def adaptive_recalls(preds_dir: Path, inliers_dir: Path, logreg_npz: Path) -> dict[int, float]:
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
        str(TOPK),
        "--positive-dist-threshold",
        str(POS_THRESH),
        "--recall-values",
        *[str(r) for r in RECALLS],
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)
    return parse_recalls(out.stdout)


def apply_gate_to_features(gate_pkl: Path, features_npz: Path, out_npz: Path) -> float:
    """Write a stage_5-compatible npz and return hard%."""
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
    hard_pct = float(is_hard.mean() * 100.0)

    out_npz.parent.mkdir(parents=True, exist_ok=True)
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
    return hard_pct


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    preds_dir: Path
    inliers_dir: Path
    features_npz: Path


def main() -> int:
    out_dir = Path("temp") / "amir_v2_r1_r20"
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

    # Cache baseline/full per dataset (independent of model)
    baseline_cache: dict[str, dict[int, float]] = {}
    full_cache: dict[str, dict[int, float]] = {}

    lines: list[str] = []
    lines.append("| Dataset | Model | Hard% | Baseline R@1 | R@5 | R@10 | R@20 | Adaptive R@1 | R@5 | R@10 | R@20 | Full R@1 | R@5 | R@10 | R@20 |")
    lines.append("|---------|-------|------:|-------------:|----:|-----:|-----:|-------------:|----:|-----:|-----:|--------:|---:|-----:|-----:|")

    for ds in datasets:
        if ds.name not in baseline_cache:
            baseline_cache[ds.name] = baseline_recalls(ds.preds_dir)
        if ds.name not in full_cache:
            full_cache[ds.name] = full_recalls(ds.preds_dir, ds.inliers_dir)

        b = baseline_cache[ds.name]
        f = full_cache[ds.name]

        for model_name, gate_pkl in models:
            gate_npz = out_dir / f"gate_{ds.name.replace(' ','_').replace('-','_')}_{model_name.replace(' ','_').replace('+','plus')}.npz"
            hard_pct = apply_gate_to_features(gate_pkl, ds.features_npz, gate_npz)
            a = adaptive_recalls(ds.preds_dir, ds.inliers_dir, gate_npz)

            lines.append(
                f"| {ds.name} | {model_name} | {hard_pct:5.1f}% | "
                f"{b[1]:6.2f}% | {b[5]:5.2f}% | {b[10]:6.2f}% | {b[20]:6.2f}% | "
                f"{a[1]:6.2f}% | {a[5]:5.2f}% | {a[10]:6.2f}% | {a[20]:6.2f}% | "
                f"{f[1]:6.2f}% | {f[5]:5.2f}% | {f[10]:6.2f}% | {f[20]:6.2f}% |"
            )

    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


