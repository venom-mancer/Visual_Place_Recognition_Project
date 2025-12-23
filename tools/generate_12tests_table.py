"""
Generate a 12-row table (4 datasets x 3 models) comparing:
- Baseline retrieval R@1
- Adaptive R@1 (rerank only hard queries)
- Full re-ranking R@1 (rerank all queries)

Adaptive is simulated by copying full inliers (.torch) only for the detected hard queries
and writing empty .torch files for easy queries (skip re-ranking).

Outputs JSON to stdout.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# Ensure project root is on sys.path so we can import util.py when invoked as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


POS_THRESH = 25
NUM_PREDS = 20


@dataclass(frozen=True)
class ModelCfg:
    name: str
    key: str
    model_pkl: Path


@dataclass(frozen=True)
class DatasetCfg:
    name: str
    key: str
    preds_dir: Path
    full_inliers_dir: Path
    feature_npz: Path
    info_log: Path | None


def parse_r1_from_log_text(text: str) -> float | None:
    m = re.search(r"R@1:\s*(\d+\.\d+)", text)
    return float(m.group(1)) if m else None


def baseline_r1(preds_dir: Path, info_log: Path | None) -> float:
    if info_log and info_log.exists():
        r1 = parse_r1_from_log_text(info_log.read_text(encoding="utf-8", errors="ignore"))
        if r1 is not None:
            return r1

    # Fallback: compute from top-1 distance in preds/*.txt
    from util import get_list_distances_from_preds

    files = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem))
    ok = 0
    for f in files:
        dists = get_list_distances_from_preds(str(f))
        if dists and dists[0] <= POS_THRESH:
            ok += 1
    return ok / len(files) * 100.0


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
    r1 = parse_r1_from_log_text(out.stdout)
    if r1 is None:
        raise RuntimeError("Could not parse R@1 from reranking.py output.")
    return r1


def run_stage4(model_pkl: Path, feature_npz: Path, out_npz: Path, out_txt: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "extension_6_1.stage_4_apply_logreg_easy_queries",
        "--model-path",
        str(model_pkl),
        "--feature-path",
        str(feature_npz),
        "--output-path",
        str(out_npz),
        "--hard-queries-output",
        str(out_txt),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr)


def simulate_inliers_from_full(
    full_inliers_dir: Path, hard_ids: set[int], out_dir: Path, n_queries: int
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect filename padding from the full inliers directory.
    sample = next(iter(full_inliers_dir.glob("*.torch")))
    pad = len(sample.stem)

    for qid in range(n_queries):
        stem = str(qid).zfill(pad)
        src = full_inliers_dir / f"{stem}.torch"
        dst = out_dir / f"{stem}.torch"
        if qid in hard_ids:
            shutil.copyfile(src, dst)
        else:
            torch.save([], dst)


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
    r1 = parse_r1_from_log_text(out.stdout)
    if r1 is None:
        raise RuntimeError("Could not parse R@1 from adaptive eval output.")
    return r1


def main() -> int:
    work = Path("temp") / "amv2_12tests"
    work.mkdir(parents=True, exist_ok=True)

    models = [
        ModelCfg("Night + Sun", "night_sun", Path("models_three_way_comparison/logreg_easy_night_sun.pkl")),
        ModelCfg("Night Only", "night_only", Path("models_three_way_comparison/logreg_easy_night_only.pkl")),
        ModelCfg("Sun Only", "sun_only", Path("models_three_way_comparison/logreg_easy_sun_only.pkl")),
    ]

    datasets = [
        DatasetCfg(
            "SF-XS test",
            "sf_xs",
            Path("logs/log_sf_xs_test/2025-12-17_21-14-10/preds"),
            Path("logs/log_sf_xs_test/2025-12-17_21-14-10/preds_superpoint-lg"),
            Path("data/features_and_predictions/features_sf_xs_test_improved.npz"),
            Path("logs/log_sf_xs_test/2025-12-17_21-14-10/info.log"),
        ),
        DatasetCfg(
            "Tokyo-XS test",
            "tokyo_xs",
            Path("log_tokyo_xs_test/2025-12-18_14-43-02/preds"),
            Path("log_tokyo_xs_test/2025-12-18_14-43-02/preds_superpoint-lg"),
            Path("data/features_and_predictions/features_tokyo_xs_test_improved.npz"),
            Path("log_tokyo_xs_test/2025-12-18_14-43-02/info.log"),
        ),
        DatasetCfg(
            "SVOX Night test",
            "svox_night",
            Path("logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds"),
            Path("logs/log_svox_night_test_cached/2025-12-21_01-27-42/preds_superpoint-lg_top20_256"),
            Path("data/features_and_predictions/features_svox_night_test_with_inliers.npz"),
            None,
        ),
        DatasetCfg(
            "SVOX Sun test",
            "svox_sun",
            Path("logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds"),
            Path("logs/log_svox_sun_test_cached/2025-12-21_01-33-34/preds_superpoint-lg_top20_256"),
            Path("data/features_and_predictions/features_svox_sun_test_with_inliers.npz"),
            None,
        ),
    ]

    rows = []

    for ds in datasets:
        b = baseline_r1(ds.preds_dir, ds.info_log)
        f = full_rerank_r1(ds.preds_dir, ds.full_inliers_dir)

        n_queries = len(list(ds.preds_dir.glob("*.txt")))

        for m in models:
            out_npz = work / f"logreg_{ds.key}_{m.key}.npz"
            out_txt = work / f"hard_{ds.key}_{m.key}.txt"
            run_stage4(m.model_pkl, ds.feature_npz, out_npz, out_txt)

            dd = np.load(out_npz)
            thr = float(dd["optimal_threshold"]) if "optimal_threshold" in dd else float("nan")
            hard_pct = float(dd["is_hard"].mean() * 100.0)

            hard_ids = set(int(x.strip()) for x in out_txt.read_text().splitlines() if x.strip())

            sim_dir = work / f"sim_inliers_{ds.key}_{m.key}"
            simulate_inliers_from_full(ds.full_inliers_dir, hard_ids, sim_dir, n_queries)

            a = adaptive_r1(ds.preds_dir, sim_dir, out_npz)
            ratio = a / f * 100.0 if f else float("nan")

            rows.append(
                {
                    "Dataset": ds.name,
                    "Model": m.name,
                    "Threshold Used": thr,
                    "Hard Queries Detected": hard_pct,
                    "Baseline R@1": b,
                    "Adaptive R@1": a,
                    "Full Re-ranking R@1": f,
                    "Ratio": ratio,
                    "Note": "ok",
                }
            )

    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


