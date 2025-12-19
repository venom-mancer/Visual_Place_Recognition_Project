import os
import time
from argparse import Namespace
from pathlib import Path
import json

from match_queries_preds import main as match_main
from reranking import main as rerank_main


def run_dataset(name: str, preds_dir: Path, out_dir: Path, device: str = "cuda", im_size: int = 512):
    """
    Run SuperPoint+LightGlue top-20 matching and full reranking
    for a single CosPlace test dataset, and measure runtimes.
    """
    preds_dir = preds_dir.resolve()
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    timings = {}

    # 1) Matcher: top-20 inliers with SuperPoint+LightGlue
    t0 = time.time()
    match_args = Namespace(
        preds_dir=str(preds_dir),
        out_dir=str(out_dir),
        matcher="superpoint-lg",
        device=device,
        im_size=im_size,
        num_preds=20,
        start_query=-1,
        num_queries=-1,
    )
    print(f"\n=== [{name}] Matching with SuperPoint+LG (top-20) ===")
    match_main(match_args)
    t1 = time.time()
    timings["matcher_seconds"] = t1 - t0

    # 2) Full reranking using the computed inliers
    t2 = time.time()
    rerank_args = Namespace(
        preds_dir=str(preds_dir),
        inliers_dir=str(out_dir),
        num_preds=20,
        positive_dist_threshold=25,
        recall_values=[1, 5, 10, 20],
        matcher="superpoint-lg",
        vpr_method="cosplace",
    )
    print(f"\n=== [{name}] Full reranking (using SuperPoint+LG top-20) ===")
    rerank_main(rerank_args)
    t3 = time.time()
    timings["reranking_seconds"] = t3 - t2
    timings["total_seconds"] = t3 - t0

    return timings


def main():
    root = Path(__file__).parent
    vpr_root = root / "VPR-methods-evaluation"
    logs_root = vpr_root / "logs"

    # CosPlace test logs (fixed timestamps from your runs)
    datasets = {
        "svox_test_sun": {
            "preds": logs_root
            / "logs_cosplace_svox_test_sun"
            / "2025-12-16_20-56-45"
            / "preds",
        },
        "svox_test_night": {
            "preds": logs_root
            / "logs_cosplace_svox_test_night"
            / "2025-12-16_21-30-20"
            / "preds",
        },
        "sf_xs_test": {
            "preds": logs_root
            / "logs_cosplace_sf_xs_test"
            / "2025-12-16_20-37-45"
            / "preds",
        },
        "tokyo_xs_test": {
            "preds": logs_root
            / "logs_cosplace_tokyo_xs_test"
            / "2025-12-16_20-51-23"
            / "preds",
        },
    }

    results = {}

    for name, cfg in datasets.items():
        preds_dir = cfg["preds"]
        if not preds_dir.exists():
            print(f"WARNING: preds_dir does not exist for {name}: {preds_dir}")
            continue

        out_dir = preds_dir.parent / "preds_superpoint-lg_full20"

        # IMPORTANT: run from inside VPR-methods-evaluation so that
        # relative paths in the .txt files (e.g. '../data/...') resolve correctly.
        old_cwd = Path.cwd()
        try:
            os.chdir(vpr_root)
            timings = run_dataset(name, preds_dir, out_dir)
        finally:
            os.chdir(old_cwd)

        results[name] = {
            "preds_dir": str(preds_dir),
            "inliers_dir": str(out_dir),
            **timings,
        }

    # Save summary timings to a JSON file
    summary_path = root / "adaptive_reranking" / "full_pipeline_cosplace_superpoint-lg_timings.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump(results, f, indent=2)

    print("\n=== Summary (CosPlace + SuperPoint+LG full pipeline) ===")
    for name, info in results.items():
        print(
            f"{name}: matcher={info['matcher_seconds']:.1f}s, "
            f"reranking={info['reranking_seconds']:.1f}s, "
            f"total={info['total_seconds']:.1f}s"
        )
    print(f"\nTiming summary saved to: {summary_path}")


if __name__ == "__main__":
    main()


