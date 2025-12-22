"""
Batch-evaluate adaptive re-ranking for ONE (VPR, matcher) combo.

This script is a wrapper around `adaptive_reranking/adaptive_reranking_eval.py`.

It will:
  - discover LR model .pkl files inside a tuning results folder
  - run adaptive evaluation on 4 test datasets
  - parse key metrics from stdout (threshold, %hard/%easy, Recall@1, runtime)
  - write:
      1) summary.csv  (easy to copy into Excel)
      2) summary.txt  (human-readable table)
      3) raw.log      (full stdout of all runs)

Notes:
  - It can also optionally load a "full pipeline" timings JSON (if you already computed it)
    and will add `full_pipeline_time_sec` per dataset. Full re-ranking R@1 is left blank
    (fill manually if desired).
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm


DATASETS = ["sf_xs_test", "tokyo_xs_test", "svox_test_sun", "svox_test_night"]


@dataclass
class RunResult:
    combo: str
    dataset: str
    lr_model_name: str  # sun / night / combined
    lr_model_path: str
    threshold_used: Optional[float]
    pct_hard: Optional[float]
    pct_easy: Optional[float]
    adaptive_r1: Optional[float]
    adaptive_r5: Optional[float]
    adaptive_r10: Optional[float]
    adaptive_r20: Optional[float]
    processed_queries: Optional[int]
    total_queries: Optional[int]
    skipped_queries: Optional[int]
    easy_queries: Optional[int]
    hard_queries: Optional[int]
    extra_pairs_total: Optional[int]
    avg_total_pairs_incl_top1: Optional[float]
    total_runtime_sec: Optional[float]
    avg_runtime_per_query_sec: Optional[float]
    baseline_r1: Optional[float]
    full_reranking_r1: Optional[float]
    full_pipeline_time_sec: Optional[float]


def _find_lr_models(lr_models_dir: Path) -> List[Tuple[str, Path]]:
    """
    Returns a list of (model_name, path) where model_name is one of: sun/night/combined.
    """
    pkls = sorted(lr_models_dir.glob("lr_model_*.pkl"))
    out: List[Tuple[str, Path]] = []
    for p in pkls:
        m = re.match(r"lr_model_(sun|night|combined)_C.*\.pkl$", p.name)
        if m:
            out.append((m.group(1), p))
    # stable order
    order = {"combined": 0, "night": 1, "sun": 2}
    out.sort(key=lambda t: order.get(t[0], 99))
    return out


def _infer_top1_inliers_dir(preds_dir: Path, matcher: str) -> Path:
    parent = preds_dir.parent
    if matcher == "loftr":
        return parent / "preds_loftr"
    return parent / f"preds_{matcher}"


def _infer_full20_inliers_dir(preds_dir: Path, matcher: str) -> Path:
    parent = preds_dir.parent
    if matcher == "loftr":
        return parent / "preds_loftr_full20"
    return parent / f"preds_{matcher}_full20"


def _compute_full_reranking_r1(preds_dir: Path, inliers_full20_dir: Path, num_preds: int, positive_dist_threshold: float) -> Optional[float]:
    """
    Compute full re-ranking Recall@1 using precomputed top-K inliers (*.torch) and the retrieval distances from preds txt files.
    """
    from util import get_list_distances_from_preds  # local import (sys.path set in main)

    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not txts:
        return None

    correct = 0
    processed = 0

    for txt in txts:
        qid = txt.stem
        torch_file = inliers_full20_dir / f"{qid}.torch"
        if not torch_file.exists():
            continue

        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(str(txt)))[:num_preds]
        except Exception:
            continue
        if len(geo_dists) == 0:
            continue

        try:
            results = torch.load(str(torch_file), weights_only=False)
        except Exception:
            continue
        if not results:
            continue

        k = min(num_preds, len(results), len(geo_dists))
        if k == 0:
            continue

        inliers = torch.tensor([float(results[i].get("num_inliers", 0.0)) for i in range(k)], dtype=torch.float32)
        _, idxs = torch.sort(inliers, descending=True)
        reranked_dists = geo_dists[idxs]

        processed += 1
        if len(reranked_dists) > 0 and float(reranked_dists[0]) <= positive_dist_threshold:
            correct += 1

    return (correct / processed * 100.0) if processed else None


def _baseline_top1_in_positives_rate(preds_dir: Path) -> Optional[float]:
    """
    Computes baseline "top-1 correct" rate by checking if the top-1 predicted path
    is listed under "Positives paths:" in each .txt file.
    """
    txts = sorted(preds_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    if not txts:
        return None

    correct = 0
    n = 0
    for p in txts:
        lines = [l.strip() for l in p.read_text(encoding="utf-8", errors="ignore").splitlines()]
        if "Predictions paths:" not in lines or "Positives paths:" not in lines:
            continue
        i = lines.index("Predictions paths:") + 1
        top1 = None
        while i < len(lines) and top1 is None:
            if lines[i]:
                top1 = lines[i]
            i += 1
        j = lines.index("Positives paths:") + 1
        positives = set(l for l in lines[j:] if l)
        if top1 is None:
            continue
        n += 1
        if top1 in positives:
            correct += 1

    return (correct / n * 100.0) if n else None


class _Tee:
    """
    Duplicate writes to multiple streams (used to mirror output to console + raw.log).
    We also report isatty()=True to keep tqdm progress bars enabled.
    """

    def __init__(self, *streams):
        self._streams = streams

    def write(self, s: str) -> int:
        n = 0
        for st in self._streams:
            try:
                n = st.write(s)
                st.flush()
            except Exception:
                pass
        return n

    def flush(self) -> None:
        for st in self._streams:
            try:
                st.flush()
            except Exception:
                pass

    def isatty(self) -> bool:  # tqdm checks this
        return True


def _format_table(rows: List[RunResult]) -> str:
    """
    Pretty fixed-width table grouped by dataset.
    """
    # columns (keep it readable)
    headers = [
        "Dataset",
        "LR",
        "thr",
        "hard%",
        "easy%",
        "baseR@1",
        "adapR@1",
        "fullR@1",
        "time(s)",
    ]

    def fnum(x: Optional[float], width: int, prec: int = 2) -> str:
        if x is None:
            return "".rjust(width)
        return f"{x:.{prec}f}".rjust(width)

    lines = []
    lines.append(" | ".join(h.ljust(w) for h, w in zip(headers, [14, 8, 6, 6, 6, 7, 7, 7, 8])))
    lines.append("-" * 80)

    for r in rows:
        lines.append(
            " | ".join(
                [
                    r.dataset.ljust(14),
                    r.lr_model_name.ljust(8),
                    fnum(r.threshold_used, 6, 2),
                    fnum(r.pct_hard, 6, 1),
                    fnum(r.pct_easy, 6, 1),
                    fnum(r.baseline_r1, 7, 2),
                    fnum(r.adaptive_r1, 7, 2),
                    fnum(r.full_reranking_r1, 7, 2),
                    fnum(r.total_runtime_sec, 8, 1),
                ]
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch eval for ONE VPR+matcher combo (adaptive reranking).")
    parser.add_argument("--combo-name", required=True, help='Display name, e.g. "Cosplace+Loftr".')
    parser.add_argument("--matcher", required=True, choices=["loftr", "superpoint-lg"], help="Matcher name.")
    parser.add_argument(
        "--lr-models-dir",
        required=True,
        help="Folder containing lr_model_{sun,night,combined}_*.pkl (e.g. tuning_results/).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=DATASETS,
        help="Which test datasets to run (default: all). Example: --datasets tokyo_xs_test",
    )
    parser.add_argument("--sf-xs-test-preds", required=True, help="preds dir for SF-XS test.")
    parser.add_argument("--tokyo-xs-test-preds", required=True, help="preds dir for Tokyo-XS test.")
    parser.add_argument("--svox-sun-test-preds", required=True, help="preds dir for SVOX Sun test.")
    parser.add_argument("--svox-night-test-preds", required=True, help="preds dir for SVOX Night test.")
    parser.add_argument(
        "--full-pipeline-timings-json",
        default=None,
        help="Optional JSON produced by run_full_*.py to auto-fill full_pipeline_time_sec per dataset.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Where to write outputs. Default: lr-models-dir/../batch_eval_<timestamp>/",
    )
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--im-size", type=int, default=512)
    parser.add_argument("--num-preds", type=int, default=20)
    parser.add_argument("--positive-dist-threshold", type=int, default=25)

    args = parser.parse_args()

    project_root = Path(__file__).parent.parent
    vpr_root = project_root / "VPR-methods-evaluation"
    if not vpr_root.exists():
        raise FileNotFoundError(f"Expected VPR-methods-evaluation at: {vpr_root}")

    # Import eval module (namespace package import is fine in Py3)
    sys.path.insert(0, str(project_root))
    from adaptive_reranking import adaptive_reranking_eval as are  # type: ignore

    get_matcher, _, _ = are._setup_matching_module()

    lr_models_dir = Path(args.lr_models_dir)
    if not lr_models_dir.exists():
        raise FileNotFoundError(f"lr-models-dir does not exist: {lr_models_dir}")

    models = _find_lr_models(lr_models_dir)
    if not models:
        raise FileNotFoundError(
            f"No lr_model_{{sun,night,combined}}_*.pkl found in: {lr_models_dir}"
        )

    selected_datasets: List[str] = list(args.datasets)

    preds_dirs: Dict[str, Path] = {
        "sf_xs_test": Path(args.sf_xs_test_preds),
        "tokyo_xs_test": Path(args.tokyo_xs_test_preds),
        "svox_test_sun": Path(args.svox_sun_test_preds),
        "svox_test_night": Path(args.svox_night_test_preds),
    }
    for k in selected_datasets:
        p = preds_dirs[k]
        if not p.exists():
            raise FileNotFoundError(f"{k} preds dir does not exist: {p}")

    # optional full pipeline timings
    full_pipeline_times: Dict[str, float] = {}
    if args.full_pipeline_timings_json is not None:
        jp = Path(args.full_pipeline_timings_json)
        if not jp.exists():
            raise FileNotFoundError(f"full-pipeline-timings-json does not exist: {jp}")
        data = json.loads(jp.read_text(encoding="utf-8"))
        for ds in DATASETS:
            if ds in data and isinstance(data[ds], dict) and "total_seconds" in data[ds]:
                full_pipeline_times[ds] = float(data[ds]["total_seconds"])

    # output
    if args.output_dir is None:
        ts = time.strftime("%Y%m%d_%H%M%S")
        output_dir = lr_models_dir.parent / f"batch_eval_{ts}"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_log_path = output_dir / "raw.log"
    summary_csv_path = output_dir / "summary.csv"
    summary_txt_path = output_dir / "summary.txt"

    all_rows: List[RunResult] = []

    total_runs = len(selected_datasets) * len(models)
    print(
        f"\nRunning batch eval for: {args.combo_name} | matcher={args.matcher} | "
        f"datasets={selected_datasets} | models={[m for m, _ in models]} | total runs={total_runs}\n"
    )

    with raw_log_path.open("w", encoding="utf-8") as raw_log:
        raw_log.write(f"combo: {args.combo_name}\n")
        raw_log.write(f"matcher: {args.matcher}\n")
        raw_log.write(f"lr_models_dir: {lr_models_dir}\n\n")

        run_pairs: List[Tuple[str, str, Path]] = []
        for dataset_name in selected_datasets:
            for model_name, model_path in models:
                run_pairs.append((dataset_name, model_name, model_path))

        for dataset_name, model_name, model_path in tqdm(run_pairs, desc=f"{args.combo_name} batch", unit="run"):
            preds_dir = preds_dirs[dataset_name].resolve()
            top1_inliers_dir = _infer_top1_inliers_dir(preds_dir, args.matcher).resolve()
            if not top1_inliers_dir.exists():
                raise FileNotFoundError(
                    f"Top-1 inliers dir not found for {dataset_name}: inferred {top1_inliers_dir}\n"
                    f"If your folder name differs, rename it to match or adjust this script."
                )

            baseline_r1 = _baseline_top1_in_positives_rate(preds_dir)

            print(f"\n[RUN] dataset={dataset_name} | lr={model_name} | model={model_path.name}")
            raw_log.write("=" * 100 + "\n")
            raw_log.write(f"[{dataset_name}] model={model_name} | model_path={model_path}\n")
            raw_log.write("-" * 100 + "\n")

            # Build an args namespace compatible with adaptive_reranking_eval.evaluate()
            eval_args = argparse.Namespace(
                preds_dir=str(preds_dir),
                top1_inliers_dir=str(top1_inliers_dir),
                lr_model=str(model_path.resolve()),
                num_preds=args.num_preds,
                positive_dist_threshold=args.positive_dist_threshold,
                recall_values=[1, 5, 10, 20],
                threshold=None,
                matcher=args.matcher,
                device=args.device,
                im_size=args.im_size,
                wandb_project=None,
                wandb_run_name=None,
                tqdm_desc=f"{dataset_name} | lr={model_name}",
                no_progress=False,
            )
            eval_args._get_matcher = get_matcher

            t0 = time.time()
            old_cwd = Path.cwd()
            old_stdout = sys.stdout
            try:
                # run from inside VPR-methods-evaluation so that relative paths in .txt files resolve
                import os

                os.chdir(vpr_root)
                sys.stdout = _Tee(old_stdout, raw_log)
                metrics = are.evaluate(eval_args)

                # Compute full reranking R@1 if precomputed top-20 inliers exist
                full20_dir = _infer_full20_inliers_dir(preds_dir, args.matcher).resolve()
                full_r1 = None
                if full20_dir.exists():
                    full_r1 = _compute_full_reranking_r1(
                        preds_dir=preds_dir,
                        inliers_full20_dir=full20_dir,
                        num_preds=args.num_preds,
                        positive_dist_threshold=float(args.positive_dist_threshold),
                    )
            except Exception as e:
                metrics = None
                full_r1 = None
                print(f"[ERROR] {dataset_name} | lr={model_name} failed: {e}")
            finally:
                sys.stdout = old_stdout
                try:
                    os.chdir(old_cwd)
                except Exception:
                    pass

            wall = time.time() - t0
            print(f"[DONE] dataset={dataset_name} | lr={model_name} | wall_time={wall:.1f}s")

            if not metrics:
                all_rows.append(
                    RunResult(
                        combo=args.combo_name,
                        dataset=dataset_name,
                        lr_model_name=model_name,
                        lr_model_path=str(model_path),
                        threshold_used=None,
                        pct_hard=None,
                        pct_easy=None,
                        adaptive_r1=None,
                        adaptive_r5=None,
                        adaptive_r10=None,
                        adaptive_r20=None,
                        processed_queries=None,
                        total_queries=None,
                        skipped_queries=None,
                        easy_queries=None,
                        hard_queries=None,
                        extra_pairs_total=None,
                        avg_total_pairs_incl_top1=None,
                        total_runtime_sec=None,
                        avg_runtime_per_query_sec=None,
                        baseline_r1=baseline_r1,
                        full_reranking_r1=None,
                        full_pipeline_time_sec=full_pipeline_times.get(dataset_name),
                    )
                )
                continue

            all_rows.append(
                RunResult(
                    combo=args.combo_name,
                    dataset=dataset_name,
                    lr_model_name=model_name,
                    lr_model_path=str(model_path),
                    threshold_used=metrics.get("threshold_used"),
                    pct_hard=metrics.get("pct_hard"),
                    pct_easy=metrics.get("pct_easy"),
                    adaptive_r1=metrics.get("recall@1"),
                    adaptive_r5=metrics.get("recall@5"),
                    adaptive_r10=metrics.get("recall@10"),
                    adaptive_r20=metrics.get("recall@20"),
                    processed_queries=metrics.get("processed_queries"),
                    total_queries=metrics.get("total_queries"),
                    skipped_queries=metrics.get("skipped_queries"),
                    easy_queries=metrics.get("easy_queries"),
                    hard_queries=metrics.get("hard_queries"),
                    extra_pairs_total=metrics.get("extra_pairs_total"),
                    avg_total_pairs_incl_top1=metrics.get("avg_total_pairs_incl_top1"),
                    total_runtime_sec=metrics.get("total_runtime_sec"),
                    avg_runtime_per_query_sec=metrics.get("avg_runtime_per_query_sec"),
                    baseline_r1=baseline_r1,
                    full_reranking_r1=full_r1,
                    full_pipeline_time_sec=full_pipeline_times.get(dataset_name),
                )
            )

    # Write CSV
    with summary_csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(all_rows[0]).keys()))
        writer.writeheader()
        for r in all_rows:
            writer.writerow(asdict(r))

    # Write pretty table (grouped by dataset)
    grouped: List[RunResult] = []
    for ds in selected_datasets:
        ds_rows = [r for r in all_rows if r.dataset == ds]
        # keep model order
        order = {"combined": 0, "night": 1, "sun": 2}
        ds_rows.sort(key=lambda r: order.get(r.lr_model_name, 99))
        grouped.extend(ds_rows)

    table = _format_table(grouped)
    summary_txt = (
        f"Combo: {args.combo_name}\n"
        f"Matcher: {args.matcher}\n"
        f"LR models dir: {lr_models_dir}\n"
        f"Device: {args.device}, K={args.num_preds}, im_size={args.im_size}\n\n"
        + table
        + "\n\n"
        + "Notes:\n"
        + "  - baseR@1 is computed as (top1 path in Positives paths) from the .txt files.\n"
        + "  - fullR@1 is computed from precomputed *_full20 inliers if available; otherwise blank.\n"
        + "  - full_pipeline_time_sec is filled only if you pass --full-pipeline-timings-json.\n"
    )
    summary_txt_path.write_text(summary_txt, encoding="utf-8")

    print("\n" + summary_txt)
    print(f"\nSaved:\n  - {summary_csv_path}\n  - {summary_txt_path}\n  - {raw_log_path}\n")


if __name__ == "__main__":
    main()


