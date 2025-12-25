"""
Generate a markdown timing table for the true Amir_V2 pipeline:

Total time ≈ retrieval_time + top1_time + hard_fraction * full_top20_time

Where:
- retrieval_time comes from info.log when available
- top1_time comes from a recent measurement (you can re-run and update constants)
- full_top20_time is approximated via mtime-span of an existing full inliers folder

Reads hard% from: temp/amir_v2_12tests_rows.json
Prints a markdown table to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path


# Measured top-1 matching runtimes (seconds) from this workspace run
TOP1_SECONDS = {
    "SF-XS test": 520.39,
    "Tokyo-XS test": 164.90,
    "SVOX Night test": 414.99,
    "SVOX Sun test": 431.89,
}

# Retrieval runtimes from info.log where available (seconds)
RETRIEVAL_SECONDS = {
    "SF-XS test": 551.13,
    "Tokyo-XS test": 182.43,
}

# Full top-20 matching runtimes approximated via mtime span (seconds)
FULL_TOP20_SECONDS = {
    "SF-XS test": 9343.83,
    "Tokyo-XS test": 2936.45,
    "SVOX Night test": 20075.01,
    "SVOX Sun test": 20446.41,
}


def fmt_s(secs: float | None) -> str:
    if secs is None:
        return "N/A"
    if secs >= 3600:
        return f"{secs/3600:.2f}h"
    if secs >= 60:
        return f"{secs/60:.1f}m"
    return f"{secs:.1f}s"


def main() -> int:
    # PowerShell Out-File may add a UTF-8 BOM; use utf-8-sig to be safe.
    rows = json.loads(Path("temp/amir_v2_12tests_rows.json").read_text(encoding="utf-8-sig"))
    # hard fraction by (dataset, model)
    hard = {(r["Dataset"], r["Model"]): float(r["Hard Queries Detected"]) / 100.0 for r in rows}

    datasets = ["SF-XS test", "Tokyo-XS test", "SVOX Night test", "SVOX Sun test"]
    models = ["Night + Sun", "Night Only", "Sun Only"]

    print("| Dataset | Model | Hard% | Time Saving | Retrieval | Top-1 match | Top-K match (hard only, est.) | Total (est.) | Full pipeline (est.) |")
    print("|---------|-------|-------|------------|----------|------------|-------------------------------|-------------|----------------------|")

    for ds in datasets:
        rt = RETRIEVAL_SECONDS.get(ds)
        t1 = TOP1_SECONDS.get(ds)
        t20 = FULL_TOP20_SECONDS.get(ds)
        full_total = (rt or 0.0) + (t20 or 0.0)

        for m in models:
            h = hard[(ds, m)]
            topk_hard = (t20 or 0.0) * h
            total = (rt or 0.0) + (t1 or 0.0) + topk_hard
            print(
                f"| {ds} | {m} | {h*100:.1f}% | {(1-h)*100:.1f}% | {fmt_s(rt)} | {fmt_s(t1)} | {fmt_s(topk_hard)} | {fmt_s(total)} | {fmt_s(full_total)} |"
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


