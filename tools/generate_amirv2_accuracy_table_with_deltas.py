"""
Generate a markdown table for Amir_V2 accuracy including:
- Hard% (and time saving)
- Baseline R@1, Adaptive R@1, Full R@1
- Delta vs baseline and delta vs full

Reads rows from: temp/amir_v2_12tests_rows.json
Prints markdown table to stdout.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> int:
    rows = json.loads(Path("temp/amir_v2_12tests_rows.json").read_text(encoding="utf-8-sig"))

    # stable order
    dataset_order = {"SF-XS test": 0, "Tokyo-XS test": 1, "SVOX Night test": 2, "SVOX Sun test": 3}
    model_order = {"Night + Sun": 0, "Night Only": 1, "Sun Only": 2}

    rows.sort(key=lambda r: (dataset_order.get(r["Dataset"], 99), model_order.get(r["Model"], 99)))

    # Avoid non-ASCII characters for Windows console encoding compatibility.
    print("| Dataset | Model | Hard% | Time Saving | Baseline R@1 | Adaptive R@1 | Full R@1 | dR1 vs Baseline | dR1 vs Full |")
    print("|---------|-------|-------|------------|-------------|-------------|--------|----------------|------------|")

    for r in rows:
        hard = float(r["Hard Queries Detected"])
        save = float(r["Time Savings"])
        b = float(r["Baseline R@1"])
        a = float(r["Adaptive R@1"])
        f = float(r["Full Re-ranking R@1"])
        d_base = a - b
        d_full = a - f
        print(
            f"| {r['Dataset']} | {r['Model']} | {hard:.1f}% | {save:.1f}% | {b:.2f}% | {a:.2f}% | {f:.2f}% | {d_base:+.2f}% | {d_full:+.2f}% |"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


