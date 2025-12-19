from pathlib import Path
import torch
from util import read_file_preds


LOGS_ROOT = Path("./logs")
OUT_FILE = Path("logs_summary.txt")


def parse_info_log(info_log_path):
    dataset = None
    vpr_method = None

    with open(info_log_path, "r") as f:
        for line in f:
            line = line.strip().lower()

            if "dataset" in line:
                dataset = line.split(":")[-1].strip()

            if "vpr" in line:
                vpr_method = line.split(":")[-1].strip()

    return dataset, vpr_method


def compute_avg_inliers(preds_dir, inliers_dir):
    preds_path = Path(preds_dir)
    inliers_path = Path(inliers_dir)

    correct, incorrect = [], []

    txt_files = sorted(preds_path.glob("*.txt"), key=lambda p: int(p.stem))

    for txt_file in txt_files:
        query_id = txt_file.stem
        torch_file = inliers_path / f"{query_id}.torch"

        if not torch_file.exists():
            continue

        _, pred_paths = read_file_preds(str(txt_file))
        if not pred_paths:
            continue

        top1_path = pred_paths[0].strip()

        with open(txt_file, "r") as f:
            lines = [l.strip() for l in f.readlines()]

        positives = []
        if "Positives paths:" in lines:
            idx = lines.index("Positives paths:") + 1
            positives = [l for l in lines[idx:] if l]

        results = torch.load(torch_file, weights_only=False)
        if not results:
            continue

        inliers = int(results[0]["num_inliers"])

        if top1_path in positives:
            correct.append(inliers)
        else:
            incorrect.append(inliers)

    avg_correct = sum(correct) / len(correct) if correct else 0.0
    avg_incorrect = sum(incorrect) / len(incorrect) if incorrect else 0.0

    return avg_correct, avg_incorrect


def main():
    lines = []
    header = "Dataset | VPR | Matcher | Avg Correct | Avg Incorrect"
    lines.append(header)
    lines.append("-" * len(header))

    for run_dir in LOGS_ROOT.rglob("*"):
        info_log = run_dir / "info.log"
        preds_dir = run_dir / "preds"

        if not info_log.exists() or not preds_dir.exists():
            continue

        dataset, vpr_method = parse_info_log(info_log)

        for inliers_dir in run_dir.glob("preds_*"):
            matcher_method = inliers_dir.name.replace("preds_", "")

            avg_c, avg_i = compute_avg_inliers(preds_dir, inliers_dir)

            lines.append(
                f"{dataset} | {vpr_method} | {matcher_method} | "
                f"{avg_c:.2f} | {avg_i:.2f}"
            )

    OUT_FILE.write_text("\n".join(lines))
    print(f"Summary written to {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
