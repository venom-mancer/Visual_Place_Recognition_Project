"""
Compare all solutions for detecting hard queries on problematic test sets.

Solutions:
1. Baseline (original threshold)
2. Threshold Calibration
3. Temperature Scaling
4. Ensemble Method
5. Robust Normalization
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file


def load_solution_results(solution_path):
    """Load results from a solution."""
    if not Path(solution_path).exists():
        return None
    
    data = np.load(solution_path)
    return {
        "probs": data["probs"],
        "is_hard": data["is_hard"],
        "labels": data["labels"],
        "hard_rate": data["is_hard"].mean(),
        "actual_hard_rate": (1 - data["labels"].mean()),
        "accuracy": data.get("accuracy", (data["is_easy"] == data["labels"]).mean()),
        "f1": data.get("f1", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all solutions for hard query detection"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["tokyo", "svox"],
        required=True,
        help="Test set to compare"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/solutions_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define solution paths
    if args.test_set == "tokyo":
        test_name = "Tokyo-XS Test"
        baseline_path = "data/features_and_predictions/logreg_tokyo_xs_test_C_tuned.npz"
        threshold_cal_path = "data/features_and_predictions/logreg_tokyo_xs_calibrated.npz"
        temp_scale_path = "data/features_and_predictions/logreg_tokyo_xs_temperature_scaled.npz"
        ensemble_path = "data/features_and_predictions/logreg_tokyo_xs_ensemble.npz"
    else:  # svox
        test_name = "SVOX Test"
        baseline_path = None  # Need to create
        threshold_cal_path = None  # Need to create
        temp_scale_path = "data/features_and_predictions/logreg_svox_temperature_scaled.npz"
        ensemble_path = None  # Need to create
    
    # Load all solutions
    solutions = {}
    
    if baseline_path and Path(baseline_path).exists():
        solutions["Baseline"] = load_solution_results(baseline_path)
    
    if threshold_cal_path and Path(threshold_cal_path).exists():
        solutions["Threshold Calibration"] = load_solution_results(threshold_cal_path)
    
    if temp_scale_path and Path(temp_scale_path).exists():
        solutions["Temperature Scaling"] = load_solution_results(temp_scale_path)
    
    if ensemble_path and Path(ensemble_path).exists():
        solutions["Ensemble"] = load_solution_results(ensemble_path)
    
    # Create comparison table
    table_path = output_dir / f"{args.test_set}_solutions_comparison.md"
    with open(table_path, 'w') as f:
        f.write(f"# Solutions Comparison: {test_name}\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Solution | Hard Query Rate | Actual Hard Rate | Accuracy | F1-Score | Status |\n")
        f.write("|----------|-----------------|------------------|----------|----------|--------|\n")
        
        for sol_name, sol_data in solutions.items():
            if sol_data is None:
                continue
            status = "OK" if sol_data["hard_rate"] > 0.01 else "FAIL"
            f.write(f"| {sol_name} | {sol_data['hard_rate']:.1%} | {sol_data['actual_hard_rate']:.1%} | "
                   f"{sol_data['accuracy']:.1%} | {sol_data['f1']:.4f} | {status} |\n")
    
    print(f"Saved comparison table to: {table_path}")
    
    # Create comparison chart
    if len(solutions) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sol_names = []
        hard_rates = []
        actual_hard_rate = None
        
        for sol_name, sol_data in solutions.items():
            if sol_data is None:
                continue
            sol_names.append(sol_name)
            hard_rates.append(sol_data["hard_rate"] * 100)
            if actual_hard_rate is None:
                actual_hard_rate = sol_data["actual_hard_rate"] * 100
        
        x = np.arange(len(sol_names))
        width = 0.35
        
        bars = ax.bar(x, hard_rates, width, label='Predicted Hard Rate', alpha=0.7)
        
        # Add actual hard rate line
        if actual_hard_rate is not None:
            ax.axhline(y=actual_hard_rate, color='red', linestyle='--', linewidth=2,
                      label=f'Actual Hard Rate: {actual_hard_rate:.1f}%')
        
        ax.set_xlabel('Solution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Solutions Comparison: {test_name}\nHard Query Detection Rate', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sol_names, rotation=15, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, hard_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = output_dir / f"{args.test_set}_solutions_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison chart to: {chart_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Solutions Comparison: {test_name}")
    print(f"{'='*70}")
    for sol_name, sol_data in solutions.items():
        if sol_data is None:
            continue
        status = "OK" if sol_data["hard_rate"] > 0.01 else "FAIL"
        print(f"\n{sol_name}:")
        print(f"  Hard Query Rate: {sol_data['hard_rate']:.1%}")
        print(f"  Actual Hard Rate: {sol_data['actual_hard_rate']:.1%}")
        print(f"  Accuracy: {sol_data['accuracy']:.1%}")
        print(f"  F1-Score: {sol_data['f1']:.4f}")
        print(f"  Status: {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

Compare all solutions for detecting hard queries on problematic test sets.

Solutions:
1. Baseline (original threshold)
2. Threshold Calibration
3. Temperature Scaling
4. Ensemble Method
5. Robust Normalization
"""

import argparse
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file


def load_solution_results(solution_path):
    """Load results from a solution."""
    if not Path(solution_path).exists():
        return None
    
    data = np.load(solution_path)
    return {
        "probs": data["probs"],
        "is_hard": data["is_hard"],
        "labels": data["labels"],
        "hard_rate": data["is_hard"].mean(),
        "actual_hard_rate": (1 - data["labels"].mean()),
        "accuracy": data.get("accuracy", (data["is_easy"] == data["labels"]).mean()),
        "f1": data.get("f1", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all solutions for hard query detection"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["tokyo", "svox"],
        required=True,
        help="Test set to compare"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/solutions_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define solution paths
    if args.test_set == "tokyo":
        test_name = "Tokyo-XS Test"
        baseline_path = "data/features_and_predictions/logreg_tokyo_xs_test_C_tuned.npz"
        threshold_cal_path = "data/features_and_predictions/logreg_tokyo_xs_calibrated.npz"
        temp_scale_path = "data/features_and_predictions/logreg_tokyo_xs_temperature_scaled.npz"
        ensemble_path = "data/features_and_predictions/logreg_tokyo_xs_ensemble.npz"
    else:  # svox
        test_name = "SVOX Test"
        baseline_path = None  # Need to create
        threshold_cal_path = None  # Need to create
        temp_scale_path = "data/features_and_predictions/logreg_svox_temperature_scaled.npz"
        ensemble_path = None  # Need to create
    
    # Load all solutions
    solutions = {}
    
    if baseline_path and Path(baseline_path).exists():
        solutions["Baseline"] = load_solution_results(baseline_path)
    
    if threshold_cal_path and Path(threshold_cal_path).exists():
        solutions["Threshold Calibration"] = load_solution_results(threshold_cal_path)
    
    if temp_scale_path and Path(temp_scale_path).exists():
        solutions["Temperature Scaling"] = load_solution_results(temp_scale_path)
    
    if ensemble_path and Path(ensemble_path).exists():
        solutions["Ensemble"] = load_solution_results(ensemble_path)
    
    # Create comparison table
    table_path = output_dir / f"{args.test_set}_solutions_comparison.md"
    with open(table_path, 'w') as f:
        f.write(f"# Solutions Comparison: {test_name}\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Solution | Hard Query Rate | Actual Hard Rate | Accuracy | F1-Score | Status |\n")
        f.write("|----------|-----------------|------------------|----------|----------|--------|\n")
        
        for sol_name, sol_data in solutions.items():
            if sol_data is None:
                continue
            status = "OK" if sol_data["hard_rate"] > 0.01 else "FAIL"
            f.write(f"| {sol_name} | {sol_data['hard_rate']:.1%} | {sol_data['actual_hard_rate']:.1%} | "
                   f"{sol_data['accuracy']:.1%} | {sol_data['f1']:.4f} | {status} |\n")
    
    print(f"Saved comparison table to: {table_path}")
    
    # Create comparison chart
    if len(solutions) > 0:
        fig, ax = plt.subplots(figsize=(12, 8))
        
        sol_names = []
        hard_rates = []
        actual_hard_rate = None
        
        for sol_name, sol_data in solutions.items():
            if sol_data is None:
                continue
            sol_names.append(sol_name)
            hard_rates.append(sol_data["hard_rate"] * 100)
            if actual_hard_rate is None:
                actual_hard_rate = sol_data["actual_hard_rate"] * 100
        
        x = np.arange(len(sol_names))
        width = 0.35
        
        bars = ax.bar(x, hard_rates, width, label='Predicted Hard Rate', alpha=0.7)
        
        # Add actual hard rate line
        if actual_hard_rate is not None:
            ax.axhline(y=actual_hard_rate, color='red', linestyle='--', linewidth=2,
                      label=f'Actual Hard Rate: {actual_hard_rate:.1f}%')
        
        ax.set_xlabel('Solution', fontsize=12, fontweight='bold')
        ax.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'Solutions Comparison: {test_name}\nHard Query Detection Rate', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(sol_names, rotation=15, ha='right')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, (bar, rate) in enumerate(zip(bars, hard_rates)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        chart_path = output_dir / f"{args.test_set}_solutions_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved comparison chart to: {chart_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Solutions Comparison: {test_name}")
    print(f"{'='*70}")
    for sol_name, sol_data in solutions.items():
        if sol_data is None:
            continue
        status = "OK" if sol_data["hard_rate"] > 0.01 else "FAIL"
        print(f"\n{sol_name}:")
        print(f"  Hard Query Rate: {sol_data['hard_rate']:.1%}")
        print(f"  Actual Hard Rate: {sol_data['actual_hard_rate']:.1%}")
        print(f"  Accuracy: {sol_data['accuracy']:.1%}")
        print(f"  F1-Score: {sol_data['f1']:.4f}")
        print(f"  Status: {status}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


