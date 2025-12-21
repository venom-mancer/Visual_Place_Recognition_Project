"""
Complete evaluation pipeline:
1. Run adaptive image matching for temperature-scaled predictions
2. Run adaptive re-ranking evaluation
3. Compare with full re-ranking
4. Generate accuracy comparison charts
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm


def run_adaptive_matching(preds_dir, hard_queries_list, out_dir, matcher="superpoint-lg", device="cuda", num_preds=20):
    """Run adaptive image matching for hard queries only."""
    cmd = [
        sys.executable,
        "match_queries_preds_adaptive.py",
        "--preds-dir", str(preds_dir),
        "--hard-queries-list", str(hard_queries_list),
        "--out-dir", str(out_dir),
        "--matcher", matcher,
        "--device", device,
        "--num-preds", str(num_preds)
    ]
    
    print(f"Running adaptive image matching...")
    print(f"  Input: {preds_dir}")
    print(f"  Hard queries: {hard_queries_list}")
    print(f"  Output: {out_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    
    print(f"  [OK] Adaptive matching complete")
    return True


def run_adaptive_evaluation(preds_dir, inliers_dir, logreg_output_path, num_preds=20, positive_dist_threshold=25):
    """Run adaptive re-ranking evaluation and extract R@1."""
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir", str(preds_dir),
        "--inliers-dir", str(inliers_dir),
        "--logreg-output", str(logreg_output_path),
        "--num-preds", str(num_preds),
        "--positive-dist-threshold", str(positive_dist_threshold),
        "--recall-values", "1", "5", "10", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    # Parse R@1 from output
    r1 = None
    for line in result.stdout.split('\n'):
        if 'R@1' in line or 'Recall@1' in line or 'recall@1' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                r1 = float(match.group(1))
                break
    
    return r1


def get_full_reranking_r1(preds_dir, inliers_dir, num_preds=20, positive_dist_threshold=25):
    """Get full re-ranking R@1 (all queries get re-ranking)."""
    # Use reranking.py or stage_5 with all queries as hard
    # For now, we'll try to extract from existing results or compute
    
    # Check if we can use the inliers directory directly
    if not Path(inliers_dir).exists():
        return None
    
    # Try to compute from inliers directory
    # This is a simplified version - full implementation would use reranking.py
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Complete evaluation pipeline with temperature scaling"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["tokyo", "svox", "both"],
        default="both",
        help="Test set to evaluate"
    )
    parser.add_argument(
        "--tokyo-preds-dir",
        type=str,
        help="Tokyo-XS test predictions directory"
    )
    parser.add_argument(
        "--tokyo-inliers-dir",
        type=str,
        help="Tokyo-XS test inliers directory (full re-ranking)"
    )
    parser.add_argument(
        "--svox-preds-dir",
        type=str,
        help="SVOX test predictions directory"
    )
    parser.add_argument(
        "--svox-inliers-dir",
        type=str,
        help="SVOX test inliers directory (full re-ranking)"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip image matching (assume already done)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_accuracy_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        "Night + Sun": "night_sun",
        "Night Only": "night_only",
        "Sun Only": "sun_only"
    }
    
    test_configs = {}
    
    if args.test_set in ["tokyo", "both"] and args.tokyo_preds_dir:
        test_configs["Tokyo-XS"] = {
            "preds_dir": Path(args.tokyo_preds_dir),
            "inliers_dir": Path(args.tokyo_inliers_dir) if args.tokyo_inliers_dir else None,
            "test_key": "tokyo_xs"
        }
    
    if args.test_set in ["svox", "both"] and args.svox_preds_dir:
        test_configs["SVOX"] = {
            "preds_dir": Path(args.svox_preds_dir),
            "inliers_dir": Path(args.svox_inliers_dir) if args.svox_inliers_dir else None,
            "test_key": "svox"
        }
    
    # Collect results
    all_results = {}
    
    for test_name, config in test_configs.items():
        print(f"\n{'='*70}")
        print(f"Processing: {test_name}")
        print(f"{'='*70}")
        
        all_results[test_name] = {}
        
        # Get full re-ranking R@1 if available
        full_r1 = None
        if config["inliers_dir"] and config["inliers_dir"].exists():
            # Try to get from existing evaluation or compute
            print(f"\nFull re-ranking inliers directory: {config['inliers_dir']}")
            print(f"  (Full re-ranking R@1 needs to be computed separately)")
        
        all_results[test_name]["full_reranking"] = full_r1
        
        # Evaluate each model
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            # Check if temperature-scaled predictions exist
            logreg_path = Path(f"data/features_and_predictions/logreg_{config['test_key']}_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_{config['test_key']}_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists():
                print(f"  [SKIP] Temperature-scaled predictions not found: {logreg_path}")
                continue
            
            if not hard_queries_path.exists():
                print(f"  [SKIP] Hard queries list not found: {hard_queries_path}")
                continue
            
            # Run adaptive image matching
            adaptive_inliers_dir = config["preds_dir"].parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            if not args.skip_matching:
                if not adaptive_inliers_dir.exists() or len(list(adaptive_inliers_dir.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    success = run_adaptive_matching(
                        config["preds_dir"],
                        hard_queries_path,
                        adaptive_inliers_dir
                    )
                    if not success:
                        print(f"  [ERROR] Adaptive matching failed")
                        continue
                else:
                    print(f"  [SKIP] Adaptive matching already done: {adaptive_inliers_dir}")
            else:
                if not adaptive_inliers_dir.exists():
                    print(f"  [ERROR] Adaptive matching not done and --skip-matching set")
                    continue
            
            # Run adaptive evaluation
            print(f"  Running adaptive re-ranking evaluation...")
            adaptive_r1 = run_adaptive_evaluation(
                config["preds_dir"],
                adaptive_inliers_dir,
                logreg_path
            )
            
            if adaptive_r1 is not None:
                all_results[test_name][model_name] = {
                    "adaptive_r1": adaptive_r1,
                    "hard_queries_path": str(hard_queries_path),
                    "inliers_dir": str(adaptive_inliers_dir)
                }
                print(f"  [OK] Adaptive R@1: {adaptive_r1:.2f}%")
            else:
                print(f"  [ERROR] Failed to get adaptive R@1")
    
    # Generate comparison charts
    print(f"\n{'='*70}")
    print(f"Generating Comparison Charts")
    print(f"{'='*70}")
    
    # Chart 1: Adaptive R@1 Comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    test_names = list(all_results.keys())
    x = np.arange(len(test_names))
    width = 0.25
    
    model_names = list(models.keys())
    colors = ['blue', 'green', 'red']
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        r1_values = []
        for test_name in test_names:
            if model_name in all_results[test_name]:
                r1_values.append(all_results[test_name][model_name]["adaptive_r1"])
            else:
                r1_values.append(0)
        
        ax1.bar(x + (i - 1) * width, r1_values, width, label=f'Model {i+1}: {model_name}', 
               color=color, alpha=0.7)
    
    # Add full re-ranking line if available
    full_r1_values = []
    for test_name in test_names:
        full_r1 = all_results[test_name].get("full_reranking")
        full_r1_values.append(full_r1 if full_r1 is not None else 0)
    
    if any(v > 0 for v in full_r1_values):
        ax1.plot(x, full_r1_values, 'r--', linewidth=3, marker='*', markersize=15,
                label='Full Re-ranking (Ground Truth)', zorder=10)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Adaptive Re-ranking R@1 Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_adaptive_r1_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Save results table
    table_path = output_dir / "accuracy_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Adaptive Re-ranking R@1 Results\n\n")
        f.write("| Model | Test Set | Adaptive R@1 | Full Re-ranking R@1 | Performance Ratio |\n")
        f.write("|-------|----------|--------------|---------------------|-------------------|\n")
        
        for test_name in test_names:
            for model_name in model_names:
                if model_name in all_results[test_name]:
                    adaptive_r1 = all_results[test_name][model_name]["adaptive_r1"]
                    full_r1 = all_results[test_name].get("full_reranking")
                    ratio = (adaptive_r1 / full_r1 * 100) if full_r1 and full_r1 > 0 else None
                    
                    full_r1_str = f"{full_r1:.2f}%" if full_r1 else "-"
                    ratio_str = f"{ratio:.1f}%" if ratio else "-"
                    
                    f.write(f"| {model_name} | {test_name} | {adaptive_r1:.2f}% | {full_r1_str} | {ratio_str} |\n")
    
    print(f"  Saved: {table_path}")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

Complete evaluation pipeline:
1. Run adaptive image matching for temperature-scaled predictions
2. Run adaptive re-ranking evaluation
3. Compare with full re-ranking
4. Generate accuracy comparison charts
"""

import argparse
import subprocess
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import re
from tqdm import tqdm


def run_adaptive_matching(preds_dir, hard_queries_list, out_dir, matcher="superpoint-lg", device="cuda", num_preds=20):
    """Run adaptive image matching for hard queries only."""
    cmd = [
        sys.executable,
        "match_queries_preds_adaptive.py",
        "--preds-dir", str(preds_dir),
        "--hard-queries-list", str(hard_queries_list),
        "--out-dir", str(out_dir),
        "--matcher", matcher,
        "--device", device,
        "--num-preds", str(num_preds)
    ]
    
    print(f"Running adaptive image matching...")
    print(f"  Input: {preds_dir}")
    print(f"  Hard queries: {hard_queries_list}")
    print(f"  Output: {out_dir}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    
    print(f"  [OK] Adaptive matching complete")
    return True


def run_adaptive_evaluation(preds_dir, inliers_dir, logreg_output_path, num_preds=20, positive_dist_threshold=25):
    """Run adaptive re-ranking evaluation and extract R@1."""
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir", str(preds_dir),
        "--inliers-dir", str(inliers_dir),
        "--logreg-output", str(logreg_output_path),
        "--num-preds", str(num_preds),
        "--positive-dist-threshold", str(positive_dist_threshold),
        "--recall-values", "1", "5", "10", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return None
    
    # Parse R@1 from output
    r1 = None
    for line in result.stdout.split('\n'):
        if 'R@1' in line or 'Recall@1' in line or 'recall@1' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                r1 = float(match.group(1))
                break
    
    return r1


def get_full_reranking_r1(preds_dir, inliers_dir, num_preds=20, positive_dist_threshold=25):
    """Get full re-ranking R@1 (all queries get re-ranking)."""
    # Use reranking.py or stage_5 with all queries as hard
    # For now, we'll try to extract from existing results or compute
    
    # Check if we can use the inliers directory directly
    if not Path(inliers_dir).exists():
        return None
    
    # Try to compute from inliers directory
    # This is a simplified version - full implementation would use reranking.py
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Complete evaluation pipeline with temperature scaling"
    )
    parser.add_argument(
        "--test-set",
        type=str,
        choices=["tokyo", "svox", "both"],
        default="both",
        help="Test set to evaluate"
    )
    parser.add_argument(
        "--tokyo-preds-dir",
        type=str,
        help="Tokyo-XS test predictions directory"
    )
    parser.add_argument(
        "--tokyo-inliers-dir",
        type=str,
        help="Tokyo-XS test inliers directory (full re-ranking)"
    )
    parser.add_argument(
        "--svox-preds-dir",
        type=str,
        help="SVOX test predictions directory"
    )
    parser.add_argument(
        "--svox-inliers-dir",
        type=str,
        help="SVOX test inliers directory (full re-ranking)"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip image matching (assume already done)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_accuracy_comparison",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = {
        "Night + Sun": "night_sun",
        "Night Only": "night_only",
        "Sun Only": "sun_only"
    }
    
    test_configs = {}
    
    if args.test_set in ["tokyo", "both"] and args.tokyo_preds_dir:
        test_configs["Tokyo-XS"] = {
            "preds_dir": Path(args.tokyo_preds_dir),
            "inliers_dir": Path(args.tokyo_inliers_dir) if args.tokyo_inliers_dir else None,
            "test_key": "tokyo_xs"
        }
    
    if args.test_set in ["svox", "both"] and args.svox_preds_dir:
        test_configs["SVOX"] = {
            "preds_dir": Path(args.svox_preds_dir),
            "inliers_dir": Path(args.svox_inliers_dir) if args.svox_inliers_dir else None,
            "test_key": "svox"
        }
    
    # Collect results
    all_results = {}
    
    for test_name, config in test_configs.items():
        print(f"\n{'='*70}")
        print(f"Processing: {test_name}")
        print(f"{'='*70}")
        
        all_results[test_name] = {}
        
        # Get full re-ranking R@1 if available
        full_r1 = None
        if config["inliers_dir"] and config["inliers_dir"].exists():
            # Try to get from existing evaluation or compute
            print(f"\nFull re-ranking inliers directory: {config['inliers_dir']}")
            print(f"  (Full re-ranking R@1 needs to be computed separately)")
        
        all_results[test_name]["full_reranking"] = full_r1
        
        # Evaluate each model
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            # Check if temperature-scaled predictions exist
            logreg_path = Path(f"data/features_and_predictions/logreg_{config['test_key']}_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_{config['test_key']}_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists():
                print(f"  [SKIP] Temperature-scaled predictions not found: {logreg_path}")
                continue
            
            if not hard_queries_path.exists():
                print(f"  [SKIP] Hard queries list not found: {hard_queries_path}")
                continue
            
            # Run adaptive image matching
            adaptive_inliers_dir = config["preds_dir"].parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            if not args.skip_matching:
                if not adaptive_inliers_dir.exists() or len(list(adaptive_inliers_dir.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    success = run_adaptive_matching(
                        config["preds_dir"],
                        hard_queries_path,
                        adaptive_inliers_dir
                    )
                    if not success:
                        print(f"  [ERROR] Adaptive matching failed")
                        continue
                else:
                    print(f"  [SKIP] Adaptive matching already done: {adaptive_inliers_dir}")
            else:
                if not adaptive_inliers_dir.exists():
                    print(f"  [ERROR] Adaptive matching not done and --skip-matching set")
                    continue
            
            # Run adaptive evaluation
            print(f"  Running adaptive re-ranking evaluation...")
            adaptive_r1 = run_adaptive_evaluation(
                config["preds_dir"],
                adaptive_inliers_dir,
                logreg_path
            )
            
            if adaptive_r1 is not None:
                all_results[test_name][model_name] = {
                    "adaptive_r1": adaptive_r1,
                    "hard_queries_path": str(hard_queries_path),
                    "inliers_dir": str(adaptive_inliers_dir)
                }
                print(f"  [OK] Adaptive R@1: {adaptive_r1:.2f}%")
            else:
                print(f"  [ERROR] Failed to get adaptive R@1")
    
    # Generate comparison charts
    print(f"\n{'='*70}")
    print(f"Generating Comparison Charts")
    print(f"{'='*70}")
    
    # Chart 1: Adaptive R@1 Comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    
    test_names = list(all_results.keys())
    x = np.arange(len(test_names))
    width = 0.25
    
    model_names = list(models.keys())
    colors = ['blue', 'green', 'red']
    
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        r1_values = []
        for test_name in test_names:
            if model_name in all_results[test_name]:
                r1_values.append(all_results[test_name][model_name]["adaptive_r1"])
            else:
                r1_values.append(0)
        
        ax1.bar(x + (i - 1) * width, r1_values, width, label=f'Model {i+1}: {model_name}', 
               color=color, alpha=0.7)
    
    # Add full re-ranking line if available
    full_r1_values = []
    for test_name in test_names:
        full_r1 = all_results[test_name].get("full_reranking")
        full_r1_values.append(full_r1 if full_r1 is not None else 0)
    
    if any(v > 0 for v in full_r1_values):
        ax1.plot(x, full_r1_values, 'r--', linewidth=3, marker='*', markersize=15,
                label='Full Re-ranking (Ground Truth)', zorder=10)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Adaptive Re-ranking R@1 Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_adaptive_r1_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Save results table
    table_path = output_dir / "accuracy_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Adaptive Re-ranking R@1 Results\n\n")
        f.write("| Model | Test Set | Adaptive R@1 | Full Re-ranking R@1 | Performance Ratio |\n")
        f.write("|-------|----------|--------------|---------------------|-------------------|\n")
        
        for test_name in test_names:
            for model_name in model_names:
                if model_name in all_results[test_name]:
                    adaptive_r1 = all_results[test_name][model_name]["adaptive_r1"]
                    full_r1 = all_results[test_name].get("full_reranking")
                    ratio = (adaptive_r1 / full_r1 * 100) if full_r1 and full_r1 > 0 else None
                    
                    full_r1_str = f"{full_r1:.2f}%" if full_r1 else "-"
                    ratio_str = f"{ratio:.1f}%" if ratio else "-"
                    
                    f.write(f"| {model_name} | {test_name} | {adaptive_r1:.2f}% | {full_r1_str} | {ratio_str} |\n")
    
    print(f"  Saved: {table_path}")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


