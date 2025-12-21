"""
Complete evaluation pipeline for temperature-scaled models:
1. Run adaptive image matching (if needed)
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


def run_adaptive_matching(preds_dir, hard_queries_list, out_dir):
    """Run adaptive image matching."""
    cmd = [
        sys.executable,
        "match_queries_preds_adaptive.py",
        "--preds-dir", str(preds_dir),
        "--hard-queries-list", str(hard_queries_list),
        "--out-dir", str(out_dir),
        "--matcher", "superpoint-lg",
        "--device", "cuda",
        "--num-preds", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_evaluation(preds_dir, inliers_dir, logreg_output):
    """Run adaptive re-ranking evaluation and extract R@1."""
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir", str(preds_dir),
        "--inliers-dir", str(inliers_dir),
        "--logreg-output", str(logreg_output),
        "--num-preds", "20",
        "--positive-dist-threshold", "25",
        "--recall-values", "1", "5", "10", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None
    
    # Parse R@1
    for line in result.stdout.split('\n'):
        if 'R@1' in line or 'recall@1' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Complete evaluation with temperature scaling"
    )
    parser.add_argument(
        "--tokyo-preds-dir",
        type=str,
        default="log_tokyo_xs_test/2025-12-18_14-24-37/preds",
        help="Tokyo-XS predictions directory"
    )
    parser.add_argument(
        "--tokyo-inliers-dir",
        type=str,
        help="Tokyo-XS full re-ranking inliers directory"
    )
    parser.add_argument(
        "--svox-preds-dir",
        type=str,
        default="log_svox_test/2025-12-18_16-01-59/preds",
        help="SVOX predictions directory"
    )
    parser.add_argument(
        "--svox-inliers-dir",
        type=str,
        help="SVOX full re-ranking inliers directory"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip image matching (assume done)"
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
    
    results = {}
    
    # Tokyo-XS Test
    tokyo_preds = Path(args.tokyo_preds_dir)
    if tokyo_preds.exists():
        print(f"\n{'='*70}")
        print(f"Tokyo-XS Test Evaluation")
        print(f"{'='*70}")
        
        results["Tokyo-XS"] = {}
        
        # Get full re-ranking R@1 if available
        if args.tokyo_inliers_dir and Path(args.tokyo_inliers_dir).exists():
            print(f"\nFull re-ranking inliers: {args.tokyo_inliers_dir}")
            # Would need to run full re-ranking evaluation
            results["Tokyo-XS"]["full_reranking"] = None
        else:
            results["Tokyo-XS"]["full_reranking"] = None
        
        # Evaluate each model
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            logreg_path = Path(f"data/features_and_predictions/logreg_tokyo_xs_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_tokyo_xs_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists() or not hard_queries_path.exists():
                print(f"  [SKIP] Files not found")
                continue
            
            # Adaptive inliers directory
            adaptive_inliers = tokyo_preds.parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            # Run adaptive matching if needed
            if not args.skip_matching:
                if not adaptive_inliers.exists() or len(list(adaptive_inliers.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    if run_adaptive_matching(tokyo_preds, hard_queries_path, adaptive_inliers):
                        print(f"  [OK] Matching complete")
                    else:
                        print(f"  [ERROR] Matching failed")
                        continue
                else:
                    print(f"  [SKIP] Matching already done")
            
            # Run evaluation
            print(f"  Running evaluation...")
            r1 = run_evaluation(tokyo_preds, adaptive_inliers, logreg_path)
            
            if r1 is not None:
                results["Tokyo-XS"][model_name] = r1
                print(f"  [OK] Adaptive R@1: {r1:.2f}%")
            else:
                print(f"  [ERROR] Evaluation failed")
    
    # SVOX Test
    svox_preds = Path(args.svox_preds_dir)
    if svox_preds.exists():
        print(f"\n{'='*70}")
        print(f"SVOX Test Evaluation")
        print(f"{'='*70}")
        
        results["SVOX"] = {}
        
        if args.svox_inliers_dir and Path(args.svox_inliers_dir).exists():
            results["SVOX"]["full_reranking"] = None
        else:
            results["SVOX"]["full_reranking"] = None
        
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            logreg_path = Path(f"data/features_and_predictions/logreg_svox_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_svox_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists() or not hard_queries_path.exists():
                print(f"  [SKIP] Files not found")
                continue
            
            adaptive_inliers = svox_preds.parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            if not args.skip_matching:
                if not adaptive_inliers.exists() or len(list(adaptive_inliers.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    if run_adaptive_matching(svox_preds, hard_queries_path, adaptive_inliers):
                        print(f"  [OK] Matching complete")
                    else:
                        print(f"  [ERROR] Matching failed")
                        continue
                else:
                    print(f"  [SKIP] Matching already done")
            
            print(f"  Running evaluation...")
            r1 = run_evaluation(svox_preds, adaptive_inliers, logreg_path)
            
            if r1 is not None:
                results["SVOX"][model_name] = r1
                print(f"  [OK] Adaptive R@1: {r1:.2f}%")
            else:
                print(f"  [ERROR] Evaluation failed")
    
    # Generate charts
    print(f"\n{'='*70}")
    print(f"Generating Comparison Charts")
    print(f"{'='*70}")
    
    test_names = [tn for tn in results.keys() if tn != "full_reranking"]
    
    if len(test_names) > 0:
        # Chart: Adaptive R@1 Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(test_names))
        width = 0.25
        
        model_names = list(models.keys())
        colors = ['blue', 'green', 'red']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            r1_values = []
            for test_name in test_names:
                r1 = results[test_name].get(model_name, 0)
                r1_values.append(r1)
            
            ax.bar(x + (i - 1) * width, r1_values, width, 
                  label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
        
        ax.set_xlabel('Test Set', fontsize=12, fontweight='bold')
        ax.set_ylabel('Adaptive Re-ranking Recall@1 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Models Accuracy Comparison: Adaptive R@1 with Temperature Scaling', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_path = output_dir / "chart_models_accuracy_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart_path}")
        
        # Save results table
        table_path = output_dir / "accuracy_comparison_table.md"
        with open(table_path, 'w') as f:
            f.write("# Models Accuracy Comparison (Temperature Scaling)\n\n")
            f.write("## Adaptive Re-ranking R@1 Results\n\n")
            f.write("| Model | Test Set | Adaptive R@1 |\n")
            f.write("|-------|----------|--------------|\n")
            
            for test_name in test_names:
                for model_name in model_names:
                    r1 = results[test_name].get(model_name)
                    if r1 is not None:
                        f.write(f"| {model_name} | {test_name} | {r1:.2f}% |\n")
        
        print(f"  Saved: {table_path}")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

Complete evaluation pipeline for temperature-scaled models:
1. Run adaptive image matching (if needed)
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


def run_adaptive_matching(preds_dir, hard_queries_list, out_dir):
    """Run adaptive image matching."""
    cmd = [
        sys.executable,
        "match_queries_preds_adaptive.py",
        "--preds-dir", str(preds_dir),
        "--hard-queries-list", str(hard_queries_list),
        "--out-dir", str(out_dir),
        "--matcher", "superpoint-lg",
        "--device", "cuda",
        "--num-preds", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.returncode == 0


def run_evaluation(preds_dir, inliers_dir, logreg_output):
    """Run adaptive re-ranking evaluation and extract R@1."""
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_5_adaptive_reranking_eval",
        "--preds-dir", str(preds_dir),
        "--inliers-dir", str(inliers_dir),
        "--logreg-output", str(logreg_output),
        "--num-preds", "20",
        "--positive-dist-threshold", "25",
        "--recall-values", "1", "5", "10", "20"
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        return None
    
    # Parse R@1
    for line in result.stdout.split('\n'):
        if 'R@1' in line or 'recall@1' in line.lower():
            match = re.search(r'(\d+\.\d+)', line)
            if match:
                return float(match.group(1))
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Complete evaluation with temperature scaling"
    )
    parser.add_argument(
        "--tokyo-preds-dir",
        type=str,
        default="log_tokyo_xs_test/2025-12-18_14-24-37/preds",
        help="Tokyo-XS predictions directory"
    )
    parser.add_argument(
        "--tokyo-inliers-dir",
        type=str,
        help="Tokyo-XS full re-ranking inliers directory"
    )
    parser.add_argument(
        "--svox-preds-dir",
        type=str,
        default="log_svox_test/2025-12-18_16-01-59/preds",
        help="SVOX predictions directory"
    )
    parser.add_argument(
        "--svox-inliers-dir",
        type=str,
        help="SVOX full re-ranking inliers directory"
    )
    parser.add_argument(
        "--skip-matching",
        action="store_true",
        help="Skip image matching (assume done)"
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
    
    results = {}
    
    # Tokyo-XS Test
    tokyo_preds = Path(args.tokyo_preds_dir)
    if tokyo_preds.exists():
        print(f"\n{'='*70}")
        print(f"Tokyo-XS Test Evaluation")
        print(f"{'='*70}")
        
        results["Tokyo-XS"] = {}
        
        # Get full re-ranking R@1 if available
        if args.tokyo_inliers_dir and Path(args.tokyo_inliers_dir).exists():
            print(f"\nFull re-ranking inliers: {args.tokyo_inliers_dir}")
            # Would need to run full re-ranking evaluation
            results["Tokyo-XS"]["full_reranking"] = None
        else:
            results["Tokyo-XS"]["full_reranking"] = None
        
        # Evaluate each model
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            logreg_path = Path(f"data/features_and_predictions/logreg_tokyo_xs_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_tokyo_xs_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists() or not hard_queries_path.exists():
                print(f"  [SKIP] Files not found")
                continue
            
            # Adaptive inliers directory
            adaptive_inliers = tokyo_preds.parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            # Run adaptive matching if needed
            if not args.skip_matching:
                if not adaptive_inliers.exists() or len(list(adaptive_inliers.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    if run_adaptive_matching(tokyo_preds, hard_queries_path, adaptive_inliers):
                        print(f"  [OK] Matching complete")
                    else:
                        print(f"  [ERROR] Matching failed")
                        continue
                else:
                    print(f"  [SKIP] Matching already done")
            
            # Run evaluation
            print(f"  Running evaluation...")
            r1 = run_evaluation(tokyo_preds, adaptive_inliers, logreg_path)
            
            if r1 is not None:
                results["Tokyo-XS"][model_name] = r1
                print(f"  [OK] Adaptive R@1: {r1:.2f}%")
            else:
                print(f"  [ERROR] Evaluation failed")
    
    # SVOX Test
    svox_preds = Path(args.svox_preds_dir)
    if svox_preds.exists():
        print(f"\n{'='*70}")
        print(f"SVOX Test Evaluation")
        print(f"{'='*70}")
        
        results["SVOX"] = {}
        
        if args.svox_inliers_dir and Path(args.svox_inliers_dir).exists():
            results["SVOX"]["full_reranking"] = None
        else:
            results["SVOX"]["full_reranking"] = None
        
        for model_name, model_key in models.items():
            print(f"\n{model_name}:")
            
            logreg_path = Path(f"data/features_and_predictions/logreg_svox_{model_key}_temp_scaled.npz")
            hard_queries_path = Path(f"data/features_and_predictions/hard_queries_svox_{model_key}_temp_scaled.txt")
            
            if not logreg_path.exists() or not hard_queries_path.exists():
                print(f"  [SKIP] Files not found")
                continue
            
            adaptive_inliers = svox_preds.parent / f"preds_superpoint-lg_{model_key}_temp_scaled"
            
            if not args.skip_matching:
                if not adaptive_inliers.exists() or len(list(adaptive_inliers.glob("*.torch"))) == 0:
                    print(f"  Running adaptive image matching...")
                    if run_adaptive_matching(svox_preds, hard_queries_path, adaptive_inliers):
                        print(f"  [OK] Matching complete")
                    else:
                        print(f"  [ERROR] Matching failed")
                        continue
                else:
                    print(f"  [SKIP] Matching already done")
            
            print(f"  Running evaluation...")
            r1 = run_evaluation(svox_preds, adaptive_inliers, logreg_path)
            
            if r1 is not None:
                results["SVOX"][model_name] = r1
                print(f"  [OK] Adaptive R@1: {r1:.2f}%")
            else:
                print(f"  [ERROR] Evaluation failed")
    
    # Generate charts
    print(f"\n{'='*70}")
    print(f"Generating Comparison Charts")
    print(f"{'='*70}")
    
    test_names = [tn for tn in results.keys() if tn != "full_reranking"]
    
    if len(test_names) > 0:
        # Chart: Adaptive R@1 Comparison
        fig, ax = plt.subplots(figsize=(14, 8))
        
        x = np.arange(len(test_names))
        width = 0.25
        
        model_names = list(models.keys())
        colors = ['blue', 'green', 'red']
        
        for i, (model_name, color) in enumerate(zip(model_names, colors)):
            r1_values = []
            for test_name in test_names:
                r1 = results[test_name].get(model_name, 0)
                r1_values.append(r1)
            
            ax.bar(x + (i - 1) * width, r1_values, width, 
                  label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
        
        ax.set_xlabel('Test Set', fontsize=12, fontweight='bold')
        ax.set_ylabel('Adaptive Re-ranking Recall@1 (%)', fontsize=12, fontweight='bold')
        ax.set_title('Models Accuracy Comparison: Adaptive R@1 with Temperature Scaling', 
                    fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(test_names)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        chart_path = output_dir / "chart_models_accuracy_comparison.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart_path}")
        
        # Save results table
        table_path = output_dir / "accuracy_comparison_table.md"
        with open(table_path, 'w') as f:
            f.write("# Models Accuracy Comparison (Temperature Scaling)\n\n")
            f.write("## Adaptive Re-ranking R@1 Results\n\n")
            f.write("| Model | Test Set | Adaptive R@1 |\n")
            f.write("|-------|----------|--------------|\n")
            
            for test_name in test_names:
                for model_name in model_names:
                    r1 = results[test_name].get(model_name)
                    if r1 is not None:
                        f.write(f"| {model_name} | {test_name} | {r1:.2f}% |\n")
        
        print(f"  Saved: {table_path}")
    
    print(f"\n{'='*70}")
    print(f"Evaluation Complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


