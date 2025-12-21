"""
Compare all 3 models on all test sets using temperature scaling.

This creates updated comparison charts showing the improved results
after applying temperature scaling to fix model overconfidence.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file


def load_temp_scaled_results(model_name, test_set):
    """Load temperature-scaled results."""
    base_path = Path("data/features_and_predictions")
    
    model_map = {
        "Night + Sun": ["night_sun", "temperature_scaled"],  # Try both naming patterns
        "Night Only": "night_only",
        "Sun Only": "sun_only"
    }
    
    test_map = {
        "tokyo": "tokyo_xs",
        "svox": "svox"
    }
    
    # Try different file naming patterns
    if isinstance(model_map[model_name], list):
        # Night + Sun might use different naming
        patterns = [
            f"logreg_{test_map[test_set]}_{model_map[model_name][0]}_temp_scaled.npz",
            f"logreg_{test_map[test_set]}_{model_map[model_name][1]}.npz",  # temperature_scaled
        ]
    else:
        patterns = [f"logreg_{test_map[test_set]}_{model_map[model_name]}_temp_scaled.npz"]
    
    file_path = None
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            file_path = candidate
            break
    
    if not file_path.exists():
        return None
    
    data = np.load(file_path)
    return {
        "hard_rate": data["is_hard"].mean(),
        "actual_hard_rate": (1 - data["labels"].mean()),
        "accuracy": data.get("accuracy", (data["is_easy"] == data["labels"]).mean()),
        "f1": data.get("f1", 0.0),
        "precision": data.get("precision", 0.0),
        "recall": data.get("recall", 0.0),
        "num_queries": len(data["labels"]),
        "temperature": data.get("temperature", 0.0),
        "threshold": data.get("threshold", 0.0)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all models with temperature scaling on all test sets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_test_comparison_temp_scaled",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["Night + Sun", "Night Only", "Sun Only"]
    test_sets = ["tokyo", "svox"]
    test_names = {"tokyo": "Tokyo-XS Test", "svox": "SVOX Test"}
    
    # Collect all results
    results = {}
    for model in models:
        results[model] = {}
        for test_set in test_sets:
            result = load_temp_scaled_results(model, test_set)
            results[model][test_set] = result
    
    # Create comparison table
    table_path = output_dir / "models_comparison_temp_scaled.md"
    with open(table_path, 'w') as f:
        f.write("# Models Comparison with Temperature Scaling\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Test Set | # Queries | Hard Query Rate | Actual Hard Rate | Accuracy | F1-Score | Temperature | Threshold |\n")
        f.write("|-------|----------|-----------|-----------------|------------------|----------|----------|-------------|-----------|\n")
        
        for model in models:
            for test_set in test_sets:
                r = results[model][test_set]
                if r is not None:
                    f.write(f"| {model} | {test_names[test_set]} | {r['num_queries']} | "
                           f"{r['hard_rate']:.1%} | {r['actual_hard_rate']:.1%} | "
                           f"{r['accuracy']:.1%} | {r['f1']:.4f} | {r['temperature']:.2f} | {r['threshold']:.3f} |\n")
                else:
                    f.write(f"| {model} | {test_names[test_set]} | - | - | - | - | - | - | - |\n")
    
    print(f"Saved comparison table to: {table_path}")
    
    # Create comparison charts
    print("\nGenerating comparison charts...")
    
    # Chart 1: Hard Query Rate Comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(test_sets))
    width = 0.25
    
    for i, model in enumerate(models):
        hard_rates = []
        for test_set in test_sets:
            r = results[model][test_set]
            hard_rates.append(r['hard_rate'] * 100 if r is not None else 0)
        
        ax1.bar(x + (i - 1) * width, hard_rates, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    # Add actual hard rates
    for i, test_set in enumerate(test_sets):
        r = results[models[0]][test_set]
        if r is not None:
            ax1.axhline(y=r['actual_hard_rate'] * 100, xmin=(i-0.4)/len(test_sets), 
                       xmax=(i+0.4)/len(test_sets), color='red', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Hard Query Rate Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([test_names[ts] for ts in test_sets])
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_hard_query_rate_temp_scaled.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Chart 2: F1-Score Comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        f1_scores = []
        for test_set in test_sets:
            r = results[model][test_set]
            f1_scores.append(r['f1'] if r is not None else 0)
        
        ax2.bar(x + (i - 1) * width, f1_scores, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([test_names[ts] for ts in test_sets])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart2_path = output_dir / "chart_f1_score_temp_scaled.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart2_path}")
    
    # Chart 3: Accuracy Comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        accuracies = []
        for test_set in test_sets:
            r = results[model][test_set]
            accuracies.append(r['accuracy'] * 100 if r is not None else 0)
        
        ax3.bar(x + (i - 1) * width, accuracies, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Classification Accuracy Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([test_names[ts] for ts in test_sets])
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 100])
    
    plt.tight_layout()
    chart3_path = output_dir / "chart_accuracy_temp_scaled.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart3_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Comparison Complete: Temperature Scaling Results")
    print(f"{'='*70}")
    for model in models:
        print(f"\n{model}:")
        for test_set in test_sets:
            r = results[model][test_set]
            if r is not None:
                print(f"  {test_names[test_set]}:")
                print(f"    Hard Query Rate: {r['hard_rate']:.1%} (actual: {r['actual_hard_rate']:.1%})")
                print(f"    Accuracy: {r['accuracy']:.1%}")
                print(f"    F1-Score: {r['f1']:.4f}")
                print(f"    Temperature: {r['temperature']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



This creates updated comparison charts showing the improved results
after applying temperature scaling to fix model overconfidence.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file


def load_temp_scaled_results(model_name, test_set):
    """Load temperature-scaled results."""
    base_path = Path("data/features_and_predictions")
    
    model_map = {
        "Night + Sun": ["night_sun", "temperature_scaled"],  # Try both naming patterns
        "Night Only": "night_only",
        "Sun Only": "sun_only"
    }
    
    test_map = {
        "tokyo": "tokyo_xs",
        "svox": "svox"
    }
    
    # Try different file naming patterns
    if isinstance(model_map[model_name], list):
        # Night + Sun might use different naming
        patterns = [
            f"logreg_{test_map[test_set]}_{model_map[model_name][0]}_temp_scaled.npz",
            f"logreg_{test_map[test_set]}_{model_map[model_name][1]}.npz",  # temperature_scaled
        ]
    else:
        patterns = [f"logreg_{test_map[test_set]}_{model_map[model_name]}_temp_scaled.npz"]
    
    file_path = None
    for pattern in patterns:
        candidate = base_path / pattern
        if candidate.exists():
            file_path = candidate
            break
    
    if not file_path.exists():
        return None
    
    data = np.load(file_path)
    return {
        "hard_rate": data["is_hard"].mean(),
        "actual_hard_rate": (1 - data["labels"].mean()),
        "accuracy": data.get("accuracy", (data["is_easy"] == data["labels"]).mean()),
        "f1": data.get("f1", 0.0),
        "precision": data.get("precision", 0.0),
        "recall": data.get("recall", 0.0),
        "num_queries": len(data["labels"]),
        "temperature": data.get("temperature", 0.0),
        "threshold": data.get("threshold", 0.0)
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare all models with temperature scaling on all test sets"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_test_comparison_temp_scaled",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    models = ["Night + Sun", "Night Only", "Sun Only"]
    test_sets = ["tokyo", "svox"]
    test_names = {"tokyo": "Tokyo-XS Test", "svox": "SVOX Test"}
    
    # Collect all results
    results = {}
    for model in models:
        results[model] = {}
        for test_set in test_sets:
            result = load_temp_scaled_results(model, test_set)
            results[model][test_set] = result
    
    # Create comparison table
    table_path = output_dir / "models_comparison_temp_scaled.md"
    with open(table_path, 'w') as f:
        f.write("# Models Comparison with Temperature Scaling\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Test Set | # Queries | Hard Query Rate | Actual Hard Rate | Accuracy | F1-Score | Temperature | Threshold |\n")
        f.write("|-------|----------|-----------|-----------------|------------------|----------|----------|-------------|-----------|\n")
        
        for model in models:
            for test_set in test_sets:
                r = results[model][test_set]
                if r is not None:
                    f.write(f"| {model} | {test_names[test_set]} | {r['num_queries']} | "
                           f"{r['hard_rate']:.1%} | {r['actual_hard_rate']:.1%} | "
                           f"{r['accuracy']:.1%} | {r['f1']:.4f} | {r['temperature']:.2f} | {r['threshold']:.3f} |\n")
                else:
                    f.write(f"| {model} | {test_names[test_set]} | - | - | - | - | - | - | - |\n")
    
    print(f"Saved comparison table to: {table_path}")
    
    # Create comparison charts
    print("\nGenerating comparison charts...")
    
    # Chart 1: Hard Query Rate Comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(test_sets))
    width = 0.25
    
    for i, model in enumerate(models):
        hard_rates = []
        for test_set in test_sets:
            r = results[model][test_set]
            hard_rates.append(r['hard_rate'] * 100 if r is not None else 0)
        
        ax1.bar(x + (i - 1) * width, hard_rates, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    # Add actual hard rates
    for i, test_set in enumerate(test_sets):
        r = results[models[0]][test_set]
        if r is not None:
            ax1.axhline(y=r['actual_hard_rate'] * 100, xmin=(i-0.4)/len(test_sets), 
                       xmax=(i+0.4)/len(test_sets), color='red', linestyle='--', linewidth=2)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Hard Query Rate Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([test_names[ts] for ts in test_sets])
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_hard_query_rate_temp_scaled.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Chart 2: F1-Score Comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        f1_scores = []
        for test_set in test_sets:
            r = results[model][test_set]
            f1_scores.append(r['f1'] if r is not None else 0)
        
        ax2.bar(x + (i - 1) * width, f1_scores, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax2.set_title('F1-Score Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels([test_names[ts] for ts in test_sets])
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart2_path = output_dir / "chart_f1_score_temp_scaled.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart2_path}")
    
    # Chart 3: Accuracy Comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    for i, model in enumerate(models):
        accuracies = []
        for test_set in test_sets:
            r = results[model][test_set]
            accuracies.append(r['accuracy'] * 100 if r is not None else 0)
        
        ax3.bar(x + (i - 1) * width, accuracies, width, label=f'Model {i+1}: {model}', alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Classification Accuracy Comparison: All Models with Temperature Scaling', 
                 fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([test_names[ts] for ts in test_sets])
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 100])
    
    plt.tight_layout()
    chart3_path = output_dir / "chart_accuracy_temp_scaled.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart3_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Comparison Complete: Temperature Scaling Results")
    print(f"{'='*70}")
    for model in models:
        print(f"\n{model}:")
        for test_set in test_sets:
            r = results[model][test_set]
            if r is not None:
                print(f"  {test_names[test_set]}:")
                print(f"    Hard Query Rate: {r['hard_rate']:.1%} (actual: {r['actual_hard_rate']:.1%})")
                print(f"    Accuracy: {r['accuracy']:.1%}")
                print(f"    F1-Score: {r['f1']:.4f}")
                print(f"    Temperature: {r['temperature']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

