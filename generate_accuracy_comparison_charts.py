"""
Generate accuracy comparison charts for all models with temperature scaling.

This script creates charts comparing:
1. Classification accuracy (from temperature-scaled predictions)
2. Hard query detection rates
3. F1-scores
4. Adaptive R@1 (if available from evaluations)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_temp_scaled_results(test_set, model_key):
    """Load temperature-scaled results."""
    base_path = Path("data/features_and_predictions")
    
    # Try different naming patterns
    patterns = [
        f"logreg_{test_set}_{model_key}_temp_scaled.npz",
        f"logreg_{test_set}_temperature_scaled.npz",  # For night_sun
    ]
    
    for pattern in patterns:
        file_path = base_path / pattern
        if file_path.exists():
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
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate accuracy comparison charts for all models"
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
    
    test_sets = {
        "Tokyo-XS": "tokyo_xs",
        "SVOX": "svox"
    }
    
    # Collect all results
    results = {}
    for test_name, test_key in test_sets.items():
        results[test_name] = {}
        for model_name, model_key in models.items():
            result = load_temp_scaled_results(test_key, model_key)
            results[test_name][model_name] = result
    
    # Generate comprehensive comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Models Accuracy Comparison: All Metrics with Temperature Scaling', 
                fontsize=16, fontweight='bold', y=0.995)
    
    test_names = list(test_sets.keys())
    model_names = list(models.keys())
    x = np.arange(len(test_names))
    width = 0.25
    colors = ['blue', 'green', 'red']
    
    # Chart 1: Classification Accuracy
    ax1 = axes[0, 0]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        accuracies = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            accuracies.append(r['accuracy'] * 100 if r else 0)
        
        ax1.bar(x + (i - 1) * width, accuracies, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Chart 2: Hard Query Detection Rate
    ax2 = axes[0, 1]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        hard_rates = []
        actual_rates = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            if r:
                hard_rates.append(r['hard_rate'] * 100)
                actual_rates.append(r['actual_hard_rate'] * 100)
            else:
                hard_rates.append(0)
                actual_rates.append(0)
        
        ax2.bar(x + (i - 1) * width, hard_rates, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    # Add actual hard rate line
    if len(actual_rates) > 0:
        ax2.plot(x, actual_rates[:len(test_names)], 'r--', linewidth=2, marker='*', 
                markersize=12, label='Actual Hard Rate', zorder=10)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Hard Query Detection Rate', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: F1-Score
    ax3 = axes[1, 0]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        f1_scores = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            f1_scores.append(r['f1'] if r else 0)
        
        ax3.bar(x + (i - 1) * width, f1_scores, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.0])
    
    # Chart 4: Temperature Used
    ax4 = axes[1, 1]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        temperatures = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            temperatures.append(r['temperature'] if r else 0)
        
        ax4.bar(x + (i - 1) * width, temperatures, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax4.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Optimal Temperature', fontsize=12, fontweight='bold')
    ax4.set_title('Temperature Scaling Parameter', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(test_names)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = output_dir / "chart_comprehensive_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive chart: {chart_path}")
    
    # Create summary table
    table_path = output_dir / "comprehensive_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Comprehensive Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Test Set | # Queries | Accuracy | F1-Score | Hard Rate | Actual Hard | Temperature |\n")
        f.write("|-------|----------|-----------|----------|----------|-----------|-------------|-------------|\n")
        
        for test_name in test_names:
            for model_name in model_names:
                r = results[test_name].get(model_name)
                if r:
                    f.write(f"| {model_name} | {test_name} | {r['num_queries']} | "
                           f"{r['accuracy']:.1%} | {r['f1']:.4f} | {r['hard_rate']:.1%} | "
                           f"{r['actual_hard_rate']:.1%} | {r['temperature']:.2f} |\n")
    
    print(f"Saved comparison table: {table_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Comprehensive Accuracy Comparison")
    print(f"{'='*70}")
    for test_name in test_names:
        print(f"\n{test_name}:")
        for model_name in model_names:
            r = results[test_name].get(model_name)
            if r:
                print(f"  {model_name}:")
                print(f"    Accuracy: {r['accuracy']:.1%}")
                print(f"    F1-Score: {r['f1']:.4f}")
                print(f"    Hard Rate: {r['hard_rate']:.1%} (actual: {r['actual_hard_rate']:.1%})")
                print(f"    Temperature: {r['temperature']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

Generate accuracy comparison charts for all models with temperature scaling.

This script creates charts comparing:
1. Classification accuracy (from temperature-scaled predictions)
2. Hard query detection rates
3. F1-scores
4. Adaptive R@1 (if available from evaluations)
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def load_temp_scaled_results(test_set, model_key):
    """Load temperature-scaled results."""
    base_path = Path("data/features_and_predictions")
    
    # Try different naming patterns
    patterns = [
        f"logreg_{test_set}_{model_key}_temp_scaled.npz",
        f"logreg_{test_set}_temperature_scaled.npz",  # For night_sun
    ]
    
    for pattern in patterns:
        file_path = base_path / pattern
        if file_path.exists():
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
    
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Generate accuracy comparison charts for all models"
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
    
    test_sets = {
        "Tokyo-XS": "tokyo_xs",
        "SVOX": "svox"
    }
    
    # Collect all results
    results = {}
    for test_name, test_key in test_sets.items():
        results[test_name] = {}
        for model_name, model_key in models.items():
            result = load_temp_scaled_results(test_key, model_key)
            results[test_name][model_name] = result
    
    # Generate comprehensive comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Models Accuracy Comparison: All Metrics with Temperature Scaling', 
                fontsize=16, fontweight='bold', y=0.995)
    
    test_names = list(test_sets.keys())
    model_names = list(models.keys())
    x = np.arange(len(test_names))
    width = 0.25
    colors = ['blue', 'green', 'red']
    
    # Chart 1: Classification Accuracy
    ax1 = axes[0, 0]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        accuracies = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            accuracies.append(r['accuracy'] * 100 if r else 0)
        
        ax1.bar(x + (i - 1) * width, accuracies, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Classification Accuracy (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Classification Accuracy', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Chart 2: Hard Query Detection Rate
    ax2 = axes[0, 1]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        hard_rates = []
        actual_rates = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            if r:
                hard_rates.append(r['hard_rate'] * 100)
                actual_rates.append(r['actual_hard_rate'] * 100)
            else:
                hard_rates.append(0)
                actual_rates.append(0)
        
        ax2.bar(x + (i - 1) * width, hard_rates, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    # Add actual hard rate line
    if len(actual_rates) > 0:
        ax2.plot(x, actual_rates[:len(test_names)], 'r--', linewidth=2, marker='*', 
                markersize=12, label='Actual Hard Rate', zorder=10)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Hard Query Detection Rate', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: F1-Score
    ax3 = axes[1, 0]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        f1_scores = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            f1_scores.append(r['f1'] if r else 0)
        
        ax3.bar(x + (i - 1) * width, f1_scores, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax3.set_title('F1-Score Comparison', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.0])
    
    # Chart 4: Temperature Used
    ax4 = axes[1, 1]
    for i, (model_name, color) in enumerate(zip(model_names, colors)):
        temperatures = []
        for test_name in test_names:
            r = results[test_name].get(model_name)
            temperatures.append(r['temperature'] if r else 0)
        
        ax4.bar(x + (i - 1) * width, temperatures, width, 
               label=f'Model {i+1}: {model_name}', color=color, alpha=0.7)
    
    ax4.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Optimal Temperature', fontsize=12, fontweight='bold')
    ax4.set_title('Temperature Scaling Parameter', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(test_names)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = output_dir / "chart_comprehensive_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comprehensive chart: {chart_path}")
    
    # Create summary table
    table_path = output_dir / "comprehensive_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Comprehensive Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Results Summary\n\n")
        f.write("| Model | Test Set | # Queries | Accuracy | F1-Score | Hard Rate | Actual Hard | Temperature |\n")
        f.write("|-------|----------|-----------|----------|----------|-----------|-------------|-------------|\n")
        
        for test_name in test_names:
            for model_name in model_names:
                r = results[test_name].get(model_name)
                if r:
                    f.write(f"| {model_name} | {test_name} | {r['num_queries']} | "
                           f"{r['accuracy']:.1%} | {r['f1']:.4f} | {r['hard_rate']:.1%} | "
                           f"{r['actual_hard_rate']:.1%} | {r['temperature']:.2f} |\n")
    
    print(f"Saved comparison table: {table_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Comprehensive Accuracy Comparison")
    print(f"{'='*70}")
    for test_name in test_names:
        print(f"\n{test_name}:")
        for model_name in model_names:
            r = results[test_name].get(model_name)
            if r:
                print(f"  {model_name}:")
                print(f"    Accuracy: {r['accuracy']:.1%}")
                print(f"    F1-Score: {r['f1']:.4f}")
                print(f"    Hard Rate: {r['hard_rate']:.1%} (actual: {r['actual_hard_rate']:.1%})")
                print(f"    Temperature: {r['temperature']:.2f}")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


