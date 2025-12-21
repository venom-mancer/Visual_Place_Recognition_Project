"""
Generate final accuracy comparison charts with full re-ranking baseline.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate final accuracy comparison with full re-ranking"
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
    
    # Results from evaluations
    results = {
        "Tokyo-XS": {
            "Baseline": 65.1,  # From previous results
            "Full Re-ranking": 83.2,  # From previous results
            "Night + Sun": 75.2,  # From evaluation
            "Night Only": 72.70,
            "Sun Only": 74.00
        },
        "SVOX": {
            "Baseline": 96.3,  # From previous results
            "Full Re-ranking": None,  # May not be available
            "Night + Sun": 96.30,
            "Night Only": 96.40,
            "Sun Only": 96.50
        }
    }
    
    # Try to get Night + Sun for Tokyo-XS from evaluation output
    # This will be updated after running the evaluation
    
    # Generate comprehensive comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Final Models Accuracy Comparison: Adaptive R@1 with Temperature Scaling', 
                fontsize=16, fontweight='bold', y=0.995)
    
    test_names = ["Tokyo-XS", "SVOX"]
    model_names = ["Night + Sun", "Night Only", "Sun Only"]
    x = np.arange(len(test_names))
    width = 0.2
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Chart 1: Adaptive R@1 Comparison
    ax1 = axes[0, 0]
    
    # Baseline
    baseline_values = [results["Tokyo-XS"]["Baseline"], results["SVOX"]["Baseline"]]
    ax1.bar(x - 1.5*width, baseline_values, width, label='Baseline (Retrieval-only)', 
           color='gray', alpha=0.7)
    
    # Full re-ranking
    full_rerank_values = [
        results["Tokyo-XS"]["Full Re-ranking"] if results["Tokyo-XS"]["Full Re-ranking"] else 0,
        results["SVOX"]["Full Re-ranking"] if results["SVOX"]["Full Re-ranking"] else 0
    ]
    if any(v > 0 for v in full_rerank_values):
        ax1.bar(x - 0.5*width, full_rerank_values, width, label='Full Re-ranking', 
               color='orange', alpha=0.7)
    
    # Adaptive models
    for i, model_name in enumerate(model_names):
        r1_values = []
        for test_name in test_names:
            r1 = results[test_name].get(model_name)
            r1_values.append(r1 if r1 else 0)
        
        ax1.bar(x + (i + 0.5) * width, r1_values, width, 
               label=f'Adaptive: {model_name}', color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Adaptive R@1 vs Baseline & Full Re-ranking', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Chart 2: Performance Gain vs Baseline
    ax2 = axes[0, 1]
    
    for i, model_name in enumerate(model_names):
        gains = []
        for test_name in test_names:
            baseline = results[test_name]["Baseline"]
            adaptive = results[test_name].get(model_name)
            if adaptive:
                gains.append(adaptive - baseline)
            else:
                gains.append(0)
        
        ax2.bar(x + (i - 1) * width, gains, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    # Full re-ranking gain
    full_gains = []
    for test_name in test_names:
        baseline = results[test_name]["Baseline"]
        full = results[test_name].get("Full Re-ranking")
        if full:
            full_gains.append(full - baseline)
        else:
            full_gains.append(0)
    
    if any(v > 0 for v in full_gains):
        ax2.bar(x - width, full_gains, width, label='Full Re-ranking', 
               color='orange', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R@1 Gain vs Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Gain vs Baseline', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Performance Ratio (Adaptive / Full Re-ranking)
    ax3 = axes[1, 0]
    
    for i, model_name in enumerate(model_names):
        ratios = []
        for test_name in test_names:
            adaptive = results[test_name].get(model_name)
            full = results[test_name].get("Full Re-ranking")
            if adaptive and full and full > 0:
                ratios.append((adaptive / full) * 100)
            else:
                ratios.append(0)
        
        ax3.bar(x + (i - 1) * width, ratios, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    # Add 100% reference line
    ax3.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% (Full Re-ranking)')
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Ratio (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Adaptive R@1 / Full Re-ranking R@1', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 110])
    
    # Chart 4: Hard Query Detection Rate (from temperature scaling)
    ax4 = axes[1, 1]
    
    # Hard query rates from temperature scaling results
    hard_rates = {
        "Tokyo-XS": {
            "Night + Sun": 24.4,
            "Night Only": 18.7,
            "Sun Only": 22.2
        },
        "SVOX": {
            "Night + Sun": 0.5,
            "Night Only": 0.4,
            "Sun Only": 0.5
        }
    }
    
    for i, model_name in enumerate(model_names):
        rates = []
        for test_name in test_names:
            rate = hard_rates[test_name].get(model_name, 0)
            rates.append(rate)
        
        ax4.bar(x + (i - 1) * width, rates, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    ax4.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Hard Query Detection Rate', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(test_names)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = output_dir / "chart_final_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final comparison chart: {chart_path}")
    
    # Save comprehensive results table
    table_path = output_dir / "final_accuracy_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Final Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Adaptive Re-ranking R@1 Results\n\n")
        f.write("| Model | Test Set | Baseline R@1 | Full Re-ranking R@1 | Adaptive R@1 | Gain vs Baseline | Performance Ratio | Hard Query Rate |\n")
        f.write("|-------|----------|--------------|---------------------|--------------|------------------|-------------------|-----------------|\n")
        
        for test_name in test_names:
            baseline = results[test_name]["Baseline"]
            full = results[test_name].get("Full Re-ranking")
            
            for model_name in model_names:
                adaptive = results[test_name].get(model_name)
                hard_rate = hard_rates[test_name].get(model_name, 0)
                
                if adaptive:
                    gain = adaptive - baseline
                    ratio = (adaptive / full * 100) if full and full > 0 else None
                    ratio_str = f"{ratio:.1f}%" if ratio else "-"
                    
                    full_str = f"{full:.2f}%" if full else "-"
                    
                    f.write(f"| {model_name} | {test_name} | {baseline:.2f}% | {full_str} | "
                           f"{adaptive:.2f}% | {gain:+.2f}% | {ratio_str} | {hard_rate:.1f}% |\n")
    
    print(f"Saved comparison table: {table_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Final Accuracy Comparison Summary")
    print(f"{'='*70}")
    for test_name in test_names:
        print(f"\n{test_name}:")
        print(f"  Baseline: {results[test_name]['Baseline']:.2f}%")
        if results[test_name].get("Full Re-ranking"):
            print(f"  Full Re-ranking: {results[test_name]['Full Re-ranking']:.2f}%")
        for model_name in model_names:
            adaptive = results[test_name].get(model_name)
            if adaptive:
                gain = adaptive - results[test_name]["Baseline"]
                print(f"  {model_name}: {adaptive:.2f}% (gain: {gain:+.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Generate final accuracy comparison with full re-ranking"
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
    
    # Results from evaluations
    results = {
        "Tokyo-XS": {
            "Baseline": 65.1,  # From previous results
            "Full Re-ranking": 83.2,  # From previous results
            "Night + Sun": 75.2,  # From evaluation
            "Night Only": 72.70,
            "Sun Only": 74.00
        },
        "SVOX": {
            "Baseline": 96.3,  # From previous results
            "Full Re-ranking": None,  # May not be available
            "Night + Sun": 96.30,
            "Night Only": 96.40,
            "Sun Only": 96.50
        }
    }
    
    # Try to get Night + Sun for Tokyo-XS from evaluation output
    # This will be updated after running the evaluation
    
    # Generate comprehensive comparison chart
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Final Models Accuracy Comparison: Adaptive R@1 with Temperature Scaling', 
                fontsize=16, fontweight='bold', y=0.995)
    
    test_names = ["Tokyo-XS", "SVOX"]
    model_names = ["Night + Sun", "Night Only", "Sun Only"]
    x = np.arange(len(test_names))
    width = 0.2
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    
    # Chart 1: Adaptive R@1 Comparison
    ax1 = axes[0, 0]
    
    # Baseline
    baseline_values = [results["Tokyo-XS"]["Baseline"], results["SVOX"]["Baseline"]]
    ax1.bar(x - 1.5*width, baseline_values, width, label='Baseline (Retrieval-only)', 
           color='gray', alpha=0.7)
    
    # Full re-ranking
    full_rerank_values = [
        results["Tokyo-XS"]["Full Re-ranking"] if results["Tokyo-XS"]["Full Re-ranking"] else 0,
        results["SVOX"]["Full Re-ranking"] if results["SVOX"]["Full Re-ranking"] else 0
    ]
    if any(v > 0 for v in full_rerank_values):
        ax1.bar(x - 0.5*width, full_rerank_values, width, label='Full Re-ranking', 
               color='orange', alpha=0.7)
    
    # Adaptive models
    for i, model_name in enumerate(model_names):
        r1_values = []
        for test_name in test_names:
            r1 = results[test_name].get(model_name)
            r1_values.append(r1 if r1 else 0)
        
        ax1.bar(x + (i + 0.5) * width, r1_values, width, 
               label=f'Adaptive: {model_name}', color=colors[i], alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Adaptive R@1 vs Baseline & Full Re-ranking', fontsize=13, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(test_names)
    ax1.legend(loc='best', fontsize=9, ncol=2)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 100])
    
    # Chart 2: Performance Gain vs Baseline
    ax2 = axes[0, 1]
    
    for i, model_name in enumerate(model_names):
        gains = []
        for test_name in test_names:
            baseline = results[test_name]["Baseline"]
            adaptive = results[test_name].get(model_name)
            if adaptive:
                gains.append(adaptive - baseline)
            else:
                gains.append(0)
        
        ax2.bar(x + (i - 1) * width, gains, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    # Full re-ranking gain
    full_gains = []
    for test_name in test_names:
        baseline = results[test_name]["Baseline"]
        full = results[test_name].get("Full Re-ranking")
        if full:
            full_gains.append(full - baseline)
        else:
            full_gains.append(0)
    
    if any(v > 0 for v in full_gains):
        ax2.bar(x - width, full_gains, width, label='Full Re-ranking', 
               color='orange', alpha=0.7)
    
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('R@1 Gain vs Baseline (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Performance Gain vs Baseline', fontsize=13, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(test_names)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Chart 3: Performance Ratio (Adaptive / Full Re-ranking)
    ax3 = axes[1, 0]
    
    for i, model_name in enumerate(model_names):
        ratios = []
        for test_name in test_names:
            adaptive = results[test_name].get(model_name)
            full = results[test_name].get("Full Re-ranking")
            if adaptive and full and full > 0:
                ratios.append((adaptive / full) * 100)
            else:
                ratios.append(0)
        
        ax3.bar(x + (i - 1) * width, ratios, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    # Add 100% reference line
    ax3.axhline(y=100, color='green', linestyle='--', linewidth=2, label='100% (Full Re-ranking)')
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Performance Ratio (%)', fontsize=12, fontweight='bold')
    ax3.set_title('Adaptive R@1 / Full Re-ranking R@1', fontsize=13, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(test_names)
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 110])
    
    # Chart 4: Hard Query Detection Rate (from temperature scaling)
    ax4 = axes[1, 1]
    
    # Hard query rates from temperature scaling results
    hard_rates = {
        "Tokyo-XS": {
            "Night + Sun": 24.4,
            "Night Only": 18.7,
            "Sun Only": 22.2
        },
        "SVOX": {
            "Night + Sun": 0.5,
            "Night Only": 0.4,
            "Sun Only": 0.5
        }
    }
    
    for i, model_name in enumerate(model_names):
        rates = []
        for test_name in test_names:
            rate = hard_rates[test_name].get(model_name, 0)
            rates.append(rate)
        
        ax4.bar(x + (i - 1) * width, rates, width, 
               label=f'{model_name}', color=colors[i], alpha=0.7)
    
    ax4.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax4.set_title('Hard Query Detection Rate', fontsize=13, fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(test_names)
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    chart_path = output_dir / "chart_final_accuracy_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved final comparison chart: {chart_path}")
    
    # Save comprehensive results table
    table_path = output_dir / "final_accuracy_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Final Models Accuracy Comparison (Temperature Scaling)\n\n")
        f.write("## Adaptive Re-ranking R@1 Results\n\n")
        f.write("| Model | Test Set | Baseline R@1 | Full Re-ranking R@1 | Adaptive R@1 | Gain vs Baseline | Performance Ratio | Hard Query Rate |\n")
        f.write("|-------|----------|--------------|---------------------|--------------|------------------|-------------------|-----------------|\n")
        
        for test_name in test_names:
            baseline = results[test_name]["Baseline"]
            full = results[test_name].get("Full Re-ranking")
            
            for model_name in model_names:
                adaptive = results[test_name].get(model_name)
                hard_rate = hard_rates[test_name].get(model_name, 0)
                
                if adaptive:
                    gain = adaptive - baseline
                    ratio = (adaptive / full * 100) if full and full > 0 else None
                    ratio_str = f"{ratio:.1f}%" if ratio else "-"
                    
                    full_str = f"{full:.2f}%" if full else "-"
                    
                    f.write(f"| {model_name} | {test_name} | {baseline:.2f}% | {full_str} | "
                           f"{adaptive:.2f}% | {gain:+.2f}% | {ratio_str} | {hard_rate:.1f}% |\n")
    
    print(f"Saved comparison table: {table_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print(f"Final Accuracy Comparison Summary")
    print(f"{'='*70}")
    for test_name in test_names:
        print(f"\n{test_name}:")
        print(f"  Baseline: {results[test_name]['Baseline']:.2f}%")
        if results[test_name].get("Full Re-ranking"):
            print(f"  Full Re-ranking: {results[test_name]['Full Re-ranking']:.2f}%")
        for model_name in model_names:
            adaptive = results[test_name].get(model_name)
            if adaptive:
                gain = adaptive - results[test_name]["Baseline"]
                print(f"  {model_name}: {adaptive:.2f}% (gain: {gain:+.2f}%)")
    
    print(f"\n{'='*70}")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

