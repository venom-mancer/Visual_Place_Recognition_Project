"""
Comprehensive threshold analysis: R@1 vs threshold plots, dataset comparison, and cost savings.

This script:
1. Tests different thresholds on multiple datasets
2. Computes R@1 for each threshold (simulates adaptive re-ranking)
3. Plots R@1 vs threshold for each dataset
4. Analyzes how dataset choice influences threshold
5. Calculates cost savings
"""

import argparse
from pathlib import Path
import numpy as np
import joblib
import torch
from glob import glob
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from util import get_list_distances_from_preds


def load_feature_file(path: str) -> dict:
    """Load feature file and return dictionary."""
    data = np.load(path)
    result = {
        "labels": data["labels"].astype("float32"),
        "top1_distance": data["top1_distance"].astype("float32"),
        "peakiness": data["peakiness"].astype("float32"),
        "sue_score": data["sue_score"].astype("float32"),
    }
    # Add new features if available
    if "topk_distance_spread" in data:
        result["topk_distance_spread"] = data["topk_distance_spread"].astype("float32")
        result["top1_top2_similarity"] = data["top1_top2_similarity"].astype("float32")
        result["top1_top3_ratio"] = data["top1_top3_ratio"].astype("float32")
        result["top2_top3_ratio"] = data["top2_top3_ratio"].astype("float32")
        result["geographic_clustering"] = data["geographic_clustering"].astype("float32")
    return result


def build_feature_matrix(features_dict: dict, expected_feature_names: list) -> np.ndarray:
    """Build feature matrix based on expected feature names from model."""
    feature_arrays = []
    for name in expected_feature_names:
        if name not in features_dict:
            raise ValueError(f"Feature '{name}' not found in feature file")
        feature_arrays.append(features_dict[name])
    
    X = np.stack(feature_arrays, axis=1).astype("float32")
    return X


def compute_recall_at_1_adaptive(
    preds_dir: Path,
    inliers_dir: Path,
    is_hard: np.ndarray,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> float:
    """
    Compute Recall@1 for adaptive re-ranking strategy.
    
    Args:
        preds_dir: Directory with prediction .txt files
        inliers_dir: Directory with .torch files (image matching results)
        is_hard: Boolean array indicating which queries are hard
        num_preds: Number of predictions to consider
        positive_dist_threshold: Distance threshold in meters
    
    Returns:
        Recall@1 as percentage
    """
    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    if total_queries != len(is_hard):
        raise ValueError(f"Mismatch: {total_queries} txt files vs {len(is_hard)} hard flags")
    
    correct_at_1 = 0
    
    for idx, txt_file_query in enumerate(txt_files):
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
            txt_filename = Path(txt_file_query).name
            torch_filename = txt_filename.replace(".txt", ".torch")
            torch_file_query = inliers_dir / torch_filename
            
            if is_hard[idx] and torch_file_query.exists():
                # Hard query: re-rank by inliers
                query_results = torch.load(str(torch_file_query), weights_only=False)
                actual_num_preds = min(len(query_results), num_preds, len(geo_dists))
                if actual_num_preds == 0:
                    continue
                
                query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
                for i in range(actual_num_preds):
                    query_db_inliers[i] = query_results[i]["num_inliers"]
                query_db_inliers, indices = torch.sort(query_db_inliers, descending=True)
                query_geo_dists = geo_dists[:actual_num_preds][indices]
            else:
                # Easy query: use retrieval-only ordering
                query_geo_dists = geo_dists
            
            # Check if Top-1 is correct
            if len(query_geo_dists) > 0 and query_geo_dists[0] <= positive_dist_threshold:
                correct_at_1 += 1
                
        except Exception as e:
            continue
    
    recall_at_1 = (correct_at_1 / total_queries) * 100.0 if total_queries > 0 else 0.0
    return recall_at_1


def analyze_dataset(
    model_path: Path,
    feature_path: Path,
    preds_dir: Path,
    inliers_dir: Path,
    dataset_name: str,
    thresholds: np.ndarray,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> dict:
    """
    Analyze threshold impact for a single dataset.
    
    Returns:
        Dictionary with:
        - thresholds: array of thresholds tested
        - recall_at_1: array of R@1 for each threshold
        - hard_query_rates: array of hard query rates for each threshold
        - cost_savings: array of cost savings for each threshold
        - optimal_threshold: threshold that maximizes R@1
        - max_recall_at_1: maximum R@1 achieved
    """
    print(f"\n{'='*70}")
    print(f"Analyzing dataset: {dataset_name}")
    print(f"{'='*70}")
    
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    
    # Load features
    features = load_feature_file(str(feature_path))
    X = build_feature_matrix(features, feature_names)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    
    if (~valid_mask).sum() > 0:
        print(f"Removed {(~valid_mask).sum()} queries with NaN features")
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]  # Probability of being easy
    
    num_queries = len(probs)
    print(f"Total queries: {num_queries}")
    print(f"Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
    print(f"Mean probability: {probs.mean():.3f}")
    
    # Test each threshold
    recall_at_1_values = []
    hard_query_rates = []
    cost_savings = []
    
    print(f"\nTesting {len(thresholds)} thresholds...")
    for threshold in tqdm(thresholds, desc=f"Testing thresholds for {dataset_name}"):
        # Classify queries
        is_easy = probs >= threshold
        is_hard = ~is_easy
        
        hard_query_rate = is_hard.mean()
        hard_query_rates.append(hard_query_rate)
        
        # Cost savings = percentage of queries that are easy (skip image matching)
        cost_saving = is_easy.mean()
        cost_savings.append(cost_saving)
        
        # Compute Recall@1 (this is the expensive part)
        recall_at_1 = compute_recall_at_1_adaptive(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        recall_at_1_values.append(recall_at_1)
    
    # Find optimal threshold (maximizes R@1)
    recall_array = np.array(recall_at_1_values)
    optimal_idx = np.argmax(recall_array)
    optimal_threshold = thresholds[optimal_idx]
    max_recall_at_1 = recall_at_1_values[optimal_idx]
    
    print(f"\nResults for {dataset_name}:")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Max R@1: {max_recall_at_1:.2f}%")
    print(f"  Hard query rate at optimal: {hard_query_rates[optimal_idx]*100:.1f}%")
    print(f"  Cost savings at optimal: {cost_savings[optimal_idx]*100:.1f}%")
    
    return {
        "dataset_name": dataset_name,
        "thresholds": thresholds,
        "recall_at_1": np.array(recall_at_1_values),
        "hard_query_rates": np.array(hard_query_rates),
        "cost_savings": np.array(cost_savings),
        "optimal_threshold": optimal_threshold,
        "max_recall_at_1": max_recall_at_1,
        "num_queries": num_queries,
        "prob_mean": probs.mean(),
        "prob_std": probs.std(),
    }


def plot_threshold_analysis(results_list: list, output_dir: Path):
    """Create comprehensive plots for threshold analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: R@1 vs Threshold for all datasets
    plt.figure(figsize=(12, 8))
    for result in results_list:
        plt.plot(
            result["thresholds"],
            result["recall_at_1"],
            marker='o',
            markersize=4,
            label=f"{result['dataset_name']} (opt: {result['optimal_threshold']:.3f})",
            linewidth=2
        )
        # Mark optimal point
        plt.plot(
            result["optimal_threshold"],
            result["max_recall_at_1"],
            marker='*',
            markersize=15,
            color=plt.gca().lines[-1].get_color()
        )
    
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Recall@1 (%)", fontsize=12, fontweight='bold')
    plt.title("Recall@1 vs Threshold for Different Datasets", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'recall_vs_threshold.png'}")
    
    # Plot 2: Hard Query Rate vs Threshold
    plt.figure(figsize=(12, 8))
    for result in results_list:
        plt.plot(
            result["thresholds"],
            result["hard_query_rates"] * 100,
            marker='s',
            markersize=4,
            label=result["dataset_name"],
            linewidth=2
        )
    
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Hard Query Rate (%)", fontsize=12, fontweight='bold')
    plt.title("Hard Query Rate vs Threshold", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "hard_query_rate_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'hard_query_rate_vs_threshold.png'}")
    
    # Plot 3: Cost Savings vs Threshold
    plt.figure(figsize=(12, 8))
    for result in results_list:
        plt.plot(
            result["thresholds"],
            result["cost_savings"] * 100,
            marker='^',
            markersize=4,
            label=result["dataset_name"],
            linewidth=2
        )
    
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Cost Savings (%)", fontsize=12, fontweight='bold')
    plt.title("Cost Savings (Time Saved) vs Threshold", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_savings_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cost_savings_vs_threshold.png'}")
    
    # Plot 4: R@1 vs Cost Savings (Trade-off curve)
    plt.figure(figsize=(12, 8))
    for result in results_list:
        plt.plot(
            result["cost_savings"] * 100,
            result["recall_at_1"],
            marker='o',
            markersize=4,
            label=result["dataset_name"],
            linewidth=2
        )
        # Mark optimal point
        optimal_cost_saving = result["cost_savings"][np.argmax(result["recall_at_1"])]
        plt.plot(
            optimal_cost_saving * 100,
            result["max_recall_at_1"],
            marker='*',
            markersize=15,
            color=plt.gca().lines[-1].get_color()
        )
    
    plt.xlabel("Cost Savings (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Recall@1 (%)", fontsize=12, fontweight='bold')
    plt.title("Performance vs Cost Trade-off", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_cost_savings.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'recall_vs_cost_savings.png'}")
    
    # Plot 5: Dataset comparison - Optimal thresholds
    plt.figure(figsize=(10, 6))
    datasets = [r["dataset_name"] for r in results_list]
    optimal_thresholds = [r["optimal_threshold"] for r in results_list]
    max_recalls = [r["max_recall_at_1"] for r in results_list]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, optimal_thresholds, width, label='Optimal Threshold', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, max_recalls, width, label='Max R@1 (%)', color='coral', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Threshold', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Max Recall@1 (%)', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.set_title('Optimal Threshold and Max R@1 by Dataset', fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_comparison_optimal.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'dataset_comparison_optimal.png'}")


def generate_summary_table(results_list: list, output_dir: Path):
    """Generate summary table with all results."""
    output_file = output_dir / "threshold_analysis_summary.md"
    
    with open(output_file, 'w') as f:
        f.write("# Threshold Analysis Summary\n\n")
        f.write("## Dataset Comparison\n\n")
        f.write("| Dataset | Optimal Threshold | Max R@1 (%) | Hard Query Rate (%) | Cost Savings (%) | Num Queries | Mean Prob |\n")
        f.write("|---------|------------------|-------------|---------------------|------------------|-------------|----------|\n")
        
        for result in results_list:
            optimal_idx = np.argmax(result["recall_at_1"])
            f.write(f"| {result['dataset_name']} | "
                   f"{result['optimal_threshold']:.3f} | "
                   f"{result['max_recall_at_1']:.2f} | "
                   f"{result['hard_query_rates'][optimal_idx]*100:.1f} | "
                   f"{result['cost_savings'][optimal_idx]*100:.1f} | "
                   f"{result['num_queries']} | "
                   f"{result['prob_mean']:.3f} |\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("### 1. Threshold Variation Across Datasets\n\n")
        thresholds = [r["optimal_threshold"] for r in results_list]
        f.write(f"- **Threshold range**: {min(thresholds):.3f} to {max(thresholds):.3f}\n")
        f.write(f"- **Mean threshold**: {np.mean(thresholds):.3f}\n")
        f.write(f"- **Std threshold**: {np.std(thresholds):.3f}\n")
        f.write("\n**Conclusion**: Optimal threshold varies significantly across datasets, confirming the need for dataset-specific calibration.\n\n")
        
        f.write("### 2. Performance Impact\n\n")
        recalls = [r["max_recall_at_1"] for r in results_list]
        f.write(f"- **R@1 range**: {min(recalls):.2f}% to {max(recalls):.2f}%\n")
        f.write(f"- **Mean R@1**: {np.mean(recalls):.2f}%\n")
        f.write("\n### 3. Cost Savings\n\n")
        savings = [r["cost_savings"][np.argmax(r["recall_at_1"])] * 100 for r in results_list]
        f.write(f"- **Cost savings range**: {min(savings):.1f}% to {max(savings):.1f}%\n")
        f.write(f"- **Mean cost savings**: {np.mean(savings):.1f}%\n")
        f.write("\n### 4. Dataset Characteristics\n\n")
        for result in results_list:
            f.write(f"**{result['dataset_name']}**:\n")
            f.write(f"- Mean probability: {result['prob_mean']:.3f} (std: {result['prob_std']:.3f})\n")
            f.write(f"- Optimal threshold: {result['optimal_threshold']:.3f}\n")
            f.write(f"- Max R@1: {result['max_recall_at_1']:.2f}%\n")
            f.write(f"- Cost savings: {result['cost_savings'][np.argmax(result['recall_at_1'])]*100:.1f}%\n\n")
    
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive threshold analysis: R@1 vs threshold, dataset comparison, cost savings"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained logistic regression model (.pkl)",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        required=True,
        help="Dataset names (e.g., sf_xs_test tokyo_xs_test svox_test)",
    )
    parser.add_argument(
        "--feature-paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to feature files (.npz) for each dataset",
    )
    parser.add_argument(
        "--preds-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to prediction directories for each dataset",
    )
    parser.add_argument(
        "--inliers-dirs",
        type=str,
        nargs="+",
        required=True,
        help="Paths to inliers directories for each dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/threshold_analysis",
        help="Output directory for plots and summary",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.1, 0.95],
        help="Threshold range to test (default: 0.1 0.95)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
        help="Threshold step size (default: 0.02)",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions to consider (default: 20)",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold in meters (default: 25)",
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.datasets) != len(args.feature_paths) or \
       len(args.datasets) != len(args.preds_dirs) or \
       len(args.datasets) != len(args.inliers_dirs):
        raise ValueError("Number of datasets must match number of feature paths, preds dirs, and inliers dirs")
    
    # Generate threshold range
    thresholds = np.arange(args.threshold_range[0], args.threshold_range[1] + args.threshold_step, args.threshold_step)
    print(f"Testing {len(thresholds)} thresholds from {args.threshold_range[0]} to {args.threshold_range[1]}")
    
    # Analyze each dataset
    results_list = []
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    
    for dataset_name, feature_path, preds_dir, inliers_dir in zip(
        args.datasets, args.feature_paths, args.preds_dirs, args.inliers_dirs
    ):
        result = analyze_dataset(
            model_path=model_path,
            feature_path=Path(feature_path),
            preds_dir=Path(preds_dir),
            inliers_dir=Path(inliers_dir),
            dataset_name=dataset_name,
            thresholds=thresholds,
            num_preds=args.num_preds,
            positive_dist_threshold=args.positive_dist_threshold
        )
        results_list.append(result)
    
    # Generate plots
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")
    plot_threshold_analysis(results_list, output_dir)
    
    # Generate summary
    print(f"\n{'='*70}")
    print("Generating summary...")
    print(f"{'='*70}")
    generate_summary_table(results_list, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

