"""
Comprehensive threshold analysis for adaptive re-ranking.

This script:
1. Tests different thresholds (without using full re-ranking results)
2. For each threshold, runs adaptive re-ranking
3. Evaluates performance for each threshold
4. Compares with full re-ranking (ground truth) at the end
5. Generates plots and cost savings analysis
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


def compute_recall_metrics_adaptive(
    preds_dir: Path,
    inliers_dir: Path,
    is_hard: np.ndarray,
    recall_values: list,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> dict:
    """
    Compute Recall@N metrics for adaptive re-ranking strategy.
    
    Returns:
        Dictionary with recall metrics
    """
    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    if total_queries != len(is_hard):
        raise ValueError(f"Mismatch: {total_queries} txt files vs {len(is_hard)} hard flags")
    
    recalls = np.zeros(len(recall_values), dtype="float32")
    processed_queries = 0
    
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
            
            # Compute recall contributions
            for i, n in enumerate(recall_values):
                if n <= len(query_geo_dists) and torch.any(query_geo_dists[:n] <= positive_dist_threshold):
                    recalls[i:] += 1
                    break
            
            processed_queries += 1
                
        except Exception as e:
            continue
    
    if processed_queries > 0:
        recalls = recalls / processed_queries * 100.0
    
    return {
        "recall_at_1": recalls[0] if len(recall_values) > 0 else 0.0,
        "recall_at_5": recalls[1] if len(recall_values) > 1 else 0.0,
        "recall_at_10": recalls[2] if len(recall_values) > 2 else 0.0,
        "recall_at_20": recalls[3] if len(recall_values) > 3 else 0.0,
        "processed_queries": processed_queries
    }


def analyze_thresholds_for_dataset(
    model_path: Path,
    feature_path: Path,
    preds_dir: Path,
    inliers_dir: Path,
    dataset_name: str,
    thresholds: np.ndarray,
    recall_values: list = [1, 5, 10, 20],
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> dict:
    """
    Analyze different thresholds for a dataset.
    
    Returns:
        Dictionary with results for each threshold
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
    print(f"Testing {len(thresholds)} thresholds...\n")
    
    # Test each threshold
    results = []
    
    for threshold in tqdm(thresholds, desc=f"Testing thresholds for {dataset_name}"):
        # Classify queries
        is_easy = probs >= threshold
        is_hard = ~is_easy
        
        hard_query_rate = is_hard.mean()
        cost_savings = is_easy.mean()  # Percentage of queries that skip image matching
        
        # Compute adaptive re-ranking performance
        recall_metrics = compute_recall_metrics_adaptive(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=is_hard,
            recall_values=recall_values,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        
        results.append({
            "threshold": threshold,
            "hard_query_rate": hard_query_rate,
            "cost_savings": cost_savings,
            "recall_at_1": recall_metrics["recall_at_1"],
            "recall_at_5": recall_metrics["recall_at_5"],
            "recall_at_10": recall_metrics["recall_at_10"],
            "recall_at_20": recall_metrics["recall_at_20"],
        })
    
    return {
        "dataset_name": dataset_name,
        "num_queries": num_queries,
        "prob_mean": probs.mean(),
        "prob_std": probs.std(),
        "prob_min": probs.min(),
        "prob_max": probs.max(),
        "results": results
    }


def load_full_reranking_results(preds_dir: Path, inliers_dir: Path, recall_values: list, num_preds: int = 20, positive_dist_threshold: int = 25) -> dict:
    """
    Load full re-ranking results (ground truth).
    Assumes all queries are hard (is_hard = all True).
    """
    txt_files = glob(os.path.join(str(preds_dir), "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    is_hard = np.ones(total_queries, dtype=bool)  # All queries are hard (full re-ranking)
    
    recall_metrics = compute_recall_metrics_adaptive(
        preds_dir=preds_dir,
        inliers_dir=inliers_dir,
        is_hard=is_hard,
        recall_values=recall_values,
        num_preds=num_preds,
        positive_dist_threshold=positive_dist_threshold
    )
    
    return recall_metrics


def plot_threshold_analysis(all_results: list, full_reranking_results: dict, output_dir: Path):
    """Create comprehensive plots for threshold analysis."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: R@1 vs Threshold for all datasets (with ground-truth star markers and selected thresholds)
    plt.figure(figsize=(16, 10))
    colors = plt.cm.tab10(np.linspace(0, 1, len(all_results)))
    
    for idx, dataset_result in enumerate(all_results):
        thresholds = [r["threshold"] for r in dataset_result["results"]]
        recall_at_1 = [r["recall_at_1"] for r in dataset_result["results"]]
        full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
        
        # Find selected threshold (>=90% performance with highest cost savings)
        target_r1 = full_r1 * 0.90
        valid_results = [r for r in dataset_result["results"] if r["recall_at_1"] >= target_r1]
        
        if valid_results:
            selected_result = max(valid_results, key=lambda x: x["cost_savings"])
        else:
            selected_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
        
        selected_threshold = selected_result["threshold"]
        selected_r1 = selected_result["recall_at_1"]
        
        # Plot adaptive re-ranking line
        plt.plot(
            thresholds,
            recall_at_1,
            marker='o',
            markersize=6,
            label=f"{dataset_result['dataset_name']} (Adaptive)",
            linewidth=2.5,
            color=colors[idx],
            alpha=0.8
        )
        
        # Highlight selected threshold with diamond marker
        plt.plot(
            selected_threshold,
            selected_r1,
            marker='D',
            markersize=14,
            color='green',
            markeredgecolor='darkgreen',
            markeredgewidth=2,
            zorder=15,
            label=f"{dataset_result['dataset_name']} (Selected: {selected_threshold:.3f})"
        )
        
        # Add ground-truth star marker (full re-ranking)
        # Place star at the rightmost threshold position
        max_threshold = max(thresholds)
        plt.plot(
            max_threshold,
            full_r1,
            marker='*',
            markersize=20,
            color=colors[idx],
            label=f"{dataset_result['dataset_name']} (Full Re-ranking: {full_r1:.2f}%)",
            linestyle='None',
            markeredgecolor='black',
            markeredgewidth=1.5
        )
        
        # Add horizontal dashed line for full re-ranking
        plt.axhline(
            y=full_r1,
            color=colors[idx],
            linestyle='--',
            alpha=0.4,
            linewidth=1.5
        )
        
        # Add vertical line at selected threshold
        plt.axvline(
            x=selected_threshold,
            color='green',
            linestyle=':',
            alpha=0.5,
            linewidth=1.5
        )
    
    plt.xlabel("Threshold", fontsize=14, fontweight='bold')
    plt.ylabel("Recall@1 Accuracy (%)", fontsize=14, fontweight='bold')
    plt.title("Recall@1 Accuracy vs Threshold Range\n(Star markers show Full Re-ranking Ground Truth)", 
              fontsize=16, fontweight='bold', pad=20)
    plt.legend(loc='best', fontsize=11, ncol=2, framealpha=0.9)
    plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
    plt.xlim([min([min([r["threshold"] for r in dr["results"]]) for dr in all_results]) - 0.05,
              max([max([r["threshold"] for r in dr["results"]]) for dr in all_results]) + 0.05])
    plt.tight_layout()
    plt.savefig(output_dir / "recall_at_1_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'recall_at_1_vs_threshold.png'}")
    
    # Plot 1b: Individual plot for each dataset (more detailed)
    for dataset_result in all_results:
        thresholds = [r["threshold"] for r in dataset_result["results"]]
        recall_at_1 = [r["recall_at_1"] for r in dataset_result["results"]]
        full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
        
        # Find selected threshold (>=90% performance with highest cost savings)
        target_r1 = full_r1 * 0.90
        valid_results = [r for r in dataset_result["results"] if r["recall_at_1"] >= target_r1]
        
        if valid_results:
            selected_result = max(valid_results, key=lambda x: x["cost_savings"])
        else:
            # If no result achieves 90%, use highest R@1
            selected_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
        
        selected_threshold = selected_result["threshold"]
        selected_r1 = selected_result["recall_at_1"]
        
        plt.figure(figsize=(12, 8))
        
        # Plot adaptive re-ranking line
        plt.plot(
            thresholds,
            recall_at_1,
            marker='o',
            markersize=8,
            label=f"Adaptive Re-ranking",
            linewidth=3,
            color='steelblue',
            alpha=0.8
        )
        
        # Highlight selected threshold with a special marker
        plt.plot(
            selected_threshold,
            selected_r1,
            marker='D',
            markersize=18,
            color='green',
            label=f"Selected Threshold: {selected_threshold:.3f} (R@1: {selected_r1:.2f}%)",
            linestyle='None',
            markeredgecolor='darkgreen',
            markeredgewidth=2.5,
            zorder=15
        )
        
        # Add vertical line at selected threshold
        plt.axvline(
            x=selected_threshold,
            color='green',
            linestyle=':',
            alpha=0.7,
            linewidth=2,
            label=f"Selected: {selected_threshold:.3f}"
        )
        
        # Add ground-truth star marker
        max_threshold = max(thresholds)
        plt.plot(
            max_threshold,
            full_r1,
            marker='*',
            markersize=25,
            color='red',
            label=f"Full Re-ranking (Ground Truth: {full_r1:.2f}%)",
            linestyle='None',
            markeredgecolor='darkred',
            markeredgewidth=2,
            zorder=10
        )
        
        # Add horizontal dashed line for full re-ranking
        plt.axhline(
            y=full_r1,
            color='red',
            linestyle='--',
            alpha=0.6,
            linewidth=2,
            label=f"Ground Truth: {full_r1:.2f}%"
        )
        
        # Annotate selected threshold
        plt.annotate(
            f'Selected\n{selected_threshold:.3f}\nR@1: {selected_r1:.2f}%',
            (selected_threshold, selected_r1),
            textcoords="offset points",
            xytext=(20, 30),
            ha='left',
            fontsize=11,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8, edgecolor='darkgreen', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='darkgreen', lw=2)
        )
        
        # Annotate key points with R@1 value
        for i, (thresh, r1) in enumerate(zip(thresholds, recall_at_1)):
            if i % 3 == 0 or abs(r1 - full_r1) < 2.0 or abs(thresh - selected_threshold) < 0.05:
                if abs(thresh - selected_threshold) > 0.05:  # Don't annotate selected point (already annotated)
                    plt.annotate(
                        f'{r1:.1f}%',
                        (thresh, r1),
                        textcoords="offset points",
                        xytext=(0, 10),
                        ha='center',
                        fontsize=9,
                        alpha=0.7
                    )
        
        plt.xlabel("Threshold", fontsize=14, fontweight='bold')
        plt.ylabel("Recall@1 Accuracy (%)", fontsize=14, fontweight='bold')
        plt.title(f"{dataset_result['dataset_name'].upper()}: Recall@1 Accuracy vs Threshold\n"
                  f"Selected Threshold: {selected_threshold:.3f} | Full Re-ranking Ground Truth: {full_r1:.2f}%", 
                  fontsize=15, fontweight='bold', pad=20)
        plt.legend(loc='best', fontsize=11, framealpha=0.9)
        plt.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
        plt.xlim([min(thresholds) - 0.05, max(thresholds) + 0.05])
        plt.ylim([min(min(recall_at_1), full_r1) - 2, max(max(recall_at_1), full_r1) + 2])
        plt.tight_layout()
        
        safe_name = dataset_result['dataset_name'].replace(' ', '_').replace('/', '_')
        plt.savefig(output_dir / f"recall_at_1_vs_threshold_{safe_name}.png", dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved: {output_dir / f'recall_at_1_vs_threshold_{safe_name}.png'}")
    
    # Plot 2: Performance Ratio vs Threshold (Adaptive / Full Re-ranking)
    plt.figure(figsize=(14, 8))
    for dataset_result in all_results:
        thresholds = [r["threshold"] for r in dataset_result["results"]]
        recall_at_1 = [r["recall_at_1"] for r in dataset_result["results"]]
        full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
        performance_ratio = [r1 / full_r1 if full_r1 > 0 else 0.0 for r1 in recall_at_1]
        
        plt.plot(
            thresholds,
            performance_ratio,
            marker='s',
            markersize=4,
            label=f"{dataset_result['dataset_name']}",
            linewidth=2
        )
    
    plt.axhline(y=0.90, color='red', linestyle='--', alpha=0.5, label='90% Target')
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Performance Ratio (Adaptive / Full Re-ranking)", fontsize=12, fontweight='bold')
    plt.title("Performance Ratio vs Threshold", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "performance_ratio_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'performance_ratio_vs_threshold.png'}")
    
    # Plot 3: Cost Savings vs Threshold
    plt.figure(figsize=(14, 8))
    for dataset_result in all_results:
        thresholds = [r["threshold"] for r in dataset_result["results"]]
        cost_savings = [r["cost_savings"] * 100 for r in dataset_result["results"]]
        
        plt.plot(
            thresholds,
            cost_savings,
            marker='^',
            markersize=4,
            label=dataset_result["dataset_name"],
            linewidth=2
        )
    
    plt.xlabel("Threshold", fontsize=12, fontweight='bold')
    plt.ylabel("Cost Savings (%)", fontsize=12, fontweight='bold')
    plt.title("Cost Savings vs Threshold", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "cost_savings_vs_threshold.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'cost_savings_vs_threshold.png'}")
    
    # Plot 4: R@1 vs Cost Savings (Trade-off curve)
    plt.figure(figsize=(14, 8))
    for dataset_result in all_results:
        cost_savings = [r["cost_savings"] * 100 for r in dataset_result["results"]]
        recall_at_1 = [r["recall_at_1"] for r in dataset_result["results"]]
        full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
        
        plt.plot(
            cost_savings,
            recall_at_1,
            marker='o',
            markersize=4,
            label=dataset_result["dataset_name"],
            linewidth=2
        )
        
        # Mark full re-ranking point (0% savings, full performance)
        plt.plot(
            0,
            full_r1,
            marker='*',
            markersize=15,
            color=plt.gca().lines[-1].get_color(),
            label=f"{dataset_result['dataset_name']} (Full Re-ranking)"
        )
    
    plt.xlabel("Cost Savings (%)", fontsize=12, fontweight='bold')
    plt.ylabel("Recall@1 (%)", fontsize=12, fontweight='bold')
    plt.title("Performance vs Cost Trade-off", fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10, ncol=2)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "recall_vs_cost_savings.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'recall_vs_cost_savings.png'}")
    
    # Plot 5: Dataset comparison - Optimal thresholds
    plt.figure(figsize=(12, 8))
    datasets = []
    optimal_thresholds = []
    optimal_r1 = []
    full_r1_values = []
    
    for dataset_result in all_results:
        # Find threshold with highest R@1
        best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
        datasets.append(dataset_result["dataset_name"])
        optimal_thresholds.append(best_result["threshold"])
        optimal_r1.append(best_result["recall_at_1"])
        full_r1_values.append(full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"])
    
    x = np.arange(len(datasets))
    width = 0.35
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()
    
    bars1 = ax1.bar(x - width/2, optimal_thresholds, width, label='Optimal Threshold', color='skyblue', alpha=0.8)
    bars2 = ax2.bar(x + width/2, optimal_r1, width, label='Optimal R@1', color='coral', alpha=0.8)
    bars3 = ax2.bar(x + width/2, full_r1_values, width, bottom=optimal_r1, label='Full Re-ranking R@1', color='lightgreen', alpha=0.8)
    
    ax1.set_xlabel('Dataset', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Optimal Threshold', fontsize=12, fontweight='bold', color='blue')
    ax2.set_ylabel('Recall@1 (%)', fontsize=12, fontweight='bold', color='red')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets, rotation=15, ha='right')
    ax1.set_title('Optimal Threshold and Performance by Dataset', fontsize=14, fontweight='bold')
    
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(output_dir / "dataset_comparison_optimal.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'dataset_comparison_optimal.png'}")


def generate_detailed_threshold_table(all_results: list, full_reranking_results: dict, output_dir: Path):
    """Generate detailed table showing Recall@1 for each threshold."""
    output_file = output_dir / "detailed_threshold_table.md"
    
    with open(output_file, 'w') as f:
        f.write("# Detailed Threshold Analysis - Recall@1 for Each Threshold\n\n")
        
        for dataset_result in all_results:
            dataset_name = dataset_result["dataset_name"]
            full_r1 = full_reranking_results[dataset_name]["recall_at_1"]
            
            f.write(f"## {dataset_name.upper()}\n\n")
            f.write(f"**Full Re-ranking R@1 (Ground Truth)**: {full_r1:.2f}%\n\n")
            f.write("| Threshold | Hard Query Rate | Recall@1 | Performance Ratio | Cost Savings |\n")
            f.write("|-----------|-----------------|----------|-------------------|--------------|\n")
            
            # Sort by threshold
            sorted_results = sorted(dataset_result["results"], key=lambda x: x["threshold"])
            
            for result in sorted_results:
                performance_ratio = result["recall_at_1"] / full_r1 if full_r1 > 0 else 0.0
                f.write(f"| {result['threshold']:.3f} | "
                       f"{result['hard_query_rate']*100:.1f}% | "
                       f"{result['recall_at_1']:.2f}% | "
                       f"{performance_ratio:.1%} | "
                       f"{result['cost_savings']*100:.1f}% |\n")
            
            # Find best threshold (>=90% performance with highest cost savings)
            target_r1 = full_r1 * 0.90
            valid_results = [r for r in dataset_result["results"] if r["recall_at_1"] >= target_r1]
            
            if valid_results:
                best_result = max(valid_results, key=lambda x: x["cost_savings"])
                f.write(f"\n**Recommended Threshold**: {best_result['threshold']:.3f}\n")
                f.write(f"- Recall@1: {best_result['recall_at_1']:.2f}% ({best_result['recall_at_1']/full_r1:.1%} of full re-ranking)\n")
                f.write(f"- Hard Query Rate: {best_result['hard_query_rate']*100:.1f}%\n")
                f.write(f"- Cost Savings: {best_result['cost_savings']*100:.1f}%\n")
            else:
                best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
                f.write(f"\n**Best Threshold (Highest R@1)**: {best_result['threshold']:.3f}\n")
                f.write(f"- Recall@1: {best_result['recall_at_1']:.2f}% ({best_result['recall_at_1']/full_r1:.1%} of full re-ranking)\n")
                f.write(f"- Hard Query Rate: {best_result['hard_query_rate']*100:.1f}%\n")
                f.write(f"- Cost Savings: {best_result['cost_savings']*100:.1f}%\n")
            
            f.write("\n---\n\n")
    
    print(f"Saved: {output_file}")


def generate_summary_report(all_results: list, full_reranking_results: dict, output_dir: Path):
    """Generate comprehensive summary report."""
    output_file = output_dir / "threshold_analysis_summary.md"
    
    with open(output_file, 'w') as f:
        f.write("# Threshold Analysis Summary\n\n")
        f.write("## Dataset Comparison\n\n")
        f.write("| Dataset | Optimal Threshold | Optimal R@1 | Full Re-ranking R@1 | Performance Ratio | Hard Query Rate | Cost Savings |\n")
        f.write("|---------|------------------|-------------|---------------------|-------------------|-----------------|--------------|\n")
        
        for dataset_result in all_results:
            # Find threshold with best balance: >=90% performance and highest cost savings
            full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
            target_r1 = full_r1 * 0.90  # 90% of full re-ranking
            
            # Filter results that achieve >=90% performance
            valid_results = [r for r in dataset_result["results"] if r["recall_at_1"] >= target_r1]
            
            if valid_results:
                # Among valid results, pick one with highest cost savings
                best_result = max(valid_results, key=lambda x: x["cost_savings"])
            else:
                # If no result achieves 90%, use highest R@1
                best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
            
            performance_ratio = best_result["recall_at_1"] / full_r1 if full_r1 > 0 else 0.0
            
            f.write(f"| {dataset_result['dataset_name']} | "
                   f"{best_result['threshold']:.3f} | "
                   f"{best_result['recall_at_1']:.2f}% | "
                   f"{full_r1:.2f}% | "
                   f"{performance_ratio:.1%} | "
                   f"{best_result['hard_query_rate']*100:.1f}% | "
                   f"{best_result['cost_savings']*100:.1f}% |\n")
        
        f.write("\n## Cost Savings Analysis\n\n")
        f.write("### Time Savings Calculation\n\n")
        f.write("**Cost Savings = (Easy Queries / Total Queries) × 100%**\n\n")
        f.write("Where:\n")
        f.write("- **Easy queries**: Queries predicted as easy (skip image matching)\n")
        f.write("- **Time per query**: ~9.5 seconds (SuperPoint + LightGlue)\n")
        f.write("- **Total time saved**: Easy queries × 9.5 seconds\n\n")
        
        for dataset_result in all_results:
            best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
            easy_queries = int(dataset_result["num_queries"] * (1 - best_result["hard_query_rate"]))
            time_saved = easy_queries * 9.5 / 60  # minutes
            
            f.write(f"**{dataset_result['dataset_name']}**:\n")
            f.write(f"- Total queries: {dataset_result['num_queries']}\n")
            f.write(f"- Easy queries: {easy_queries} ({best_result['cost_savings']*100:.1f}%)\n")
            f.write(f"- Time saved: {time_saved:.1f} minutes\n\n")
        
        f.write("\n## Key Findings\n\n")
        f.write("### 1. Threshold Variation Across Datasets\n\n")
        optimal_thresholds = [max(r["results"], key=lambda x: x["recall_at_1"])["threshold"] for r in all_results]
        f.write(f"- **Threshold range**: {min(optimal_thresholds):.3f} to {max(optimal_thresholds):.3f}\n")
        f.write(f"- **Mean threshold**: {np.mean(optimal_thresholds):.3f}\n")
        f.write(f"- **Std threshold**: {np.std(optimal_thresholds):.3f}\n")
        f.write("\n**Conclusion**: Optimal threshold varies significantly across datasets.\n\n")
        
        f.write("### 2. Performance vs Full Re-ranking\n\n")
        for dataset_result in all_results:
            best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
            full_r1 = full_reranking_results[dataset_result["dataset_name"]]["recall_at_1"]
            performance_ratio = best_result["recall_at_1"] / full_r1 if full_r1 > 0 else 0.0
            f.write(f"**{dataset_result['dataset_name']}**: {performance_ratio:.1%} of full re-ranking performance\n")
        
        f.write("\n### 3. Cost Savings\n\n")
        for dataset_result in all_results:
            best_result = max(dataset_result["results"], key=lambda x: x["recall_at_1"])
            f.write(f"**{dataset_result['dataset_name']}**: {best_result['cost_savings']*100:.1f}% time savings\n")
    
    print(f"Saved: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive threshold analysis for adaptive re-ranking"
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
        help="Dataset names (e.g., sf_xs_test tokyo_xs_test)",
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
        help="Paths to inliers directories (from full re-ranking) for each dataset",
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
    model_path = Path(args.model_path)
    output_dir = Path(args.output_dir)
    all_results = []
    full_reranking_results = {}
    
    recall_values = [1, 5, 10, 20]
    
    for dataset_name, feature_path, preds_dir, inliers_dir in zip(
        args.datasets, args.feature_paths, args.preds_dirs, args.inliers_dirs
    ):
        # Analyze thresholds
        result = analyze_thresholds_for_dataset(
            model_path=model_path,
            feature_path=Path(feature_path),
            preds_dir=Path(preds_dir),
            inliers_dir=Path(inliers_dir),
            dataset_name=dataset_name,
            thresholds=thresholds,
            recall_values=recall_values,
            num_preds=args.num_preds,
            positive_dist_threshold=args.positive_dist_threshold
        )
        all_results.append(result)
        
        # Load full re-ranking results (ground truth)
        print(f"\nLoading full re-ranking results for {dataset_name}...")
        full_reranking_results[dataset_name] = load_full_reranking_results(
            preds_dir=Path(preds_dir),
            inliers_dir=Path(inliers_dir),
            recall_values=recall_values,
            num_preds=args.num_preds,
            positive_dist_threshold=args.positive_dist_threshold
        )
        print(f"Full re-ranking R@1: {full_reranking_results[dataset_name]['recall_at_1']:.2f}%")
    
    # Generate plots
    print(f"\n{'='*70}")
    print("Generating plots...")
    print(f"{'='*70}")
    plot_threshold_analysis(all_results, full_reranking_results, output_dir)
    
    # Generate detailed threshold table
    print(f"\n{'='*70}")
    print("Generating detailed threshold table...")
    print(f"{'='*70}")
    generate_detailed_threshold_table(all_results, full_reranking_results, output_dir)
    
    # Generate summary
    print(f"\n{'='*70}")
    print("Generating summary...")
    print(f"{'='*70}")
    generate_summary_report(all_results, full_reranking_results, output_dir)
    
    print(f"\n{'='*70}")
    print("Analysis complete!")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")


if __name__ == "__main__":
    main()

