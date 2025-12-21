"""
Plot validation threshold analysis: Recall@1 accuracy vs threshold range.

This script visualizes how Recall@1 accuracy changes with different thresholds
during validation, and marks the chosen optimal threshold.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm


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


def build_feature_matrix(features_dict: dict) -> tuple[np.ndarray, np.ndarray, list]:
    """Build (X, y) from a feature dictionary."""
    top1_distance = features_dict["top1_distance"]
    peakiness = features_dict["peakiness"]
    sue_score = features_dict["sue_score"]
    
    # Use 8 improved features
    if "topk_distance_spread" in features_dict:
        topk_distance_spread = features_dict["topk_distance_spread"]
        top1_top2_similarity = features_dict["top1_top2_similarity"]
        top1_top3_ratio = features_dict["top1_top3_ratio"]
        top2_top3_ratio = features_dict["top2_top3_ratio"]
        geographic_clustering = features_dict["geographic_clustering"]
        
        X = np.stack(
            [top1_distance, peakiness, sue_score,
             topk_distance_spread, top1_top2_similarity,
             top1_top3_ratio, top2_top3_ratio, geographic_clustering],
            axis=1,
        ).astype("float32")
        feature_names = ["top1_distance", "peakiness", "sue_score",
                        "topk_distance_spread", "top1_top2_similarity",
                        "top1_top3_ratio", "top2_top3_ratio", "geographic_clustering"]
    else:
        # Fallback to 3 basic features
        X = np.stack(
            [top1_distance, peakiness, sue_score],
            axis=1,
        ).astype("float32")
        feature_names = ["top1_distance", "peakiness", "sue_score"]
    
    # Target: Easy score (1 = easy/correct, 0 = hard/wrong)
    labels = features_dict["labels"]  # 1 = correct, 0 = wrong
    easy_score = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
    
    return X, easy_score, feature_names


def compute_recall_at_1_for_threshold(
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
) -> float:
    """
    Compute Recall@1 accuracy for a given threshold.
    
    Recall@1 accuracy = percentage of queries correctly classified as easy/hard
    This is the classification accuracy, not the VPR Recall@1 metric.
    """
    y_pred = (y_probs >= threshold).astype(int)
    accuracy = (y_pred == y_true).mean()
    return accuracy * 100.0  # Convert to percentage


def compute_full_reranking_r1(
    preds_dir: Path,
    inliers_dir: Path,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> float:
    """
    Compute full re-ranking R@1 (ground-truth).
    Assumes all queries are re-ranked.
    """
    from glob import glob
    import torch
    from util import get_list_distances_from_preds
    
    txt_files = glob(str(preds_dir / "*.txt"))
    txt_files.sort(key=lambda x: int(Path(x).stem))
    
    total_queries = len(txt_files)
    correct_at_1 = 0
    
    for txt_file_query in txt_files:
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(txt_file_query))[:num_preds]
            txt_filename = Path(txt_file_query).name
            torch_filename = txt_filename.replace(".txt", ".torch")
            torch_file_query = inliers_dir / torch_filename
            
            if torch_file_query.exists():
                # Re-rank by inliers
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
                # No inliers, use retrieval-only
                query_geo_dists = geo_dists
            
            # Check if Top-1 is correct
            if len(query_geo_dists) > 0 and query_geo_dists[0] <= positive_dist_threshold:
                correct_at_1 += 1
        except Exception:
            continue
    
    recall_at_1 = (correct_at_1 / total_queries) * 100.0 if total_queries > 0 else 0.0
    return recall_at_1


def compute_adaptive_r1_for_threshold(
    preds_dir: Path,
    inliers_dir: Path,
    is_hard: np.ndarray,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
) -> float:
    """
    Compute adaptive R@1 for a given hard query classification.
    """
    from glob import glob
    import torch
    from util import get_list_distances_from_preds
    
    txt_files = glob(str(preds_dir / "*.txt"))
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
        except Exception:
            continue
    
    recall_at_1 = (correct_at_1 / total_queries) * 100.0 if total_queries > 0 else 0.0
    return recall_at_1


def analyze_validation_thresholds(
    model_path: Path,
    val_features_path: Path,
    threshold_range: tuple = (0.1, 0.95),
    threshold_step: float = 0.01,
    output_path: Path = None,
    preds_dir: Path = None,
    inliers_dir: Path = None,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
):
    """
    Analyze how Recall@1 accuracy changes with different thresholds during validation.
    
    Args:
        model_path: Path to trained model
        val_features_path: Path to validation features
        threshold_range: (min, max) threshold range to test
        threshold_step: Step size for threshold testing
        output_path: Path to save the plot
    """
    # Load model
    print(f"Loading model from: {model_path}")
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    optimal_threshold = bundle.get("optimal_threshold", 0.5)
    threshold_method = bundle.get("threshold_method", "f1")
    
    print(f"  Optimal threshold (from model): {optimal_threshold:.3f}")
    print(f"  Threshold method: {threshold_method}")
    
    # Load validation features
    print(f"\nLoading validation features from: {val_features_path}")
    val_features = load_feature_file(str(val_features_path))
    X_val, y_val, _ = build_feature_matrix(val_features)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[valid_mask]
    y_val = y_val[valid_mask]
    
    print(f"  Valid queries: {len(y_val)}")
    print(f"  Easy queries (actual): {y_val.sum():.0f} ({100*y_val.mean():.1f}%)")
    print(f"  Hard queries (actual): {(1-y_val).sum():.0f} ({100*(1-y_val).mean():.1f}%)")
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    y_val_probs = model.predict_proba(X_val_scaled)[:, 1]  # Probability of being easy
    
    print(f"\n  Probability range: [{y_val_probs.min():.3f}, {y_val_probs.max():.3f}]")
    print(f"  Mean probability: {y_val_probs.mean():.3f}")
    
    # Compute full re-ranking R@1 (ground-truth) if inliers available
    full_reranking_r1 = None
    if preds_dir is not None and inliers_dir is not None:
        print(f"\nComputing full re-ranking R@1 (ground-truth)...")
        print(f"  Preds dir: {preds_dir}")
        print(f"  Inliers dir: {inliers_dir}")
        if preds_dir.exists() and inliers_dir.exists():
            full_reranking_r1 = compute_full_reranking_r1(
                preds_dir=preds_dir,
                inliers_dir=inliers_dir,
                num_preds=num_preds,
                positive_dist_threshold=positive_dist_threshold
            )
            print(f"  Full re-ranking R@1: {full_reranking_r1:.2f}%")
        else:
            print(f"  Warning: Preds or inliers directory not found. Skipping full re-ranking computation.")
    else:
        print(f"\nNo preds/inliers directories provided. Skipping full re-ranking computation.")
        print(f"  (Classification accuracy will still be computed)")
    
    # Test different thresholds (include 1.0 to show full re-ranking)
    threshold_max = max(threshold_range[1], 1.0)  # Ensure we test up to 1.0
    thresholds = np.arange(threshold_range[0], threshold_max + threshold_step, threshold_step)
    # Ensure 1.0 is included
    if 1.0 not in thresholds:
        thresholds = np.append(thresholds, 1.0)
    thresholds = np.sort(np.unique(thresholds))
    print(f"\nTesting {len(thresholds)} thresholds from {threshold_range[0]:.2f} to {threshold_max:.2f}...")
    print(f"  Note: Threshold = 1.0 means ALL queries are hard (full re-ranking)")
    
    results = []
    for threshold in tqdm(thresholds, desc="Testing thresholds"):
        # Compute Recall@1 accuracy (classification accuracy)
        classification_accuracy = compute_recall_at_1_for_threshold(y_val, y_val_probs, threshold)
        
        # Also compute F1, precision, recall for reference
        y_pred = (y_val_probs >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred, zero_division=0)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        
        # Count predictions
        num_easy_pred = y_pred.sum()
        num_hard_pred = (1 - y_pred).sum()
        
        # Compute adaptive VPR R@1 if inliers available
        adaptive_r1 = None
        if preds_dir is not None and inliers_dir is not None and preds_dir.exists() and inliers_dir.exists():
            # Hard = probability < threshold (not easy)
            # At threshold = 1.0, ALL queries are hard (full re-ranking)
            is_hard = (y_val_probs < threshold).astype(bool)
            adaptive_r1 = compute_adaptive_r1_for_threshold(
                preds_dir=preds_dir,
                inliers_dir=inliers_dir,
                is_hard=is_hard,
                num_preds=num_preds,
                positive_dist_threshold=positive_dist_threshold
            )
            
            # Verify: at threshold = 1.0, should match full re-ranking
            if abs(threshold - 1.0) < 0.01 and full_reranking_r1 is not None:
                if abs(adaptive_r1 - full_reranking_r1) > 0.1:  # Allow small numerical differences
                    print(f"  Warning: At threshold=1.0, adaptive R@1 ({adaptive_r1:.2f}%) != full re-ranking ({full_reranking_r1:.2f}%)")
        
        results.append({
            "threshold": threshold,
            "classification_accuracy": classification_accuracy,
            "adaptive_r1": adaptive_r1,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "num_easy_pred": num_easy_pred,
            "num_hard_pred": num_hard_pred,
        })
    
    # Find best threshold based on F1 (same as training)
    best_result = max(results, key=lambda x: x["f1"])
    best_threshold_f1 = best_result["threshold"]
    
    print(f"\n{'='*70}")
    print(f"Validation Threshold Analysis Results")
    print(f"{'='*70}")
    print(f"Best threshold (F1-max): {best_threshold_f1:.3f}")
    print(f"  Classification Accuracy: {best_result['classification_accuracy']:.2f}%")
    if best_result['adaptive_r1'] is not None:
        print(f"  Adaptive VPR R@1: {best_result['adaptive_r1']:.2f}%")
    print(f"  F1-Score: {best_result['f1']:.4f}")
    print(f"  Precision: {best_result['precision']:.4f}")
    print(f"  Recall: {best_result['recall']:.4f}")
    print(f"  Easy queries (predicted): {best_result['num_easy_pred']:.0f} ({100*best_result['num_easy_pred']/len(y_val):.1f}%)")
    print(f"  Hard queries (predicted): {best_result['num_hard_pred']:.0f} ({100*best_result['num_hard_pred']/len(y_val):.1f}%)")
    
    if full_reranking_r1 is not None:
        print(f"\nFull Re-ranking (Ground-truth): {full_reranking_r1:.2f}% R@1")
        if best_result['adaptive_r1'] is not None:
            performance_ratio = best_result['adaptive_r1'] / full_reranking_r1
            print(f"  Adaptive vs Full: {performance_ratio:.1%} ({best_result['adaptive_r1']:.2f}% / {full_reranking_r1:.2f}%)")
    
    print(f"\nModel's optimal threshold: {optimal_threshold:.3f}")
    optimal_classification = compute_recall_at_1_for_threshold(y_val, y_val_probs, optimal_threshold)
    print(f"  Classification Accuracy: {optimal_classification:.2f}%")
    if preds_dir is not None and inliers_dir is not None and preds_dir.exists() and inliers_dir.exists():
        optimal_is_hard = (y_val_probs < optimal_threshold).astype(bool)
        optimal_adaptive_r1 = compute_adaptive_r1_for_threshold(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=optimal_is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        print(f"  Adaptive VPR R@1: {optimal_adaptive_r1:.2f}%")
        if full_reranking_r1 is not None:
            optimal_ratio = optimal_adaptive_r1 / full_reranking_r1
            print(f"  Adaptive vs Full: {optimal_ratio:.1%} ({optimal_adaptive_r1:.2f}% / {full_reranking_r1:.2f}%)")
    print(f"{'='*70}\n")
    
    # Create plot
    thresholds_list = [r["threshold"] for r in results]
    classification_accuracy_list = [r["classification_accuracy"] for r in results]
    f1_list = [r["f1"] for r in results]
    adaptive_r1_list = [r["adaptive_r1"] for r in results if r["adaptive_r1"] is not None]
    
    # Determine what to plot
    plot_adaptive = len(adaptive_r1_list) > 0 and full_reranking_r1 is not None
    
    # Create multi-panel plot to clearly show why optimal threshold was chosen
    if plot_adaptive:
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.suptitle('Validation Threshold Analysis: Why Optimal Threshold Was Chosen', 
                     fontsize=18, fontweight='bold', y=0.995)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('Validation Threshold Analysis: Why Optimal Threshold Was Chosen', 
                     fontsize=18, fontweight='bold', y=0.995)
        axes = axes.reshape(1, -1)  # Make it 2D for consistent indexing
    
    # ===== PANEL 1: F1-Score vs Threshold (KEY: Shows why optimal threshold was chosen) =====
    ax1 = axes[0, 0] if plot_adaptive else axes[0, 0]
    
    ax1.plot(
        thresholds_list,
        f1_list,
        marker='o',
        markersize=5,
        linewidth=3,
        color='purple',
        alpha=0.8,
        label='F1-Score'
    )
    
    # Mark optimal threshold (where F1 is maximum)
    optimal_f1 = max(f1_list)
    optimal_f1_idx = f1_list.index(optimal_f1)
    optimal_f1_threshold = thresholds_list[optimal_f1_idx]
    
    ax1.plot(
        optimal_f1_threshold,
        optimal_f1,
        marker='*',
        markersize=30,
        color='red',
        markeredgecolor='darkred',
        markeredgewidth=3,
        zorder=15,
        label=f'Optimal Threshold: {optimal_f1_threshold:.3f}\n(F1-max: {optimal_f1:.4f})'
    )
    
    # Add vertical line
    ax1.axvline(x=optimal_f1_threshold, color='red', linestyle='--', alpha=0.6, linewidth=2)
    
    # Annotate
    ax1.annotate(
        f'MAX F1\n{optimal_f1_threshold:.3f}\nF1: {optimal_f1:.4f}',
        (optimal_f1_threshold, optimal_f1),
        textcoords="offset points",
        xytext=(25, 25),
        ha='left',
        fontsize=11,
        fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcoral', alpha=0.9, edgecolor='darkred', linewidth=2),
        arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='darkred', lw=2)
    )
    
    ax1.set_xlabel("Threshold", fontsize=12, fontweight='bold')
    ax1.set_ylabel("F1-Score", fontsize=12, fontweight='bold')
    ax1.set_title("Panel 1: F1-Score vs Threshold\n(Model chooses threshold that MAXIMIZES F1)", 
                  fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([min(thresholds_list) - 0.05, max(thresholds_list) + 0.05])
    
    # ===== PANEL 2: Classification Accuracy vs Threshold =====
    ax2 = axes[0, 1] if plot_adaptive else axes[0, 1]
    
    ax2.plot(
        thresholds_list,
        classification_accuracy_list,
        marker='o',
        markersize=4,
        linewidth=2,
        color='steelblue',
        alpha=0.7,
        label='Classification Accuracy'
    )
    
    optimal_classification = compute_recall_at_1_for_threshold(y_val, y_val_probs, optimal_threshold)
    ax2.plot(
        optimal_threshold,
        optimal_classification,
        marker='*',
        markersize=25,
        color='orange',
        markeredgecolor='darkorange',
        markeredgewidth=2,
        zorder=10,
        label=f'Optimal: {optimal_threshold:.3f}'
    )
    
    ax2.axvline(x=optimal_threshold, color='orange', linestyle='--', alpha=0.6, linewidth=2)
    
    ax2.set_xlabel("Threshold", fontsize=12, fontweight='bold')
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=12, fontweight='bold')
    ax2.set_title("Panel 2: Classification Accuracy vs Threshold\n(Accuracy at optimal threshold)", 
                  fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([min(thresholds_list) - 0.05, max(thresholds_list) + 0.05])
    
    # ===== PANEL 3: Adaptive VPR R@1 vs Threshold (with Full Re-ranking) =====
    if plot_adaptive:
        ax3 = axes[1, 0]
        
        adaptive_thresholds = [r["threshold"] for r in results if r["adaptive_r1"] is not None]
        ax3.plot(
            adaptive_thresholds,
            adaptive_r1_list,
            marker='s',
            markersize=5,
            linewidth=2.5,
            color='green',
            alpha=0.8,
            label='Adaptive VPR R@1'
        )
        
        # Full re-ranking ground-truth
        ax3.axhline(
            y=full_reranking_r1,
            color='red',
            linestyle='--',
            linewidth=3,
            alpha=0.8,
            label=f'Full Re-ranking (Ground-truth: {full_reranking_r1:.2f}%)'
        )
        
        # Mark threshold = 1.0 (should match full re-ranking since all queries are hard)
        max_threshold = max(thresholds_list)
        threshold_1_r1 = next((r["adaptive_r1"] for r in results if abs(r["threshold"] - 1.0) < 0.01), None)
        if threshold_1_r1 is not None:
            ax3.plot(
                max_threshold,
                threshold_1_r1,
                marker='*',
                markersize=25,
                color='red',
                markeredgecolor='darkred',
                markeredgewidth=2,
                zorder=15,
                label='Threshold = 1.0 (All Hard = Full Re-ranking)'
            )
            # Annotate threshold = 1.0
            ax3.annotate(
                f'Threshold = 1.0\nAll queries HARD\nR@1: {threshold_1_r1:.2f}%\n(Expected: {full_reranking_r1:.2f}%)',
                (max_threshold, threshold_1_r1),
                textcoords="offset points",
                xytext=(-30, 30),
                ha='right',
                fontsize=10,
                fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, edgecolor='red', linewidth=2),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='red', lw=2)
            )
        
        # Optimal threshold on adaptive curve
        optimal_is_hard = (y_val_probs < optimal_threshold).astype(bool)
        optimal_adaptive_r1 = compute_adaptive_r1_for_threshold(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=optimal_is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        ax3.plot(
            optimal_threshold,
            optimal_adaptive_r1,
            marker='D',
            markersize=18,
            color='darkgreen',
            markeredgecolor='green',
            markeredgewidth=2,
            zorder=12,
            label=f'Optimal: {optimal_threshold:.3f}'
        )
        
        ax3.axvline(x=optimal_threshold, color='green', linestyle=':', alpha=0.6, linewidth=2)
        
        ax3.set_xlabel("Threshold", fontsize=12, fontweight='bold')
        ax3.set_ylabel("VPR R@1 (%)", fontsize=12, fontweight='bold')
        ax3.set_title("Panel 3: Adaptive VPR R@1 vs Threshold\n(At threshold=1.0: All hard → Full re-ranking)", 
                      fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_xlim([min(thresholds_list) - 0.05, max(thresholds_list) + 0.05])
        
        # Add text explaining the relationship
        ax3.text(0.02, 0.98, 
                'Key Insight:\nHigher threshold → More hard queries → More re-ranking\nThreshold = 1.0 → All queries hard → Full re-ranking',
                transform=ax3.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        
        # ===== PANEL 4: Combined View (F1 + Adaptive R@1) =====
        ax4 = axes[1, 1]
        
        # F1-score (left y-axis)
        ax4_twin = ax4.twinx()
        line1 = ax4.plot(thresholds_list, f1_list, 'purple', linewidth=3, marker='o', markersize=4, 
                         label='F1-Score', alpha=0.8)
        ax4_twin.set_ylabel('F1-Score', fontsize=12, fontweight='bold', color='purple')
        ax4_twin.tick_params(axis='y', labelcolor='purple')
        ax4_twin.set_ylim([min(f1_list) - 0.05, max(f1_list) + 0.05])
        
        # Adaptive R@1 (right y-axis)
        line2 = ax4.plot(adaptive_thresholds, adaptive_r1_list, 'green', linewidth=2.5, marker='s', 
                         markersize=5, label='Adaptive VPR R@1', alpha=0.8)
        ax4.set_ylabel('Adaptive VPR R@1 (%)', fontsize=12, fontweight='bold', color='green')
        ax4.tick_params(axis='y', labelcolor='green')
        
        # Full re-ranking line
        ax4.axhline(y=full_reranking_r1, color='red', linestyle='--', linewidth=3, alpha=0.8,
                   label=f'Full Re-ranking: {full_reranking_r1:.2f}%')
        
        # Mark optimal threshold
        ax4.plot(optimal_threshold, optimal_f1, marker='*', markersize=30, color='red',
                markeredgecolor='darkred', markeredgewidth=3, zorder=15, label='Optimal Threshold')
        ax4.plot(optimal_threshold, optimal_adaptive_r1, marker='D', markersize=18, color='darkgreen',
                markeredgecolor='green', markeredgewidth=2, zorder=12)
        ax4.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.6, linewidth=2)
        
        # Combine legends
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax4.legend(lines, labels + ['Full Re-ranking', 'Optimal Threshold'], 
                  loc='best', fontsize=9)
        
        ax4.set_xlabel("Threshold", fontsize=12, fontweight='bold')
        ax4.set_title("Panel 4: F1-Score & Adaptive R@1 vs Threshold\n(Threshold → 1.0 → Full Re-ranking Performance)", 
                      fontsize=13, fontweight='bold')
        
        # Add annotation explaining the relationship
        ax4.text(0.02, 0.98, 
                'Key Insight:\n• Lower threshold → Fewer hard queries → Less re-ranking\n• Higher threshold → More hard queries → More re-ranking\n• Threshold = 1.0 → All queries hard → Full re-ranking',
                transform=ax4.transAxes,
                fontsize=9,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                family='monospace')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim([min(thresholds_list) - 0.05, max(thresholds_list) + 0.05])
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # Save plot
    if output_path is None:
        output_path = Path("validation_threshold_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot to: {output_path}")
    
    # Save results to text file
    results_file = output_path.with_suffix('.txt')
    with open(results_file, 'w') as f:
        f.write("Validation Threshold Analysis Results\n")
        f.write("="*70 + "\n\n")
        f.write(f"Model: {model_path}\n")
        f.write(f"Validation Features: {val_features_path}\n")
        f.write(f"Optimal Threshold: {optimal_threshold:.3f}\n")
        f.write(f"Threshold Method: {threshold_method}\n\n")
        if plot_adaptive:
            f.write(f"{'Threshold':<12} {'Class Acc (%)':<15} {'Adaptive R@1 (%)':<18} {'F1':<10} {'Precision':<12} {'Recall':<12} {'Easy Pred':<12} {'Hard Pred':<12}\n")
            f.write("-"*100 + "\n")
            for r in results:
                adaptive_r1_str = f"{r['adaptive_r1']:.2f}" if r['adaptive_r1'] is not None else "N/A"
                f.write(f"{r['threshold']:<12.3f} {r['classification_accuracy']:<15.2f} {adaptive_r1_str:<18} {r['f1']:<10.4f} "
                       f"{r['precision']:<12.4f} {r['recall']:<12.4f} {r['num_easy_pred']:<12.0f} {r['num_hard_pred']:<12.0f}\n")
            if full_reranking_r1 is not None:
                f.write(f"\nFull Re-ranking R@1 (Ground-truth): {full_reranking_r1:.2f}%\n")
        else:
            f.write(f"{'Threshold':<12} {'Class Acc (%)':<15} {'F1':<10} {'Precision':<12} {'Recall':<12} {'Easy Pred':<12} {'Hard Pred':<12}\n")
            f.write("-"*70 + "\n")
            for r in results:
                f.write(f"{r['threshold']:<12.3f} {r['classification_accuracy']:<15.2f} {r['f1']:<10.4f} "
                       f"{r['precision']:<12.4f} {r['recall']:<12.4f} {r['num_easy_pred']:<12.0f} {r['num_hard_pred']:<12.0f}\n")
    
    print(f"Saved results table to: {results_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation threshold analysis: Recall@1 accuracy vs threshold"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model (.pkl file)",
    )
    parser.add_argument(
        "--val-features",
        type=str,
        required=True,
        help="Path to validation features (.npz file)",
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.1, 0.95],
        help="Threshold range to test (min max)",
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size for threshold testing",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="validation_threshold_analysis.png",
        help="Output plot path",
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default=None,
        help="Path to predictions directory (for computing adaptive R@1 and full re-ranking)",
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        default=None,
        help="Path to inliers directory (for computing adaptive R@1 and full re-ranking)",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions to consider",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold (meters) for positive match",
    )
    
    args = parser.parse_args()
    
    preds_dir = Path(args.preds_dir) if args.preds_dir else None
    inliers_dir = Path(args.inliers_dir) if args.inliers_dir else None
    
    analyze_validation_thresholds(
        model_path=Path(args.model_path),
        val_features_path=Path(args.val_features),
        threshold_range=tuple(args.threshold_range),
        threshold_step=args.threshold_step,
        output_path=Path(args.output),
        preds_dir=preds_dir,
        inliers_dir=inliers_dir,
        num_preds=args.num_preds,
        positive_dist_threshold=args.positive_dist_threshold
    )


if __name__ == "__main__":
    main()

