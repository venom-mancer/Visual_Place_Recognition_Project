"""
Plot VPR Recall@1 vs Threshold for all 3 models on validation set.

This chart shows how the actual VPR Recall@1 accuracy changes as we vary
the threshold, and marks the selected thresholds for each model.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import sys
from glob import glob
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix
from util import get_list_distances_from_preds


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


def evaluate_model_recall1_at_thresholds(
    model_path: Path,
    val_features_path: Path,
    preds_dir: Path,
    inliers_dir: Path,
    threshold_range: tuple = (0.1, 1.0),
    threshold_step: float = 0.01,
    num_preds: int = 20,
    positive_dist_threshold: int = 25
):
    """
    Evaluate a model's VPR Recall@1 at different thresholds.
    
    Returns:
        dict with keys: thresholds, recall1_scores, optimal_threshold, model_name, full_reranking_r1
    """
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    optimal_threshold = bundle.get("optimal_threshold", 0.5)
    threshold_method = bundle.get("threshold_method", "f1")
    target_type = bundle.get("target_type", "hard_score")  # Default to hard_score (new approach)
    
    # Determine model name from path
    model_name = model_path.stem
    if "night_sun" in model_name or "night+sun" in model_name.lower():
        model_name = "Night + Sun"
    elif "night_only" in model_name or "night" in model_name.lower():
        model_name = "Night Only"
    elif "sun_only" in model_name or "sun" in model_name.lower():
        model_name = "Sun Only"
    else:
        model_name = model_path.stem
    
    # Load validation features
    val_features = load_feature_file(str(val_features_path))
    X_val = build_feature_matrix(val_features, feature_names)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[valid_mask]
    
    # Get labels
    labels = val_features["labels"][valid_mask].astype("float32")  # 1 = correct, 0 = wrong
    
    # Determine target based on model type
    if target_type == "hard_score":
        # Model predicts hard queries (1 = hard/wrong, 0 = easy/correct)
        # For adaptive matching: if P(hard) >= threshold, then is_hard = True
        # So: is_hard = (probs >= threshold)
        pass  # We'll use probs directly
    else:
        # Model predicts easy queries (1 = easy/correct, 0 = hard/wrong)
        # For adaptive matching: if P(easy) < threshold, then is_hard = True
        # So: is_hard = (probs < threshold)
        pass  # We'll handle this in the loop
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    y_probs = model.predict_proba(X_val_scaled)[:, 1]  # Probability of class 1
    
    # Compute full re-ranking R@1 (ground-truth)
    print(f"  Computing full re-ranking R@1 (ground-truth)...")
    full_reranking_r1 = compute_full_reranking_r1(
        preds_dir=preds_dir,
        inliers_dir=inliers_dir,
        num_preds=num_preds,
        positive_dist_threshold=positive_dist_threshold
    )
    print(f"  Full re-ranking R@1: {full_reranking_r1:.2f}%")
    
    # Test different thresholds
    threshold_max = max(threshold_range[1], 1.0)
    thresholds = np.arange(threshold_range[0], threshold_max + threshold_step, threshold_step)
    if 1.0 not in thresholds:
        thresholds = np.append(thresholds, 1.0)
    thresholds = np.sort(np.unique(thresholds))
    
    print(f"  Testing {len(thresholds)} thresholds...")
    
    recall1_scores = []
    
    for threshold in tqdm(thresholds, desc=f"  {model_name}"):
        # Determine is_hard based on model type
        if target_type == "hard_score":
            # Model predicts hard: if P(hard) >= threshold, then is_hard = True
            is_hard = (y_probs >= threshold).astype(bool)
        else:
            # Model predicts easy: if P(easy) < threshold, then is_hard = True
            is_hard = (y_probs < threshold).astype(bool)
        
        # Compute adaptive R@1
        adaptive_r1 = compute_adaptive_r1_for_threshold(
            preds_dir=preds_dir,
            inliers_dir=inliers_dir,
            is_hard=is_hard,
            num_preds=num_preds,
            positive_dist_threshold=positive_dist_threshold
        )
        
        recall1_scores.append(adaptive_r1)
    
    return {
        "thresholds": thresholds,
        "recall1_scores": np.array(recall1_scores),
        "optimal_threshold": optimal_threshold,
        "threshold_method": threshold_method,
        "model_name": model_name,
        "target_type": target_type,
        "full_reranking_r1": full_reranking_r1
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot VPR Recall@1 vs Threshold for all 3 models on validation set"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_sun.pkl",
        help="Path to Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_night_only.pkl",
        help="Path to Model 2 (Night Only)"
    )
    parser.add_argument(
        "--model3-path",
        type=str,
        default="models_three_way_comparison/logreg_easy_sun_only.pkl",
        help="Path to Model 3 (Sun Only)"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Path to validation features"
    )
    parser.add_argument(
        "--preds-dir",
        type=str,
        default=None,
        help="Path to validation predictions directory (optional, for VPR Recall@1)"
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        default=None,
        help="Path to validation inliers directory (optional, for VPR Recall@1)"
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.1, 1.0],
        help="Threshold range to test (min max)"
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.02,
        help="Step size for threshold testing (use larger step for faster computation)"
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions per query"
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold (meters) for positive match"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/recall1_vs_threshold_validation",
        help="Output directory for chart"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    preds_dir = Path(args.preds_dir) if args.preds_dir else None
    inliers_dir = Path(args.inliers_dir) if args.inliers_dir else None
    
    compute_vpr_r1 = preds_dir is not None and inliers_dir is not None
    if compute_vpr_r1:
        if not preds_dir.exists():
            print(f"Warning: Predictions directory not found: {preds_dir}")
            print("  Will use classification accuracy instead of VPR Recall@1")
            compute_vpr_r1 = False
        if not inliers_dir.exists():
            print(f"Warning: Inliers directory not found: {inliers_dir}")
            print("  Will use classification accuracy instead of VPR Recall@1")
            compute_vpr_r1 = False
    else:
        print("Note: No preds/inliers directories provided.")
        print("  Will use classification accuracy instead of VPR Recall@1")
    
    print(f"\n{'='*70}")
    print("VPR Recall@1 vs Threshold Analysis (Validation Set)")
    print(f"{'='*70}\n")
    
    # Evaluate all 3 models
    if compute_vpr_r1:
        print("Evaluating Model 1 (Night + Sun)...")
        results1 = evaluate_model_recall1_at_thresholds(
            Path(args.model1_path),
            Path(args.val_features),
            preds_dir,
            inliers_dir,
            tuple(args.threshold_range),
            args.threshold_step,
            args.num_preds,
            args.positive_dist_threshold
        )
        
        print("\nEvaluating Model 2 (Night Only)...")
        results2 = evaluate_model_recall1_at_thresholds(
            Path(args.model2_path),
            Path(args.val_features),
            preds_dir,
            inliers_dir,
            tuple(args.threshold_range),
            args.threshold_step,
            args.num_preds,
            args.positive_dist_threshold
        )
        
        print("\nEvaluating Model 3 (Sun Only)...")
        results3 = evaluate_model_recall1_at_thresholds(
            Path(args.model3_path),
            Path(args.val_features),
            preds_dir,
            inliers_dir,
            tuple(args.threshold_range),
            args.threshold_step,
            args.num_preds,
            args.positive_dist_threshold
        )
    else:
        # Use classification accuracy instead
        from plot_validation_thresholds_all_models import evaluate_model_at_thresholds
        print("Evaluating Model 1 (Night + Sun)...")
        results1 = evaluate_model_at_thresholds(
            Path(args.model1_path),
            Path(args.val_features),
            tuple(args.threshold_range),
            args.threshold_step
        )
        results1["recall1_scores"] = results1["accuracy_scores"] * 100  # Convert to percentage
        results1["full_reranking_r1"] = None
        
        print("\nEvaluating Model 2 (Night Only)...")
        results2 = evaluate_model_at_thresholds(
            Path(args.model2_path),
            Path(args.val_features),
            tuple(args.threshold_range),
            args.threshold_step
        )
        results2["recall1_scores"] = results2["accuracy_scores"] * 100
        results2["full_reranking_r1"] = None
        
        print("\nEvaluating Model 3 (Sun Only)...")
        results3 = evaluate_model_at_thresholds(
            Path(args.model3_path),
            Path(args.val_features),
            tuple(args.threshold_range),
            args.threshold_step
        )
        results3["recall1_scores"] = results3["accuracy_scores"] * 100
        results3["full_reranking_r1"] = None
    
    all_results = [results1, results2, results3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']
    
    # Get full re-ranking R@1 (should be same for all models, if available)
    full_reranking_r1 = results1.get("full_reranking_r1")
    
    # ===== CHART: VPR Recall@1 vs Threshold =====
    fig, ax = plt.subplots(1, 1, figsize=(14, 9))
    fig.suptitle('VPR Recall@1 vs Threshold (Validation Set)', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    for i, results in enumerate(all_results):
        model_name = results["model_name"]
        thresholds = results["thresholds"]
        recall1_scores = results["recall1_scores"]
        optimal_threshold = results["optimal_threshold"]
        
        # Find optimal threshold index
        optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
        optimal_recall1 = recall1_scores[optimal_idx]
        
        # Plot curve
        ax.plot(
            thresholds,
            recall1_scores,
            marker=markers[i],
            markersize=6,
            linewidth=2.5,
            color=colors[i],
            alpha=0.8,
            label=model_name
        )
        
        # Mark selected threshold
        ax.plot(
            optimal_threshold,
            optimal_recall1,
            marker='*',
            markersize=30,
            color=colors[i],
            markeredgecolor='black',
            markeredgewidth=2,
            zorder=10,
            label=f'{model_name} (Selected: {optimal_threshold:.3f})'
        )
        
        # Add vertical line
        ax.axvline(x=optimal_threshold, color=colors[i], linestyle='--', 
                  alpha=0.6, linewidth=2)
        
        # Annotate
        ax.annotate(
            f'{model_name}\nThreshold: {optimal_threshold:.3f}\nR@1: {optimal_recall1:.2f}%',
            (optimal_threshold, optimal_recall1),
            textcoords="offset points",
            xytext=(25, 25),
            ha='left',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3, 
                     edgecolor=colors[i], linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                          color=colors[i], lw=2)
        )
    
    # Add full re-ranking line (ground-truth) if available
    if full_reranking_r1 is not None:
        ax.axhline(
            y=full_reranking_r1,
            color='red',
            linestyle='--',
            linewidth=3,
            alpha=0.8,
            label=f'Full Re-ranking (Ground-truth: {full_reranking_r1:.2f}%)',
            zorder=5
        )
        
        # Mark threshold = 1.0 (should match full re-ranking)
        threshold_1_idx = np.argmin(np.abs(results1["thresholds"] - 1.0))
        threshold_1_r1 = results1["recall1_scores"][threshold_1_idx]
        ax.plot(
            1.0,
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
        ax.annotate(
            f'Threshold = 1.0\nAll queries HARD\nR@1: {threshold_1_r1:.2f}%\n(Expected: {full_reranking_r1:.2f}%)',
            (1.0, threshold_1_r1),
            textcoords="offset points",
            xytext=(-30, 30),
            ha='right',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9, 
                     edgecolor='red', linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', color='red', lw=2)
        )
    
    ax.set_xlabel("Threshold", fontsize=13, fontweight='bold')
    if compute_vpr_r1:
        ylabel = "VPR Recall@1 (%)"
        title_suffix = "VPR Recall@1 vs Threshold\n(Selected thresholds marked with stars, Full re-ranking shown as dashed line)"
    else:
        ylabel = "Classification Accuracy (%)"
        title_suffix = "Classification Accuracy vs Threshold\n(Selected thresholds marked with stars)"
    
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title_suffix, fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    # Add text explaining the relationship
    if compute_vpr_r1:
        insight_text = 'Key Insight:\n• Higher threshold → More hard queries → More re-ranking\n• Threshold = 1.0 → All queries hard → Full re-ranking\n• Selected thresholds optimize F1-score on validation'
    else:
        insight_text = 'Key Insight:\n• Higher threshold → More hard queries predicted\n• Selected thresholds optimize F1-score on validation\n• (Note: This shows classification accuracy, not VPR Recall@1)'
    
    ax.text(0.02, 0.98, 
            insight_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            family='monospace')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart_path = output_dir / "chart_recall1_vs_threshold_validation.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nSaved chart to: {chart_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    if compute_vpr_r1:
        print("VPR Recall@1 vs Threshold Summary")
    else:
        print("Classification Accuracy vs Threshold Summary")
    print(f"{'='*70}")
    if full_reranking_r1 is not None:
        print(f"Full Re-ranking (Ground-truth): {full_reranking_r1:.2f}% R@1")
    print(f"\nAt Selected Thresholds:")
    for results in all_results:
        optimal_idx = np.argmin(np.abs(results["thresholds"] - results["optimal_threshold"]))
        optimal_score = results["recall1_scores"][optimal_idx]
        print(f"\n{results['model_name']}:")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.3f}")
        if compute_vpr_r1:
            print(f"  VPR Recall@1: {optimal_score:.2f}%")
            if full_reranking_r1 is not None:
                performance_ratio = optimal_score / full_reranking_r1
                print(f"  Performance Ratio: {performance_ratio:.1%} ({optimal_score:.2f}% / {full_reranking_r1:.2f}%)")
        else:
            print(f"  Classification Accuracy: {optimal_score:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
