"""
Compare 3 models and generate 2 charts:
1. Chart 1: Metrics used for threshold selection (F1, precision, recall curves)
2. Chart 2: Recall@1 vs threshold for all models with chosen thresholds marked
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_3_train_logreg_easy_queries import load_feature_file, build_feature_matrix
from plot_validation_threshold_analysis import compute_adaptive_r1_for_threshold


def load_model_and_features(model_path: str, val_features_path: str):
    """Load model and validation features."""
    # Load model
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    scaler = model_bundle["scaler"]
    optimal_threshold = model_bundle.get("optimal_threshold", 0.5)
    optimal_C = model_bundle.get("optimal_C", 1.0)
    
    # Load validation features
    val_features = load_feature_file(val_features_path)
    X_val, y_val, feature_names = build_feature_matrix(val_features)
    
    # Handle NaNs
    val_valid_mask = ~np.isnan(X_val).any(axis=1)
    X_val = X_val[val_valid_mask]
    y_val = y_val[val_valid_mask]
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    y_val_probs = model.predict_proba(X_val_scaled)[:, 1]
    
    return {
        "model": model,
        "scaler": scaler,
        "optimal_threshold": optimal_threshold,
        "optimal_C": optimal_C,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_probs": y_val_probs,
        "feature_names": feature_names
    }


def evaluate_thresholds(y_true, y_probs, thresholds):
    """Evaluate model at different thresholds."""
    results = []
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = (y_pred == y_true).mean()
        
        results.append({
            "threshold": threshold,
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 models and generate comparison charts"
    )
    parser.add_argument(
        "--model1-path",
        type=str,
        required=True,
        help="Path to Model 1 (Night + Sun)"
    )
    parser.add_argument(
        "--model2-path",
        type=str,
        required=True,
        help="Path to Model 2 (Night Only)"
    )
    parser.add_argument(
        "--model3-path",
        type=str,
        required=True,
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
        help="Path to validation predictions directory (for computing adaptive R@1)"
    )
    parser.add_argument(
        "--inliers-dir",
        type=str,
        help="Path to validation inliers directory (for computing adaptive R@1)"
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions per query"
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=float,
        default=25.0,
        help="Positive distance threshold (meters)"
    )
    parser.add_argument(
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.1, 1.0],
        help="Threshold range to test"
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Threshold step size"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/three_models_comparison",
        help="Output directory for charts"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all 3 models
    print("Loading models and validation features...")
    model1_data = load_model_and_features(args.model1_path, args.val_features)
    model2_data = load_model_and_features(args.model2_path, args.val_features)
    model3_data = load_model_and_features(args.model3_path, args.val_features)
    
    print(f"\nModel 1 (Night + Sun): Optimal threshold = {model1_data['optimal_threshold']:.3f}")
    print(f"Model 2 (Night Only): Optimal threshold = {model2_data['optimal_threshold']:.3f}")
    print(f"Model 3 (Sun Only): Optimal threshold = {model3_data['optimal_threshold']:.3f}")
    
    # Test thresholds
    thresholds = np.arange(args.threshold_range[0], args.threshold_range[1] + args.threshold_step, args.threshold_step)
    print(f"\nEvaluating {len(thresholds)} thresholds from {args.threshold_range[0]:.2f} to {args.threshold_range[1]:.2f}...")
    
    # Evaluate each model at different thresholds
    print("\nEvaluating Model 1 (Night + Sun)...")
    model1_results = evaluate_thresholds(model1_data["y_val"], model1_data["y_val_probs"], thresholds)
    
    print("Evaluating Model 2 (Night Only)...")
    model2_results = evaluate_thresholds(model2_data["y_val"], model2_data["y_val_probs"], thresholds)
    
    print("Evaluating Model 3 (Sun Only)...")
    model3_results = evaluate_thresholds(model3_data["y_val"], model3_data["y_val_probs"], thresholds)
    
    # Compute adaptive R@1 if preds and inliers directories are provided
    compute_adaptive_r1 = args.preds_dir is not None and args.inliers_dir is not None
    if compute_adaptive_r1:
        print("\nComputing adaptive R@1 for each threshold...")
        preds_dir = Path(args.preds_dir)
        inliers_dir = Path(args.inliers_dir)
        
        for i, threshold in enumerate(tqdm(thresholds, desc="Computing adaptive R@1")):
            # Model 1
            is_hard_1 = (model1_data["y_val_probs"] < threshold).astype(bool)
            model1_results[i]["adaptive_r1"] = compute_adaptive_r1_for_threshold(
                preds_dir, inliers_dir, is_hard_1, args.num_preds, args.positive_dist_threshold
            )
            
            # Model 2
            is_hard_2 = (model2_data["y_val_probs"] < threshold).astype(bool)
            model2_results[i]["adaptive_r1"] = compute_adaptive_r1_for_threshold(
                preds_dir, inliers_dir, is_hard_2, args.num_preds, args.positive_dist_threshold
            )
            
            # Model 3
            is_hard_3 = (model3_data["y_val_probs"] < threshold).astype(bool)
            model3_results[i]["adaptive_r1"] = compute_adaptive_r1_for_threshold(
                preds_dir, inliers_dir, is_hard_3, args.num_preds, args.positive_dist_threshold
            )
    else:
        print("\nSkipping adaptive R@1 computation (preds-dir and inliers-dir not provided)")
        for results in [model1_results, model2_results, model3_results]:
            for r in results:
                r["adaptive_r1"] = None
    
    # ===== CHART 1: Metrics for Threshold Selection =====
    print("\nGenerating Chart 1: Metrics for Threshold Selection...")
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Chart 1: Metrics Used for Threshold Selection\n(All Models on SF-XS Validation Set)', 
                  fontsize=16, fontweight='bold')
    
    # F1-Score
    ax = axes1[0, 0]
    ax.plot([r["threshold"] for r in model1_results], [r["f1"] for r in model1_results], 
            'o-', label='Model 1: Night + Sun', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model2_results], [r["f1"] for r in model2_results], 
            's-', label='Model 2: Night Only', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model3_results], [r["f1"] for r in model3_results], 
            '^-', label='Model 3: Sun Only', linewidth=2, markersize=4)
    
    # Mark optimal thresholds
    ax.plot(model1_data["optimal_threshold"], 
            next(r["f1"] for r in model1_results if abs(r["threshold"] - model1_data["optimal_threshold"]) < 0.001),
            '*', markersize=20, color='blue', markeredgecolor='darkblue', markeredgewidth=2, zorder=10)
    ax.plot(model2_data["optimal_threshold"], 
            next(r["f1"] for r in model2_results if abs(r["threshold"] - model2_data["optimal_threshold"]) < 0.001),
            '*', markersize=20, color='green', markeredgecolor='darkgreen', markeredgewidth=2, zorder=10)
    ax.plot(model3_data["optimal_threshold"], 
            next(r["f1"] for r in model3_results if abs(r["threshold"] - model3_data["optimal_threshold"]) < 0.001),
            '*', markersize=20, color='red', markeredgecolor='darkred', markeredgewidth=2, zorder=10)
    
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_title('F1-Score vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Precision
    ax = axes1[0, 1]
    ax.plot([r["threshold"] for r in model1_results], [r["precision"] for r in model1_results], 
            'o-', label='Model 1: Night + Sun', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model2_results], [r["precision"] for r in model2_results], 
            's-', label='Model 2: Night Only', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model3_results], [r["precision"] for r in model3_results], 
            '^-', label='Model 3: Sun Only', linewidth=2, markersize=4)
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Recall
    ax = axes1[1, 0]
    ax.plot([r["threshold"] for r in model1_results], [r["recall"] for r in model1_results], 
            'o-', label='Model 1: Night + Sun', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model2_results], [r["recall"] for r in model2_results], 
            's-', label='Model 2: Night Only', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model3_results], [r["recall"] for r in model3_results], 
            '^-', label='Model 3: Sun Only', linewidth=2, markersize=4)
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Recall', fontsize=12, fontweight='bold')
    ax.set_title('Recall vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Classification Accuracy
    ax = axes1[1, 1]
    ax.plot([r["threshold"] for r in model1_results], [r["accuracy"] for r in model1_results], 
            'o-', label='Model 1: Night + Sun', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model2_results], [r["accuracy"] for r in model2_results], 
            's-', label='Model 2: Night Only', linewidth=2, markersize=4)
    ax.plot([r["threshold"] for r in model3_results], [r["accuracy"] for r in model3_results], 
            '^-', label='Model 3: Sun Only', linewidth=2, markersize=4)
    ax.set_xlabel('Threshold', fontsize=12, fontweight='bold')
    ax.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax.set_title('Classification Accuracy vs Threshold', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    chart1_path = output_dir / "chart1_metrics_for_threshold_selection.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # ===== CHART 2: Recall@1 vs Threshold =====
    if compute_adaptive_r1:
        print("\nGenerating Chart 2: Recall@1 vs Threshold...")
        fig2, ax2 = plt.subplots(1, 1, figsize=(14, 8))
        
        # Plot adaptive R@1 for each model
        ax2.plot([r["threshold"] for r in model1_results], 
                [r["adaptive_r1"] for r in model1_results if r["adaptive_r1"] is not None], 
                'o-', label='Model 1: Night + Sun', linewidth=2.5, markersize=5, color='blue')
        ax2.plot([r["threshold"] for r in model2_results], 
                [r["adaptive_r1"] for r in model2_results if r["adaptive_r1"] is not None], 
                's-', label='Model 2: Night Only', linewidth=2.5, markersize=5, color='green')
        ax2.plot([r["threshold"] for r in model3_results], 
                [r["adaptive_r1"] for r in model3_results if r["adaptive_r1"] is not None], 
                '^-', label='Model 3: Sun Only', linewidth=2.5, markersize=5, color='red')
        
        # Mark chosen thresholds
        r1_at_opt1 = next((r["adaptive_r1"] for r in model1_results 
                          if abs(r["threshold"] - model1_data["optimal_threshold"]) < 0.001 
                          and r["adaptive_r1"] is not None), None)
        if r1_at_opt1:
            ax2.plot(model1_data["optimal_threshold"], r1_at_opt1,
                    '*', markersize=25, color='blue', markeredgecolor='darkblue', 
                    markeredgewidth=2, zorder=10, label=f'Model 1 Optimal (t={model1_data["optimal_threshold"]:.3f})')
        
        r1_at_opt2 = next((r["adaptive_r1"] for r in model2_results 
                          if abs(r["threshold"] - model2_data["optimal_threshold"]) < 0.001 
                          and r["adaptive_r1"] is not None), None)
        if r1_at_opt2:
            ax2.plot(model2_data["optimal_threshold"], r1_at_opt2,
                    '*', markersize=25, color='green', markeredgecolor='darkgreen', 
                    markeredgewidth=2, zorder=10, label=f'Model 2 Optimal (t={model2_data["optimal_threshold"]:.3f})')
        
        r1_at_opt3 = next((r["adaptive_r1"] for r in model3_results 
                          if abs(r["threshold"] - model3_data["optimal_threshold"]) < 0.001 
                          and r["adaptive_r1"] is not None), None)
        if r1_at_opt3:
            ax2.plot(model3_data["optimal_threshold"], r1_at_opt3,
                    '*', markersize=25, color='red', markeredgecolor='darkred', 
                    markeredgewidth=2, zorder=10, label=f'Model 3 Optimal (t={model3_data["optimal_threshold"]:.3f})')
        
        ax2.set_xlabel('Threshold', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Adaptive VPR Recall@1 (%)', fontsize=14, fontweight='bold')
        ax2.set_title('Chart 2: Adaptive VPR Recall@1 vs Threshold\n(All Models on SF-XS Validation Set)', 
                     fontsize=16, fontweight='bold')
        ax2.legend(loc='best', fontsize=11, framealpha=0.9)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        chart2_path = output_dir / "chart2_recall_at_1_vs_threshold.png"
        plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {chart2_path}")
    else:
        print("\nSkipping Chart 2 (adaptive R@1 not computed)")
    
    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

