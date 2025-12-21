"""
Plot validation threshold analysis for all 3 models:
1. Chart 1: Metrics used for threshold selection (F1, Precision, Recall, Accuracy) vs Threshold
2. Chart 2: Model Accuracy vs Threshold

Shows selected thresholds for all 3 models.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from tqdm import tqdm
import sys

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


def evaluate_model_at_thresholds(
    model_path: Path,
    val_features_path: Path,
    threshold_range: tuple = (0.1, 0.95),
    threshold_step: float = 0.01
):
    """
    Evaluate a model at different thresholds on validation set.
    
    Returns:
        dict with keys: thresholds, f1_scores, precision_scores, recall_scores, 
                       accuracy_scores, optimal_threshold, model_name
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
        y_true = (1 - labels).astype("float32")  # 1 = hard/wrong, 0 = easy/correct
    else:
        # Model predicts easy queries (1 = easy/correct, 0 = hard/wrong)
        y_true = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
    
    # Scale and predict
    X_val_scaled = scaler.transform(X_val)
    y_probs = model.predict_proba(X_val_scaled)[:, 1]  # Probability of class 1
    
    # Test different thresholds
    threshold_max = max(threshold_range[1], 1.0)
    thresholds = np.arange(threshold_range[0], threshold_max + threshold_step, threshold_step)
    if 1.0 not in thresholds:
        thresholds = np.append(thresholds, 1.0)
    thresholds = np.sort(np.unique(thresholds))
    
    f1_scores = []
    precision_scores = []
    recall_scores = []
    accuracy_scores = []
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        f1 = f1_score(y_true, y_pred, zero_division=0)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)
        
        f1_scores.append(f1)
        precision_scores.append(precision)
        recall_scores.append(recall)
        accuracy_scores.append(accuracy)
    
    return {
        "thresholds": thresholds,
        "f1_scores": np.array(f1_scores),
        "precision_scores": np.array(precision_scores),
        "recall_scores": np.array(recall_scores),
        "accuracy_scores": np.array(accuracy_scores),
        "optimal_threshold": optimal_threshold,
        "threshold_method": threshold_method,
        "model_name": model_name,
        "target_type": target_type
    }


def main():
    parser = argparse.ArgumentParser(
        description="Plot validation threshold analysis for all 3 models"
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
        "--threshold-range",
        type=float,
        nargs=2,
        default=[0.1, 0.95],
        help="Threshold range to test (min max)"
    )
    parser.add_argument(
        "--threshold-step",
        type=float,
        default=0.01,
        help="Step size for threshold testing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/validation_threshold_analysis_all_models",
        help="Output directory for plots"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate all 3 models
    print("Evaluating Model 1 (Night + Sun)...")
    results1 = evaluate_model_at_thresholds(
        Path(args.model1_path),
        Path(args.val_features),
        tuple(args.threshold_range),
        args.threshold_step
    )
    
    print("Evaluating Model 2 (Night Only)...")
    results2 = evaluate_model_at_thresholds(
        Path(args.model2_path),
        Path(args.val_features),
        tuple(args.threshold_range),
        args.threshold_step
    )
    
    print("Evaluating Model 3 (Sun Only)...")
    results3 = evaluate_model_at_thresholds(
        Path(args.model3_path),
        Path(args.val_features),
        tuple(args.threshold_range),
        args.threshold_step
    )
    
    all_results = [results1, results2, results3]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 's', '^']
    
    # ===== CHART 1: Metrics for Threshold Selection =====
    fig1, axes1 = plt.subplots(2, 2, figsize=(16, 12))
    fig1.suptitle('Chart 1: Metrics Used for Threshold Selection (Validation Set)', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    metrics = [
        ("F1-Score", "f1_scores", axes1[0, 0]),
        ("Precision", "precision_scores", axes1[0, 1]),
        ("Recall", "recall_scores", axes1[1, 0]),
        ("Classification Accuracy", "accuracy_scores", axes1[1, 1])
    ]
    
    for metric_name, metric_key, ax in metrics:
        for i, results in enumerate(all_results):
            model_name = results["model_name"]
            thresholds = results["thresholds"]
            metric_values = results[metric_key]
            optimal_threshold = results["optimal_threshold"]
            
            # Find optimal threshold index
            optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
            optimal_metric_value = metric_values[optimal_idx]
            
            # Plot curve
            ax.plot(
                thresholds,
                metric_values,
                marker=markers[i],
                markersize=4,
                linewidth=2,
                color=colors[i],
                alpha=0.7,
                label=model_name
            )
            
            # Mark selected threshold
            ax.plot(
                optimal_threshold,
                optimal_metric_value,
                marker='*',
                markersize=20,
                color=colors[i],
                markeredgecolor='black',
                markeredgewidth=1.5,
                zorder=10,
                label=f'{model_name} (Selected: {optimal_threshold:.3f})'
            )
            
            # Add vertical line
            ax.axvline(x=optimal_threshold, color=colors[i], linestyle='--', 
                      alpha=0.5, linewidth=1.5)
        
        ax.set_xlabel("Threshold", fontsize=12, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=12, fontweight='bold')
        ax.set_title(f"{metric_name} vs Threshold\n(Selected thresholds marked with stars)", 
                    fontsize=13, fontweight='bold')
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    chart1_path = output_dir / "chart1_metrics_for_threshold_selection.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Chart 1 to: {chart1_path}")
    
    # ===== CHART 2: Model Accuracy vs Threshold =====
    fig2, ax2 = plt.subplots(1, 1, figsize=(12, 8))
    fig2.suptitle('Chart 2: Model Accuracy vs Threshold (Validation Set)', 
                  fontsize=16, fontweight='bold', y=0.98)
    
    for i, results in enumerate(all_results):
        model_name = results["model_name"]
        thresholds = results["thresholds"]
        accuracy_scores = results["accuracy_scores"]
        optimal_threshold = results["optimal_threshold"]
        
        # Find optimal threshold index
        optimal_idx = np.argmin(np.abs(thresholds - optimal_threshold))
        optimal_accuracy = accuracy_scores[optimal_idx]
        
        # Plot curve
        ax2.plot(
            thresholds,
            accuracy_scores * 100,  # Convert to percentage
            marker=markers[i],
            markersize=5,
            linewidth=2.5,
            color=colors[i],
            alpha=0.8,
            label=model_name
        )
        
        # Mark selected threshold
        ax2.plot(
            optimal_threshold,
            optimal_accuracy * 100,
            marker='*',
            markersize=25,
            color=colors[i],
            markeredgecolor='black',
            markeredgewidth=2,
            zorder=10,
            label=f'{model_name} (Selected: {optimal_threshold:.3f})'
        )
        
        # Add vertical line
        ax2.axvline(x=optimal_threshold, color=colors[i], linestyle='--', 
                   alpha=0.6, linewidth=2)
        
        # Annotate
        ax2.annotate(
            f'{model_name}\nThreshold: {optimal_threshold:.3f}\nAccuracy: {optimal_accuracy*100:.2f}%',
            (optimal_threshold, optimal_accuracy * 100),
            textcoords="offset points",
            xytext=(20, 20),
            ha='left',
            fontsize=10,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.5', facecolor=colors[i], alpha=0.3, 
                     edgecolor=colors[i], linewidth=2),
            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.3', 
                          color=colors[i], lw=2)
        )
    
    ax2.set_xlabel("Threshold", fontsize=13, fontweight='bold')
    ax2.set_ylabel("Classification Accuracy (%)", fontsize=13, fontweight='bold')
    ax2.set_title("Model Accuracy vs Threshold\n(Selected thresholds marked with stars)", 
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    chart2_path = output_dir / "chart2_accuracy_vs_threshold.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Chart 2 to: {chart2_path}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("Validation Threshold Analysis Summary")
    print(f"{'='*70}")
    for results in all_results:
        optimal_idx = np.argmin(np.abs(results["thresholds"] - results["optimal_threshold"]))
        print(f"\n{results['model_name']}:")
        print(f"  Optimal Threshold: {results['optimal_threshold']:.3f} ({results['threshold_method']})")
        print(f"  F1-Score: {results['f1_scores'][optimal_idx]:.4f}")
        print(f"  Precision: {results['precision_scores'][optimal_idx]:.4f}")
        print(f"  Recall: {results['recall_scores'][optimal_idx]:.4f}")
        print(f"  Accuracy: {results['accuracy_scores'][optimal_idx]*100:.2f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
