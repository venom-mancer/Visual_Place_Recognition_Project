"""
Compare 3 models (Night+Sun, Night Only, Sun Only) on 4 test sets:
1. SF-XS test
2. Tokyo-XS test
3. SVOX Sun test
4. SVOX Night test

Generate comparison charts and tables.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


def load_model(model_path: str):
    """Load a trained model."""
    model_bundle = joblib.load(model_path)
    return {
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "optimal_threshold": model_bundle.get("optimal_threshold", 0.5),
        "optimal_C": model_bundle.get("optimal_C", 1.0),
        "feature_names": model_bundle.get("feature_names", [])
    }


def evaluate_model_on_test_set(model_data, test_features_path: str, test_name: str):
    """
    Evaluate a model on a test set.
    
    Returns:
        dict with metrics: accuracy, f1, precision, recall, hard_query_rate, easy_query_rate
    """
    # Load test features
    test_features = load_feature_file(test_features_path)
    X_test = build_feature_matrix(test_features, model_data["feature_names"])
    
    # Get labels
    y_test = test_features["labels"].astype("float32")
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]
    
    # Scale and predict
    X_test_scaled = model_data["scaler"].transform(X_test)
    y_test_probs = model_data["model"].predict_proba(X_test_scaled)[:, 1]
    
    # Apply optimal threshold
    threshold = model_data["optimal_threshold"]
    y_test_pred = (y_test_probs >= threshold).astype(int)
    
    # Compute metrics
    accuracy = (y_test_pred == y_test).mean()
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Hard query rate (queries predicted as hard = probability < threshold)
    hard_query_rate = (y_test_probs < threshold).mean()
    easy_query_rate = (y_test_probs >= threshold).mean()
    
    # Actual easy/hard distribution
    actual_easy_rate = y_test.mean()
    actual_hard_rate = 1 - actual_easy_rate
    
    return {
        "test_name": test_name,
        "num_queries": len(y_test),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hard_query_rate": hard_query_rate,
        "easy_query_rate": easy_query_rate,
        "actual_easy_rate": actual_easy_rate,
        "actual_hard_rate": actual_hard_rate,
        "threshold": threshold,
        "y_test": y_test,
        "y_test_probs": y_test_probs,
        "y_test_pred": y_test_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 models on 4 test sets"
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
        "--sf-xs-test-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_test_improved.npz",
        help="SF-XS test features"
    )
    parser.add_argument(
        "--tokyo-test-features",
        type=str,
        default="data/features_and_predictions/features_tokyo_xs_test_improved.npz",
        help="Tokyo-XS test features"
    )
    parser.add_argument(
        "--svox-sun-test-features",
        type=str,
        help="SVOX Sun test features (will be created if not provided)"
    )
    parser.add_argument(
        "--svox-night-test-features",
        type=str,
        help="SVOX Night test features (will be created if not provided)"
    )
    parser.add_argument(
        "--svox-test-features",
        type=str,
        default="data/features_and_predictions/features_svox_test_improved.npz",
        help="SVOX test features (full, for filtering)"
    )
    parser.add_argument(
        "--svox-test-preds-dir",
        type=str,
        help="SVOX test predictions directory (for filtering night/sun)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_test_comparison",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model1 = load_model(args.model1_path)
    model2 = load_model(args.model2_path)
    model3 = load_model(args.model3_path)
    
    print(f"  Model 1 (Night + Sun): threshold = {model1['optimal_threshold']:.3f}")
    print(f"  Model 2 (Night Only): threshold = {model2['optimal_threshold']:.3f}")
    print(f"  Model 3 (Sun Only): threshold = {model3['optimal_threshold']:.3f}")
    
    # Prepare test sets
    test_sets = {}
    
    # SF-XS test
    if Path(args.sf_xs_test_features).exists():
        test_sets["SF-XS Test"] = args.sf_xs_test_features
    else:
        print(f"Warning: SF-XS test features not found: {args.sf_xs_test_features}")
    
    # Tokyo-XS test
    if Path(args.tokyo_test_features).exists():
        test_sets["Tokyo-XS Test"] = args.tokyo_test_features
    else:
        print(f"Warning: Tokyo-XS test features not found: {args.tokyo_test_features}")
    
    # SVOX Sun test
    if args.svox_sun_test_features and Path(args.svox_sun_test_features).exists():
        test_sets["SVOX Sun Test"] = args.svox_sun_test_features
    elif args.svox_test_features and Path(args.svox_test_features).exists() and args.svox_test_preds_dir:
        # Filter SVOX test for sun
        print("\nFiltering SVOX test features for Sun subset...")
        svox_sun_path = output_dir.parent / "data" / "features_svox_test_sun_improved.npz"
        svox_sun_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use filter script
        import subprocess
        result = subprocess.run([
            sys.executable, "filter_svox_features_by_subset.py",
            "--input-features", args.svox_test_features,
            "--preds-dir", args.svox_test_preds_dir,
            "--subset", "sun",
            "--output-features", str(svox_sun_path),
            "--queries-night-dir", "data/svox/images/test/queries_night",
            "--queries-sun-dir", "data/svox/images/test/queries_sun"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and svox_sun_path.exists():
            test_sets["SVOX Sun Test"] = str(svox_sun_path)
            print(f"  Created: {svox_sun_path}")
        else:
            print(f"  Failed to create SVOX Sun test features")
            print(result.stderr)
    
    # SVOX Night test
    if args.svox_night_test_features and Path(args.svox_night_test_features).exists():
        test_sets["SVOX Night Test"] = args.svox_night_test_features
    elif args.svox_test_features and Path(args.svox_test_features).exists() and args.svox_test_preds_dir:
        # Filter SVOX test for night
        print("\nFiltering SVOX test features for Night subset...")
        svox_night_path = output_dir.parent / "data" / "features_svox_test_night_improved.npz"
        svox_night_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use filter script
        import subprocess
        result = subprocess.run([
            sys.executable, "filter_svox_features_by_subset.py",
            "--input-features", args.svox_test_features,
            "--preds-dir", args.svox_test_preds_dir,
            "--subset", "night",
            "--output-features", str(svox_night_path),
            "--queries-night-dir", "data/svox/images/test/queries_night",
            "--queries-sun-dir", "data/svox/images/test/queries_sun"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and svox_night_path.exists():
            test_sets["SVOX Night Test"] = str(svox_night_path)
            print(f"  Created: {svox_night_path}")
        else:
            print(f"  Failed to create SVOX Night test features")
            print(result.stderr)
    
    print(f"\nTest sets to evaluate: {list(test_sets.keys())}")
    
    # Evaluate all models on all test sets
    results = {}
    models = {
        "Model 1: Night + Sun": model1,
        "Model 2: Night Only": model2,
        "Model 3: Sun Only": model3
    }
    
    print("\nEvaluating models on test sets...")
    for model_name, model_data in models.items():
        results[model_name] = {}
        for test_name, test_features_path in test_sets.items():
            print(f"  {model_name} on {test_name}...")
            try:
                result = evaluate_model_on_test_set(model_data, test_features_path, test_name)
                results[model_name][test_name] = result
            except Exception as e:
                print(f"    ERROR: {e}")
                results[model_name][test_name] = None
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    table_path = output_dir / "models_test_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Models Comparison on Test Sets\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Model | Test Set | # Queries | Accuracy | F1-Score | Precision | Recall | Hard Query Rate | Threshold |\n")
        f.write("|-------|----------|-----------|----------|----------|-----------|--------|-----------------|-----------|\n")
        
        for model_name in models.keys():
            for test_name in test_sets.keys():
                if results[model_name].get(test_name) is not None:
                    r = results[model_name][test_name]
                    f.write(f"| {model_name} | {test_name} | {r['num_queries']} | "
                           f"{r['accuracy']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | "
                           f"{r['recall']:.3f} | {r['hard_query_rate']:.1%} | {r['threshold']:.3f} |\n")
                else:
                    f.write(f"| {model_name} | {test_name} | - | - | - | - | - | - | - |\n")
    
    print(f"  Saved: {table_path}")
    
    # Generate comparison charts
    print("\nGenerating comparison charts...")
    
    # Chart 1: F1-Score comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(test_sets))
    width = 0.25
    
    model1_f1 = [results["Model 1: Night + Sun"].get(test_name, {}).get("f1", 0) 
                 if results["Model 1: Night + Sun"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    model2_f1 = [results["Model 2: Night Only"].get(test_name, {}).get("f1", 0) 
                 if results["Model 2: Night Only"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    model3_f1 = [results["Model 3: Sun Only"].get(test_name, {}).get("f1", 0) 
                 if results["Model 3: Sun Only"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    
    ax1.bar(x - width, model1_f1, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax1.bar(x, model2_f1, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax1.bar(x + width, model3_f1, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_f1_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Chart 2: Hard Query Rate comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    model1_hard = [results["Model 1: Night + Sun"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 1: Night + Sun"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    model2_hard = [results["Model 2: Night Only"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 2: Night Only"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    model3_hard = [results["Model 3: Sun Only"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 3: Sun Only"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    
    ax2.bar(x - width, model1_hard, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax2.bar(x, model2_hard, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax2.bar(x + width, model3_hard, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Hard Query Rate Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart2_path = output_dir / "chart_hard_query_rate_comparison.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart2_path}")
    
    # Chart 3: Accuracy comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    model1_acc = [results["Model 1: Night + Sun"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 1: Night + Sun"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    model2_acc = [results["Model 2: Night Only"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 2: Night Only"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    model3_acc = [results["Model 3: Sun Only"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 3: Sun Only"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    
    ax3.bar(x - width, model1_acc, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax3.bar(x, model2_acc, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax3.bar(x + width, model3_acc, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Classification Accuracy Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart3_path = output_dir / "chart_accuracy_comparison.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart3_path}")
    
    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


1. SF-XS test
2. Tokyo-XS test
3. SVOX Sun test
4. SVOX Night test

Generate comparison charts and tables.
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from sklearn.metrics import f1_score, precision_score, recall_score
import sys
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix


def load_model(model_path: str):
    """Load a trained model."""
    model_bundle = joblib.load(model_path)
    return {
        "model": model_bundle["model"],
        "scaler": model_bundle["scaler"],
        "optimal_threshold": model_bundle.get("optimal_threshold", 0.5),
        "optimal_C": model_bundle.get("optimal_C", 1.0),
        "feature_names": model_bundle.get("feature_names", [])
    }


def evaluate_model_on_test_set(model_data, test_features_path: str, test_name: str):
    """
    Evaluate a model on a test set.
    
    Returns:
        dict with metrics: accuracy, f1, precision, recall, hard_query_rate, easy_query_rate
    """
    # Load test features
    test_features = load_feature_file(test_features_path)
    X_test = build_feature_matrix(test_features, model_data["feature_names"])
    
    # Get labels
    y_test = test_features["labels"].astype("float32")
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_test).any(axis=1)
    X_test = X_test[valid_mask]
    y_test = y_test[valid_mask]
    
    # Scale and predict
    X_test_scaled = model_data["scaler"].transform(X_test)
    y_test_probs = model_data["model"].predict_proba(X_test_scaled)[:, 1]
    
    # Apply optimal threshold
    threshold = model_data["optimal_threshold"]
    y_test_pred = (y_test_probs >= threshold).astype(int)
    
    # Compute metrics
    accuracy = (y_test_pred == y_test).mean()
    f1 = f1_score(y_test, y_test_pred, zero_division=0)
    precision = precision_score(y_test, y_test_pred, zero_division=0)
    recall = recall_score(y_test, y_test_pred, zero_division=0)
    
    # Hard query rate (queries predicted as hard = probability < threshold)
    hard_query_rate = (y_test_probs < threshold).mean()
    easy_query_rate = (y_test_probs >= threshold).mean()
    
    # Actual easy/hard distribution
    actual_easy_rate = y_test.mean()
    actual_hard_rate = 1 - actual_easy_rate
    
    return {
        "test_name": test_name,
        "num_queries": len(y_test),
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hard_query_rate": hard_query_rate,
        "easy_query_rate": easy_query_rate,
        "actual_easy_rate": actual_easy_rate,
        "actual_hard_rate": actual_hard_rate,
        "threshold": threshold,
        "y_test": y_test,
        "y_test_probs": y_test_probs,
        "y_test_pred": y_test_pred
    }


def main():
    parser = argparse.ArgumentParser(
        description="Compare 3 models on 4 test sets"
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
        "--sf-xs-test-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_test_improved.npz",
        help="SF-XS test features"
    )
    parser.add_argument(
        "--tokyo-test-features",
        type=str,
        default="data/features_and_predictions/features_tokyo_xs_test_improved.npz",
        help="Tokyo-XS test features"
    )
    parser.add_argument(
        "--svox-sun-test-features",
        type=str,
        help="SVOX Sun test features (will be created if not provided)"
    )
    parser.add_argument(
        "--svox-night-test-features",
        type=str,
        help="SVOX Night test features (will be created if not provided)"
    )
    parser.add_argument(
        "--svox-test-features",
        type=str,
        default="data/features_and_predictions/features_svox_test_improved.npz",
        help="SVOX test features (full, for filtering)"
    )
    parser.add_argument(
        "--svox-test-preds-dir",
        type=str,
        help="SVOX test predictions directory (for filtering night/sun)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/models_test_comparison",
        help="Output directory for results"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load models
    print("Loading models...")
    model1 = load_model(args.model1_path)
    model2 = load_model(args.model2_path)
    model3 = load_model(args.model3_path)
    
    print(f"  Model 1 (Night + Sun): threshold = {model1['optimal_threshold']:.3f}")
    print(f"  Model 2 (Night Only): threshold = {model2['optimal_threshold']:.3f}")
    print(f"  Model 3 (Sun Only): threshold = {model3['optimal_threshold']:.3f}")
    
    # Prepare test sets
    test_sets = {}
    
    # SF-XS test
    if Path(args.sf_xs_test_features).exists():
        test_sets["SF-XS Test"] = args.sf_xs_test_features
    else:
        print(f"Warning: SF-XS test features not found: {args.sf_xs_test_features}")
    
    # Tokyo-XS test
    if Path(args.tokyo_test_features).exists():
        test_sets["Tokyo-XS Test"] = args.tokyo_test_features
    else:
        print(f"Warning: Tokyo-XS test features not found: {args.tokyo_test_features}")
    
    # SVOX Sun test
    if args.svox_sun_test_features and Path(args.svox_sun_test_features).exists():
        test_sets["SVOX Sun Test"] = args.svox_sun_test_features
    elif args.svox_test_features and Path(args.svox_test_features).exists() and args.svox_test_preds_dir:
        # Filter SVOX test for sun
        print("\nFiltering SVOX test features for Sun subset...")
        svox_sun_path = output_dir.parent / "data" / "features_svox_test_sun_improved.npz"
        svox_sun_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use filter script
        import subprocess
        result = subprocess.run([
            sys.executable, "filter_svox_features_by_subset.py",
            "--input-features", args.svox_test_features,
            "--preds-dir", args.svox_test_preds_dir,
            "--subset", "sun",
            "--output-features", str(svox_sun_path),
            "--queries-night-dir", "data/svox/images/test/queries_night",
            "--queries-sun-dir", "data/svox/images/test/queries_sun"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and svox_sun_path.exists():
            test_sets["SVOX Sun Test"] = str(svox_sun_path)
            print(f"  Created: {svox_sun_path}")
        else:
            print(f"  Failed to create SVOX Sun test features")
            print(result.stderr)
    
    # SVOX Night test
    if args.svox_night_test_features and Path(args.svox_night_test_features).exists():
        test_sets["SVOX Night Test"] = args.svox_night_test_features
    elif args.svox_test_features and Path(args.svox_test_features).exists() and args.svox_test_preds_dir:
        # Filter SVOX test for night
        print("\nFiltering SVOX test features for Night subset...")
        svox_night_path = output_dir.parent / "data" / "features_svox_test_night_improved.npz"
        svox_night_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Use filter script
        import subprocess
        result = subprocess.run([
            sys.executable, "filter_svox_features_by_subset.py",
            "--input-features", args.svox_test_features,
            "--preds-dir", args.svox_test_preds_dir,
            "--subset", "night",
            "--output-features", str(svox_night_path),
            "--queries-night-dir", "data/svox/images/test/queries_night",
            "--queries-sun-dir", "data/svox/images/test/queries_sun"
        ], capture_output=True, text=True)
        
        if result.returncode == 0 and svox_night_path.exists():
            test_sets["SVOX Night Test"] = str(svox_night_path)
            print(f"  Created: {svox_night_path}")
        else:
            print(f"  Failed to create SVOX Night test features")
            print(result.stderr)
    
    print(f"\nTest sets to evaluate: {list(test_sets.keys())}")
    
    # Evaluate all models on all test sets
    results = {}
    models = {
        "Model 1: Night + Sun": model1,
        "Model 2: Night Only": model2,
        "Model 3: Sun Only": model3
    }
    
    print("\nEvaluating models on test sets...")
    for model_name, model_data in models.items():
        results[model_name] = {}
        for test_name, test_features_path in test_sets.items():
            print(f"  {model_name} on {test_name}...")
            try:
                result = evaluate_model_on_test_set(model_data, test_features_path, test_name)
                results[model_name][test_name] = result
            except Exception as e:
                print(f"    ERROR: {e}")
                results[model_name][test_name] = None
    
    # Generate comparison table
    print("\nGenerating comparison table...")
    table_path = output_dir / "models_test_comparison_table.md"
    with open(table_path, 'w') as f:
        f.write("# Models Comparison on Test Sets\n\n")
        f.write("## Summary Table\n\n")
        f.write("| Model | Test Set | # Queries | Accuracy | F1-Score | Precision | Recall | Hard Query Rate | Threshold |\n")
        f.write("|-------|----------|-----------|----------|----------|-----------|--------|-----------------|-----------|\n")
        
        for model_name in models.keys():
            for test_name in test_sets.keys():
                if results[model_name].get(test_name) is not None:
                    r = results[model_name][test_name]
                    f.write(f"| {model_name} | {test_name} | {r['num_queries']} | "
                           f"{r['accuracy']:.3f} | {r['f1']:.3f} | {r['precision']:.3f} | "
                           f"{r['recall']:.3f} | {r['hard_query_rate']:.1%} | {r['threshold']:.3f} |\n")
                else:
                    f.write(f"| {model_name} | {test_name} | - | - | - | - | - | - | - |\n")
    
    print(f"  Saved: {table_path}")
    
    # Generate comparison charts
    print("\nGenerating comparison charts...")
    
    # Chart 1: F1-Score comparison
    fig1, ax1 = plt.subplots(figsize=(14, 8))
    x = np.arange(len(test_sets))
    width = 0.25
    
    model1_f1 = [results["Model 1: Night + Sun"].get(test_name, {}).get("f1", 0) 
                 if results["Model 1: Night + Sun"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    model2_f1 = [results["Model 2: Night Only"].get(test_name, {}).get("f1", 0) 
                 if results["Model 2: Night Only"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    model3_f1 = [results["Model 3: Sun Only"].get(test_name, {}).get("f1", 0) 
                 if results["Model 3: Sun Only"].get(test_name) else 0 
                 for test_name in test_sets.keys()]
    
    ax1.bar(x - width, model1_f1, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax1.bar(x, model2_f1, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax1.bar(x + width, model3_f1, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax1.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax1.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax1.set_title('F1-Score Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax1.legend(loc='best', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart1_path = output_dir / "chart_f1_comparison.png"
    plt.savefig(chart1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart1_path}")
    
    # Chart 2: Hard Query Rate comparison
    fig2, ax2 = plt.subplots(figsize=(14, 8))
    
    model1_hard = [results["Model 1: Night + Sun"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 1: Night + Sun"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    model2_hard = [results["Model 2: Night Only"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 2: Night Only"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    model3_hard = [results["Model 3: Sun Only"].get(test_name, {}).get("hard_query_rate", 0) 
                   if results["Model 3: Sun Only"].get(test_name) else 0 
                   for test_name in test_sets.keys()]
    
    ax2.bar(x - width, model1_hard, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax2.bar(x, model2_hard, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax2.bar(x + width, model3_hard, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax2.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Hard Query Rate (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Hard Query Rate Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax2.legend(loc='best', fontsize=10)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart2_path = output_dir / "chart_hard_query_rate_comparison.png"
    plt.savefig(chart2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart2_path}")
    
    # Chart 3: Accuracy comparison
    fig3, ax3 = plt.subplots(figsize=(14, 8))
    
    model1_acc = [results["Model 1: Night + Sun"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 1: Night + Sun"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    model2_acc = [results["Model 2: Night Only"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 2: Night Only"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    model3_acc = [results["Model 3: Sun Only"].get(test_name, {}).get("accuracy", 0) 
                  if results["Model 3: Sun Only"].get(test_name) else 0 
                  for test_name in test_sets.keys()]
    
    ax3.bar(x - width, model1_acc, width, label='Model 1: Night + Sun', color='blue', alpha=0.7)
    ax3.bar(x, model2_acc, width, label='Model 2: Night Only', color='green', alpha=0.7)
    ax3.bar(x + width, model3_acc, width, label='Model 3: Sun Only', color='red', alpha=0.7)
    
    ax3.set_xlabel('Test Set', fontsize=12, fontweight='bold')
    ax3.set_ylabel('Classification Accuracy', fontsize=12, fontweight='bold')
    ax3.set_title('Classification Accuracy Comparison: All Models on All Test Sets', fontsize=14, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels(list(test_sets.keys()), rotation=15, ha='right')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    ax3.set_ylim([0, 1.0])
    
    plt.tight_layout()
    chart3_path = output_dir / "chart_accuracy_comparison.png"
    plt.savefig(chart3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {chart3_path}")
    
    print(f"\n{'='*70}")
    print("Comparison complete!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

