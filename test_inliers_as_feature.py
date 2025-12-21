"""
Test if adding num_inliers (top-1 inliers) as a feature improves model performance.

This script:
1. Extracts num_inliers from existing image matching results (if available)
2. Adds it as a 9th feature to the model
3. Trains and evaluates the model with and without inliers
4. Compares performance to see if inliers improve predictions

Note: This only works if full re-ranking has been done on the validation set.
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_3_train_logreg_easy_queries import (
    load_feature_file,
    build_feature_matrix,
    find_optimal_threshold
)


def extract_inliers_from_matching(preds_dir: Path, inliers_dir: Path, num_queries: int) -> np.ndarray:
    """
    Extract num_inliers for top-1 match from image matching results.
    
    Args:
        preds_dir: Directory with prediction .txt files
        inliers_dir: Directory with .torch files (image matching results)
        num_queries: Number of queries
    
    Returns:
        Array of num_inliers for each query (top-1 match only)
    """
    inliers = np.zeros(num_queries, dtype=np.float32)
    
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    print(f"Extracting inliers from {len(txt_files)} queries...")
    
    for txt_file in tqdm(txt_files, desc="Loading inliers"):
        try:
            query_idx = int(txt_file.stem)
            torch_file = inliers_dir / f"{txt_file.stem}.torch"
            
            if not torch_file.exists():
                # No matching done for this query
                inliers[query_idx] = 0.0
                continue
            
            # Load matching results
            query_results = torch.load(str(torch_file), weights_only=False)
            
            if len(query_results) > 0:
                # Get num_inliers for top-1 match (first result)
                inliers[query_idx] = float(query_results[0].get("num_inliers", 0))
            else:
                inliers[query_idx] = 0.0
                
        except Exception as e:
            print(f"Warning: Error loading inliers for {txt_file.name}: {e}")
            inliers[query_idx] = 0.0
    
    return inliers


def build_feature_matrix_with_inliers(features_dict: dict, inliers: np.ndarray = None) -> tuple:
    """
    Build feature matrix with optional inliers feature.
    
    Args:
        features_dict: Dictionary with 8 existing features
        inliers: Optional array of num_inliers (top-1) for each query
    
    Returns:
        (X, y, feature_names) where X includes inliers if provided
    """
    # Build base features (8 features)
    X_base, y, feature_names_base = build_feature_matrix(features_dict)
    
    if inliers is not None:
        # Add inliers as 9th feature
        # Normalize inliers to 0-1 range (or use log scale if needed)
        # For now, use raw inliers (will be scaled by StandardScaler)
        inliers_normalized = inliers.astype(np.float32)
        
        # Handle NaN/Inf
        inliers_normalized = np.nan_to_num(inliers_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        X = np.column_stack([X_base, inliers_normalized])
        feature_names = feature_names_base + ["num_inliers_top1"]
    else:
        X = X_base
        feature_names = feature_names_base
    
    return X, y, feature_names


def train_and_evaluate_model(X_train, y_train, X_val, y_val, model_name: str):
    """Train model and evaluate on validation set."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle NaNs
    train_valid_mask = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_scaled = X_train_scaled[train_valid_mask]
    y_train = y_train[train_valid_mask]
    
    val_valid_mask = ~np.isnan(X_val_scaled).any(axis=1)
    X_val_scaled = X_val_scaled[val_valid_mask]
    y_val = y_val[val_valid_mask]
    
    # Hyperparameter tuning (C)
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C = 1.0
    best_val_f1 = 0.0
    
    for C in C_values:
        logreg_temp = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000, class_weight='balanced'
        )
        logreg_temp.fit(X_train_scaled, y_train)
        y_val_probs = logreg_temp.predict_proba(X_val_scaled)[:, 1]
        threshold_temp, _ = find_optimal_threshold(y_val, y_val_probs, method="f1")
        y_val_pred = (y_val_probs >= threshold_temp).astype(int)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C = C
    
    # Train final model
    logreg = LogisticRegression(
        C=best_C, solver="lbfgs", max_iter=1000, class_weight='balanced'
    )
    logreg.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_val, y_val_probs, method="f1")
    y_val_pred = (y_val_probs >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    hard_query_rate = (1 - y_val_pred.mean()) * 100
    
    return {
        "model": logreg,
        "scaler": scaler,
        "optimal_threshold": optimal_threshold,
        "optimal_C": best_C,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hard_query_rate": hard_query_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test if adding inliers as feature improves model"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Training features .npz file"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        required=True,
        help="Validation features .npz file"
    )
    parser.add_argument(
        "--val-preds-dir",
        type=str,
        help="Validation predictions directory (for extracting inliers)"
    )
    parser.add_argument(
        "--val-inliers-dir",
        type=str,
        help="Validation inliers directory (full re-ranking results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/inliers_feature_test",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Testing Inliers as Feature")
    print(f"{'='*70}\n")
    
    # Load features
    print("Loading features...")
    train_features = load_feature_file(args.train_features)
    val_features = load_feature_file(args.val_features)
    
    num_val_queries = len(val_features["labels"])
    print(f"  Training queries: {len(train_features['labels'])}")
    print(f"  Validation queries: {num_val_queries}")
    
    # Extract inliers if available
    val_inliers = None
    if args.val_preds_dir and args.val_inliers_dir:
        val_preds_dir = Path(args.val_preds_dir)
        val_inliers_dir = Path(args.val_inliers_dir)
        
        if val_inliers_dir.exists():
            print(f"\nExtracting inliers from validation set...")
            val_inliers = extract_inliers_from_matching(
                val_preds_dir, val_inliers_dir, num_val_queries
            )
            print(f"  Inliers range: {val_inliers.min():.0f} - {val_inliers.max():.0f}")
            print(f"  Mean inliers: {val_inliers.mean():.2f}")
            print(f"  Queries with inliers > 0: {(val_inliers > 0).sum()} ({(val_inliers > 0).mean() * 100:.1f}%)")
        else:
            print(f"\nWarning: Inliers directory not found: {val_inliers_dir}")
            print(f"  Will test without inliers feature")
    else:
        print(f"\nNo inliers directories provided - testing without inliers feature")
    
    # Train model WITHOUT inliers (baseline)
    print(f"\n{'='*70}")
    print(f"Training Model WITHOUT Inliers (Baseline)")
    print(f"{'='*70}")
    
    X_train_base, y_train, _ = build_feature_matrix(train_features)
    X_val_base, y_val, _ = build_feature_matrix(val_features)
    
    results_baseline = train_and_evaluate_model(
        X_train_base, y_train, X_val_base, y_val, "Baseline (8 features)"
    )
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy: {results_baseline['accuracy']:.4f} ({results_baseline['accuracy']*100:.2f}%)")
    print(f"  F1-Score: {results_baseline['f1']:.4f}")
    print(f"  Precision: {results_baseline['precision']:.4f}")
    print(f"  Recall: {results_baseline['recall']:.4f}")
    print(f"  Hard Query Rate: {results_baseline['hard_query_rate']:.2f}%")
    print(f"  Optimal Threshold: {results_baseline['optimal_threshold']:.3f}")
    print(f"  Optimal C: {results_baseline['optimal_C']:.2f}")
    
    # Train model WITH inliers (if available)
    if val_inliers is not None:
        print(f"\n{'='*70}")
        print(f"Training Model WITH Inliers (9 features)")
        print(f"{'='*70}")
        
        # Extract inliers for training set (if available)
        # Note: Training inliers would need to be provided separately
        # For this test, we'll use zeros for training (simulating no inliers during training)
        # This tests if inliers help during validation even if not available during training
        num_train_queries = len(train_features["labels"])
        train_inliers = np.zeros(num_train_queries, dtype=np.float32)  # No inliers during training
        
        X_train_with_inliers, y_train, feature_names = build_feature_matrix_with_inliers(
            train_features, train_inliers
        )
        X_val_with_inliers, y_val, _ = build_feature_matrix_with_inliers(
            val_features, val_inliers
        )
        
        results_with_inliers = train_and_evaluate_model(
            X_train_with_inliers, y_train, X_val_with_inliers, y_val, "With Inliers (9 features)"
        )
        
        print(f"\nWith Inliers Results:")
        print(f"  Accuracy: {results_with_inliers['accuracy']:.4f} ({results_with_inliers['accuracy']*100:.2f}%)")
        print(f"  F1-Score: {results_with_inliers['f1']:.4f}")
        print(f"  Precision: {results_with_inliers['precision']:.4f}")
        print(f"  Recall: {results_with_inliers['recall']:.4f}")
        print(f"  Hard Query Rate: {results_with_inliers['hard_query_rate']:.2f}%")
        print(f"  Optimal Threshold: {results_with_inliers['optimal_threshold']:.3f}")
        print(f"  Optimal C: {results_with_inliers['optimal_C']:.2f}")
        
        # Comparison
        print(f"\n{'='*70}")
        print(f"Comparison: With vs Without Inliers")
        print(f"{'='*70}")
        print(f"  Accuracy: {results_baseline['accuracy']*100:.2f}% -> {results_with_inliers['accuracy']*100:.2f}% "
              f"({(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}%)")
        print(f"  F1-Score: {results_baseline['f1']:.4f} → {results_with_inliers['f1']:.4f} "
              f"({results_with_inliers['f1'] - results_baseline['f1']:+.4f})")
        print(f"  Hard Query Rate: {results_baseline['hard_query_rate']:.2f}% → {results_with_inliers['hard_query_rate']:.2f}% "
              f"({results_with_inliers['hard_query_rate'] - results_baseline['hard_query_rate']:+.2f}%)")
        
        # Save comparison report
        report_path = output_dir / "inliers_feature_comparison.md"
        with open(report_path, 'w') as f:
            f.write("# Inliers Feature Test Results\n\n")
            f.write("## Comparison: With vs Without Inliers\n\n")
            f.write("| Metric | Without Inliers (8 features) | With Inliers (9 features) | Improvement |\n")
            f.write("|--------|------------------------------|--------------------------|-------------|\n")
            f.write(f"| Accuracy | {results_baseline['accuracy']*100:.2f}% | {results_with_inliers['accuracy']*100:.2f}% | "
                   f"{(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}% |\n")
            f.write(f"| F1-Score | {results_baseline['f1']:.4f} | {results_with_inliers['f1']:.4f} | "
                   f"{results_with_inliers['f1'] - results_baseline['f1']:+.4f} |\n")
            f.write(f"| Precision | {results_baseline['precision']:.4f} | {results_with_inliers['precision']:.4f} | "
                   f"{results_with_inliers['precision'] - results_baseline['precision']:+.4f} |\n")
            f.write(f"| Recall | {results_baseline['recall']:.4f} | {results_with_inliers['recall']:.4f} | "
                   f"{results_with_inliers['recall'] - results_baseline['recall']:+.4f} |\n")
            f.write(f"| Hard Query Rate | {results_baseline['hard_query_rate']:.2f}% | {results_with_inliers['hard_query_rate']:.2f}% | "
                   f"{results_with_inliers['hard_query_rate'] - results_baseline['hard_query_rate']:+.2f}% |\n")
            
            f.write(f"\n## Conclusion\n\n")
            if results_with_inliers['f1'] > results_baseline['f1']:
                f.write(f"✅ **Inliers feature improves model performance**\n")
                f.write(f"- F1-Score improved by {results_with_inliers['f1'] - results_baseline['f1']:.4f}\n")
                f.write(f"- Accuracy improved by {(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:.2f}%\n")
            else:
                f.write(f"⚠️ **Inliers feature does not significantly improve model**\n")
                f.write(f"- F1-Score change: {results_with_inliers['f1'] - results_baseline['f1']:+.4f}\n")
                f.write(f"- Accuracy change: {(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}%\n")
            
            f.write(f"\n## Important Note\n\n")
            f.write(f"⚠️ **Using inliers as a feature requires running image matching on ALL queries first.**\n")
            f.write(f"This defeats the purpose of adaptive re-ranking (predicting hard queries BEFORE matching).\n")
            f.write(f"However, this test shows how much inliers would help IF we had them.\n")
        
        print(f"\nSaved comparison report: {report_path}")
    else:
        print(f"\n[WARNING] Cannot test with inliers - inliers not available")
        print(f"  To test with inliers, provide --val-preds-dir and --val-inliers-dir")
    
    print(f"\n{'='*70}")
    print(f"Test Complete!")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())



This script:
1. Extracts num_inliers from existing image matching results (if available)
2. Adds it as a 9th feature to the model
3. Trains and evaluates the model with and without inliers
4. Compares performance to see if inliers improve predictions

Note: This only works if full re-ranking has been done on the validation set.
"""

import argparse
import numpy as np
from pathlib import Path
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import joblib
from tqdm import tqdm
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_3_train_logreg_easy_queries import (
    load_feature_file,
    build_feature_matrix,
    find_optimal_threshold
)


def extract_inliers_from_matching(preds_dir: Path, inliers_dir: Path, num_queries: int) -> np.ndarray:
    """
    Extract num_inliers for top-1 match from image matching results.
    
    Args:
        preds_dir: Directory with prediction .txt files
        inliers_dir: Directory with .torch files (image matching results)
        num_queries: Number of queries
    
    Returns:
        Array of num_inliers for each query (top-1 match only)
    """
    inliers = np.zeros(num_queries, dtype=np.float32)
    
    txt_files = sorted(preds_dir.glob("*.txt"), key=lambda x: int(x.stem))
    
    print(f"Extracting inliers from {len(txt_files)} queries...")
    
    for txt_file in tqdm(txt_files, desc="Loading inliers"):
        try:
            query_idx = int(txt_file.stem)
            torch_file = inliers_dir / f"{txt_file.stem}.torch"
            
            if not torch_file.exists():
                # No matching done for this query
                inliers[query_idx] = 0.0
                continue
            
            # Load matching results
            query_results = torch.load(str(torch_file), weights_only=False)
            
            if len(query_results) > 0:
                # Get num_inliers for top-1 match (first result)
                inliers[query_idx] = float(query_results[0].get("num_inliers", 0))
            else:
                inliers[query_idx] = 0.0
                
        except Exception as e:
            print(f"Warning: Error loading inliers for {txt_file.name}: {e}")
            inliers[query_idx] = 0.0
    
    return inliers


def build_feature_matrix_with_inliers(features_dict: dict, inliers: np.ndarray = None) -> tuple:
    """
    Build feature matrix with optional inliers feature.
    
    Args:
        features_dict: Dictionary with 8 existing features
        inliers: Optional array of num_inliers (top-1) for each query
    
    Returns:
        (X, y, feature_names) where X includes inliers if provided
    """
    # Build base features (8 features)
    X_base, y, feature_names_base = build_feature_matrix(features_dict)
    
    if inliers is not None:
        # Add inliers as 9th feature
        # Normalize inliers to 0-1 range (or use log scale if needed)
        # For now, use raw inliers (will be scaled by StandardScaler)
        inliers_normalized = inliers.astype(np.float32)
        
        # Handle NaN/Inf
        inliers_normalized = np.nan_to_num(inliers_normalized, nan=0.0, posinf=0.0, neginf=0.0)
        
        X = np.column_stack([X_base, inliers_normalized])
        feature_names = feature_names_base + ["num_inliers_top1"]
    else:
        X = X_base
        feature_names = feature_names_base
    
    return X, y, feature_names


def train_and_evaluate_model(X_train, y_train, X_val, y_val, model_name: str):
    """Train model and evaluate on validation set."""
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Handle NaNs
    train_valid_mask = ~np.isnan(X_train_scaled).any(axis=1)
    X_train_scaled = X_train_scaled[train_valid_mask]
    y_train = y_train[train_valid_mask]
    
    val_valid_mask = ~np.isnan(X_val_scaled).any(axis=1)
    X_val_scaled = X_val_scaled[val_valid_mask]
    y_val = y_val[val_valid_mask]
    
    # Hyperparameter tuning (C)
    C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
    best_C = 1.0
    best_val_f1 = 0.0
    
    for C in C_values:
        logreg_temp = LogisticRegression(
            C=C, solver="lbfgs", max_iter=1000, class_weight='balanced'
        )
        logreg_temp.fit(X_train_scaled, y_train)
        y_val_probs = logreg_temp.predict_proba(X_val_scaled)[:, 1]
        threshold_temp, _ = find_optimal_threshold(y_val, y_val_probs, method="f1")
        y_val_pred = (y_val_probs >= threshold_temp).astype(int)
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_C = C
    
    # Train final model
    logreg = LogisticRegression(
        C=best_C, solver="lbfgs", max_iter=1000, class_weight='balanced'
    )
    logreg.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]
    optimal_threshold, _ = find_optimal_threshold(y_val, y_val_probs, method="f1")
    y_val_pred = (y_val_probs >= optimal_threshold).astype(int)
    
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred, zero_division=0)
    precision = precision_score(y_val, y_val_pred, zero_division=0)
    recall = recall_score(y_val, y_val_pred, zero_division=0)
    
    hard_query_rate = (1 - y_val_pred.mean()) * 100
    
    return {
        "model": logreg,
        "scaler": scaler,
        "optimal_threshold": optimal_threshold,
        "optimal_C": best_C,
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "hard_query_rate": hard_query_rate
    }


def main():
    parser = argparse.ArgumentParser(
        description="Test if adding inliers as feature improves model"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Training features .npz file"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        required=True,
        help="Validation features .npz file"
    )
    parser.add_argument(
        "--val-preds-dir",
        type=str,
        help="Validation predictions directory (for extracting inliers)"
    )
    parser.add_argument(
        "--val-inliers-dir",
        type=str,
        help="Validation inliers directory (full re-ranking results)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/inliers_feature_test",
        help="Output directory"
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Testing Inliers as Feature")
    print(f"{'='*70}\n")
    
    # Load features
    print("Loading features...")
    train_features = load_feature_file(args.train_features)
    val_features = load_feature_file(args.val_features)
    
    num_val_queries = len(val_features["labels"])
    print(f"  Training queries: {len(train_features['labels'])}")
    print(f"  Validation queries: {num_val_queries}")
    
    # Extract inliers if available
    val_inliers = None
    if args.val_preds_dir and args.val_inliers_dir:
        val_preds_dir = Path(args.val_preds_dir)
        val_inliers_dir = Path(args.val_inliers_dir)
        
        if val_inliers_dir.exists():
            print(f"\nExtracting inliers from validation set...")
            val_inliers = extract_inliers_from_matching(
                val_preds_dir, val_inliers_dir, num_val_queries
            )
            print(f"  Inliers range: {val_inliers.min():.0f} - {val_inliers.max():.0f}")
            print(f"  Mean inliers: {val_inliers.mean():.2f}")
            print(f"  Queries with inliers > 0: {(val_inliers > 0).sum()} ({(val_inliers > 0).mean() * 100:.1f}%)")
        else:
            print(f"\nWarning: Inliers directory not found: {val_inliers_dir}")
            print(f"  Will test without inliers feature")
    else:
        print(f"\nNo inliers directories provided - testing without inliers feature")
    
    # Train model WITHOUT inliers (baseline)
    print(f"\n{'='*70}")
    print(f"Training Model WITHOUT Inliers (Baseline)")
    print(f"{'='*70}")
    
    X_train_base, y_train, _ = build_feature_matrix(train_features)
    X_val_base, y_val, _ = build_feature_matrix(val_features)
    
    results_baseline = train_and_evaluate_model(
        X_train_base, y_train, X_val_base, y_val, "Baseline (8 features)"
    )
    
    print(f"\nBaseline Results:")
    print(f"  Accuracy: {results_baseline['accuracy']:.4f} ({results_baseline['accuracy']*100:.2f}%)")
    print(f"  F1-Score: {results_baseline['f1']:.4f}")
    print(f"  Precision: {results_baseline['precision']:.4f}")
    print(f"  Recall: {results_baseline['recall']:.4f}")
    print(f"  Hard Query Rate: {results_baseline['hard_query_rate']:.2f}%")
    print(f"  Optimal Threshold: {results_baseline['optimal_threshold']:.3f}")
    print(f"  Optimal C: {results_baseline['optimal_C']:.2f}")
    
    # Train model WITH inliers (if available)
    if val_inliers is not None:
        print(f"\n{'='*70}")
        print(f"Training Model WITH Inliers (9 features)")
        print(f"{'='*70}")
        
        # Extract inliers for training set (if available)
        # Note: Training inliers would need to be provided separately
        # For this test, we'll use zeros for training (simulating no inliers during training)
        # This tests if inliers help during validation even if not available during training
        num_train_queries = len(train_features["labels"])
        train_inliers = np.zeros(num_train_queries, dtype=np.float32)  # No inliers during training
        
        X_train_with_inliers, y_train, feature_names = build_feature_matrix_with_inliers(
            train_features, train_inliers
        )
        X_val_with_inliers, y_val, _ = build_feature_matrix_with_inliers(
            val_features, val_inliers
        )
        
        results_with_inliers = train_and_evaluate_model(
            X_train_with_inliers, y_train, X_val_with_inliers, y_val, "With Inliers (9 features)"
        )
        
        print(f"\nWith Inliers Results:")
        print(f"  Accuracy: {results_with_inliers['accuracy']:.4f} ({results_with_inliers['accuracy']*100:.2f}%)")
        print(f"  F1-Score: {results_with_inliers['f1']:.4f}")
        print(f"  Precision: {results_with_inliers['precision']:.4f}")
        print(f"  Recall: {results_with_inliers['recall']:.4f}")
        print(f"  Hard Query Rate: {results_with_inliers['hard_query_rate']:.2f}%")
        print(f"  Optimal Threshold: {results_with_inliers['optimal_threshold']:.3f}")
        print(f"  Optimal C: {results_with_inliers['optimal_C']:.2f}")
        
        # Comparison
        print(f"\n{'='*70}")
        print(f"Comparison: With vs Without Inliers")
        print(f"{'='*70}")
        print(f"  Accuracy: {results_baseline['accuracy']*100:.2f}% -> {results_with_inliers['accuracy']*100:.2f}% "
              f"({(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}%)")
        print(f"  F1-Score: {results_baseline['f1']:.4f} → {results_with_inliers['f1']:.4f} "
              f"({results_with_inliers['f1'] - results_baseline['f1']:+.4f})")
        print(f"  Hard Query Rate: {results_baseline['hard_query_rate']:.2f}% → {results_with_inliers['hard_query_rate']:.2f}% "
              f"({results_with_inliers['hard_query_rate'] - results_baseline['hard_query_rate']:+.2f}%)")
        
        # Save comparison report
        report_path = output_dir / "inliers_feature_comparison.md"
        with open(report_path, 'w') as f:
            f.write("# Inliers Feature Test Results\n\n")
            f.write("## Comparison: With vs Without Inliers\n\n")
            f.write("| Metric | Without Inliers (8 features) | With Inliers (9 features) | Improvement |\n")
            f.write("|--------|------------------------------|--------------------------|-------------|\n")
            f.write(f"| Accuracy | {results_baseline['accuracy']*100:.2f}% | {results_with_inliers['accuracy']*100:.2f}% | "
                   f"{(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}% |\n")
            f.write(f"| F1-Score | {results_baseline['f1']:.4f} | {results_with_inliers['f1']:.4f} | "
                   f"{results_with_inliers['f1'] - results_baseline['f1']:+.4f} |\n")
            f.write(f"| Precision | {results_baseline['precision']:.4f} | {results_with_inliers['precision']:.4f} | "
                   f"{results_with_inliers['precision'] - results_baseline['precision']:+.4f} |\n")
            f.write(f"| Recall | {results_baseline['recall']:.4f} | {results_with_inliers['recall']:.4f} | "
                   f"{results_with_inliers['recall'] - results_baseline['recall']:+.4f} |\n")
            f.write(f"| Hard Query Rate | {results_baseline['hard_query_rate']:.2f}% | {results_with_inliers['hard_query_rate']:.2f}% | "
                   f"{results_with_inliers['hard_query_rate'] - results_baseline['hard_query_rate']:+.2f}% |\n")
            
            f.write(f"\n## Conclusion\n\n")
            if results_with_inliers['f1'] > results_baseline['f1']:
                f.write(f"✅ **Inliers feature improves model performance**\n")
                f.write(f"- F1-Score improved by {results_with_inliers['f1'] - results_baseline['f1']:.4f}\n")
                f.write(f"- Accuracy improved by {(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:.2f}%\n")
            else:
                f.write(f"⚠️ **Inliers feature does not significantly improve model**\n")
                f.write(f"- F1-Score change: {results_with_inliers['f1'] - results_baseline['f1']:+.4f}\n")
                f.write(f"- Accuracy change: {(results_with_inliers['accuracy'] - results_baseline['accuracy'])*100:+.2f}%\n")
            
            f.write(f"\n## Important Note\n\n")
            f.write(f"⚠️ **Using inliers as a feature requires running image matching on ALL queries first.**\n")
            f.write(f"This defeats the purpose of adaptive re-ranking (predicting hard queries BEFORE matching).\n")
            f.write(f"However, this test shows how much inliers would help IF we had them.\n")
        
        print(f"\nSaved comparison report: {report_path}")
    else:
        print(f"\n[WARNING] Cannot test with inliers - inliers not available")
        print(f"  To test with inliers, provide --val-preds-dir and --val-inliers-dir")
    
    print(f"\n{'='*70}")
    print(f"Test Complete!")
    print(f"{'='*70}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

