"""
Tune Logistic Regression hyperparameters (C and threshold) for adaptive re-ranking.

This script:
1. Sweeps C values (regularization strength)
2. For each C, trains a model and evaluates on validation set
3. Selects the best C based on validation ROC-AUC
4. Sweeps threshold values for the best C model
5. Reports optimal C and threshold
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib


def train_and_evaluate(train_csv, val_csv, C, feature_col="inliers_top1", label_col="is_top1_correct"):
    """Train LR model with given C and return validation metrics."""
    # Load data
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    X_train = train_df[[feature_col]].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_val = val_df[[feature_col]].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.int32)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Train
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        C=C,
    )
    logreg.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_val_proba = logreg.predict_proba(X_val_scaled)[:, 1]
    y_val_pred = logreg.predict(X_val_scaled)
    
    val_acc = accuracy_score(y_val, y_val_pred)
    try:
        val_auc = roc_auc_score(y_val, y_val_proba)
    except ValueError:
        val_auc = np.nan
    
    return {
        "model": logreg,
        "scaler": scaler,
        "val_acc": val_acc,
        "val_auc": val_auc,
        "val_proba": y_val_proba,
        "val_labels": y_val,
    }


def evaluate_threshold(proba, labels, threshold):
    """Evaluate a threshold on validation set."""
    predictions = (proba >= threshold).astype(int)
    acc = accuracy_score(labels, predictions)
    
    # Fraction of queries that are "hard" (will be re-ranked)
    pct_hard = (proba < threshold).mean() * 100
    
    return {
        "threshold": threshold,
        "accuracy": acc,
        "pct_hard": pct_hard,
        "pct_easy": 100 - pct_hard,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Tune LR hyperparameters (C and threshold) for adaptive re-ranking."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV file.",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        required=True,
        help="Path to validation CSV file.",
    )
    parser.add_argument(
        "--C-values",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
        help="C values to try (regularization strength).",
    )
    parser.add_argument(
        "--threshold-values",
        nargs="+",
        type=float,
        default=[0.3, 0.4, 0.5, 0.6, 0.7],
        help="Threshold values to try for easy/hard classification.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="tuning_results",
        help="Directory to save tuning results and best model.",
    )
    parser.add_argument(
        "--feature-col",
        type=str,
        default="inliers_top1",
        help="Name of the feature column in CSV.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="is_top1_correct",
        help="Name of the label column in CSV.",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("STEP 1: Tuning C (regularization strength)")
    print("=" * 80)
    
    # Step 1: Tune C
    best_C = None
    best_C_auc = -np.inf
    best_C_results = None
    C_results = []

    for C in args.C_values:
        print(f"\nTrying C = {C:.2f}...")
        results = train_and_evaluate(
            args.train_csv,
            args.val_csv,
            C,
            args.feature_col,
            args.label_col,
        )
        
        C_results.append({
            "C": C,
            "val_acc": results["val_acc"],
            "val_auc": results["val_auc"],
        })
        
        print(f"  Validation Accuracy: {results['val_acc']:.4f}")
        print(f"  Validation ROC-AUC:  {results['val_auc']:.4f}")
        
        # Select best C: prioritize ROC-AUC, but if tied, use accuracy
        if not np.isnan(results["val_auc"]):
            if results["val_auc"] > best_C_auc:
                best_C_auc = results["val_auc"]
                best_C = C
                best_C_results = results
            elif results["val_auc"] == best_C_auc and results["val_acc"] > best_C_results["val_acc"]:
                # Tie-breaker: if same ROC-AUC, pick higher accuracy
                best_C = C
                best_C_results = results

    print("\n" + "=" * 80)
    print("C Tuning Results:")
    print("=" * 80)
    print(f"{'C':<10} {'Val Accuracy':<15} {'Val ROC-AUC':<15}")
    print("-" * 80)
    for r in C_results:
        print(f"{r['C']:<10.2f} {r['val_acc']:<15.4f} {r['val_auc']:<15.4f}")
    print(f"\nBest C: {best_C:.2f} (ROC-AUC: {best_C_auc:.4f})")

    if best_C_results is None:
        print("ERROR: Could not find valid C value. Check your validation set.")
        return

    # Step 2: Tune threshold for best C
    print("\n" + "=" * 80)
    print(f"STEP 2: Tuning threshold for C = {best_C:.2f}")
    print("=" * 80)
    
    threshold_results = []
    for threshold in args.threshold_values:
        print(f"\nTrying threshold = {threshold:.2f}...")
        results = evaluate_threshold(
            best_C_results["val_proba"],
            best_C_results["val_labels"],
            threshold,
        )
        threshold_results.append(results)
        print(f"  Accuracy: {results['accuracy']:.4f}")
        print(f"  % Hard queries (re-ranked): {results['pct_hard']:.2f}%")
        print(f"  % Easy queries (skipped):   {results['pct_easy']:.2f}%")

    print("\n" + "=" * 80)
    print("Threshold Tuning Results:")
    print("=" * 80)
    print(f"{'Threshold':<12} {'Accuracy':<12} {'% Hard':<12} {'% Easy':<12}")
    print("-" * 80)
    for r in threshold_results:
        print(f"{r['threshold']:<12.2f} {r['accuracy']:<12.4f} {r['pct_hard']:<12.2f} {r['pct_easy']:<12.2f}")

    # Select best threshold (balance accuracy and % re-ranked)
    # You can change this logic based on your preference
    best_threshold = max(threshold_results, key=lambda x: x["accuracy"])
    print(f"\nBest threshold: {best_threshold['threshold']:.2f}")
    print(f"  Accuracy: {best_threshold['accuracy']:.4f}")
    print(f"  % Queries re-ranked: {best_threshold['pct_hard']:.2f}%")

    # Save best model
    best_model_path = output_dir / f"logreg_best_C{best_C:.2f}.pkl"
    model_bundle = {
        "scaler": best_C_results["scaler"],
        "model": best_C_results["model"],
        "feature_col": args.feature_col,
        "best_C": best_C,
        "best_threshold": best_threshold["threshold"],
        "val_auc": best_C_auc,
    }
    joblib.dump(model_bundle, best_model_path)
    print(f"\nSaved best model to {best_model_path}")

    # Save tuning summary
    summary_path = output_dir / "tuning_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LR Hyperparameter Tuning Summary\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Training CSV: {args.train_csv}\n")
        f.write(f"Validation CSV: {args.val_csv}\n\n")
        f.write("Best Hyperparameters:\n")
        f.write(f"  C (regularization): {best_C:.2f}\n")
        f.write(f"  Threshold: {best_threshold['threshold']:.2f}\n")
        f.write(f"  Validation ROC-AUC: {best_C_auc:.4f}\n")
        f.write(f"  Validation Accuracy: {best_threshold['accuracy']:.4f}\n")
        f.write(f"  % Queries re-ranked: {best_threshold['pct_hard']:.2f}%\n")
    
    print(f"Saved tuning summary to {summary_path}")
    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print("=" * 80)
    print(f"Use these parameters:")
    print(f"  --C {best_C:.2f}")
    print(f"  --threshold {best_threshold['threshold']:.2f}")


if __name__ == "__main__":
    main()

