"""
Tune Logistic Regression hyperparameters (C and threshold) for adaptive re-ranking.

This script:
1. Accepts a folder containing 3 training CSVs (sun, night, combined) and 1 validation CSV
2. Trains 3 LR models (one per training CSV)
3. For each model, evaluates R@1 and classification accuracy on validation set for different thresholds
4. Generates plots showing threshold vs R@1 for all 3 models
5. Selects best threshold based on classification accuracy (how well LR predicts if top-1 is correct)
   Note: We optimize classification accuracy instead of R@1 because validation sets (like SF-XS val)
   are often too easy, leading to threshold=0.0 (no re-ranking). Classification accuracy better
   reflects the adaptive strategy's ability to distinguish easy vs hard queries.
"""

import argparse
import pandas as pd
import numpy as np
import torch
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib
import sys

# Add parent directory to path to import util
sys.path.insert(0, str(Path(__file__).parent.parent))
from util import get_list_distances_from_preds


def train_lr_model(train_csv, C, feature_col="inliers_top1", label_col="is_top1_correct"):
    """Train LR model with given C."""
    train_df = pd.read_csv(train_csv)
    
    X_train = train_df[[feature_col]].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    
    # Normalize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Train
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        C=C,
    )
    logreg.fit(X_train_scaled, y_train)
    
    return {
        "model": logreg,
        "scaler": scaler,
    }


def compute_r1_for_threshold(
    val_csv,
    val_preds_dir,
    val_top20_inliers_dir,
    model,
    scaler,
    threshold,
    num_preds=20,
    positive_dist_threshold=25,
    debug=False,
):
    """
    Compute R@1 and classification accuracy on validation set for a given threshold using adaptive reranking.
    
    Args:
        val_csv: Path to validation CSV
        val_preds_dir: Directory with validation .txt prediction files
        val_top20_inliers_dir: Directory with validation top-20 inlier .torch files
        model: Trained LR model
        scaler: StandardScaler used for training
        threshold: Decision threshold for easy/hard classification
        num_preds: Number of predictions to consider (K)
        positive_dist_threshold: Distance threshold in meters for positive match
    
    Returns:
        recall_at_1: R@1 percentage
        classification_accuracy: Classification accuracy (how well LR predicts if top-1 is correct)
        num_easy: Number of easy queries
        num_hard: Number of hard queries
    """
    # Read CSV with query_id as string to preserve zero-padding
    val_df = pd.read_csv(val_csv, dtype={"query_id": str})
    val_preds_dir = Path(val_preds_dir)
    val_top20_inliers_dir = Path(val_top20_inliers_dir)
    
    if not val_preds_dir.exists():
        raise FileNotFoundError(f"Validation predictions directory does not exist: {val_preds_dir}")
    if not val_top20_inliers_dir.exists():
        raise FileNotFoundError(f"Validation top-20 inliers directory does not exist: {val_top20_inliers_dir}")
    
    # Get all .txt files
    txt_files = sorted(val_preds_dir.glob("*.txt"), key=lambda p: int(p.stem) if p.stem.isdigit() else 0)
    
    # Print debug info only once
    if debug:
        print(f"  Found {len(txt_files)} .txt files in {val_preds_dir}")
        print(f"  Found {len(val_df)} rows in validation CSV")
        
        # Check query_id format in CSV
        csv_query_ids = set(val_df["query_id"].astype(str))
        txt_query_ids = set([f.stem for f in txt_files[:10]])  # Check first 10
        print(f"  Sample CSV query_ids: {sorted(list(csv_query_ids))[:5]}")
        print(f"  Sample TXT query_ids: {sorted(list(txt_query_ids))[:5]}")
    
    correct_at_1 = 0
    correct_classifications = 0  # Track classification accuracy
    processed = 0
    num_easy = 0
    num_hard = 0
    skipped_no_csv_match = 0
    skipped_no_distances = 0
    all_probs = []  # Collect probabilities for debugging
    example_queries = []  # Store examples for debugging
    
    # Convert CSV query_ids to int for matching (handles both "000" and "0" formats)
    val_df["query_id_int"] = pd.to_numeric(val_df["query_id"], errors="coerce").fillna(-1).astype(int)
    
    for txt_file in txt_files:
        query_id_str = txt_file.stem  # e.g., "000", "001"
        
        # Convert TXT query_id to int for matching
        try:
            query_id_int = int(query_id_str)  # "000" -> 0, "001" -> 1
        except ValueError:
            skipped_no_csv_match += 1
            continue
        
        # Match by integer
        query_row = val_df[val_df["query_id_int"] == query_id_int]
        if len(query_row) == 0:
            skipped_no_csv_match += 1
            continue
        
        inliers_top1 = float(query_row.iloc[0]["inliers_top1"])
        is_top1_correct_gt = int(query_row.iloc[0]["is_top1_correct"])  # Ground truth
        
        # Predict probability using LR model
        inliers_array = np.array([[inliers_top1]], dtype=np.float32)
        inliers_scaled = scaler.transform(inliers_array)
        prob_correct = float(model.predict_proba(inliers_scaled)[0, 1])
        all_probs.append(prob_correct)
        
        # Classify as easy or hard
        is_easy = prob_correct >= threshold
        
        # Track classification accuracy: LR predicts "easy" (prob >= threshold) means top-1 is correct
        # So classification is correct if:
        #   - LR says "easy" (prob >= threshold) AND top-1 is actually correct (is_top1_correct_gt == 1)
        #   - OR LR says "hard" (prob < threshold) AND top-1 is actually wrong (is_top1_correct_gt == 0)
        predicted_easy = is_easy
        actual_correct = (is_top1_correct_gt == 1)
        classification_correct = (predicted_easy == actual_correct)
        if classification_correct:
            correct_classifications += 1
        
        # Collect example for debugging (first 5 queries)
        if debug and len(example_queries) < 5:
            example_queries.append({
                "query_id": query_id_str,
                "inliers_top1": inliers_top1,
                "prob_correct": prob_correct,
                "is_easy": is_easy,
                "is_top1_correct": is_top1_correct_gt,
                "threshold": threshold,
            })
        
        # Get retrieval distances
        try:
            geo_dists = torch.tensor(get_list_distances_from_preds(str(txt_file)))[:num_preds]
            if len(geo_dists) == 0:
                skipped_no_distances += 1
                continue
        except Exception as e:
            skipped_no_distances += 1
            continue
        
        if is_easy:
            # EASY: use retrieval-only ranking
            num_easy += 1
            ranking_dists = geo_dists
        else:
            # HARD: re-rank using top-20 inliers
            num_hard += 1
            
            # Load top-20 inliers (use original string format for filename)
            torch_file = val_top20_inliers_dir / f"{query_id_str}.torch"
            if not torch_file.exists():
                # If no top-20 inliers, fall back to retrieval-only
                ranking_dists = geo_dists
            else:
                try:
                    results = torch.load(str(torch_file), weights_only=False)
                    if len(results) == 0:
                        ranking_dists = geo_dists
                    else:
                        # Extract inliers for top-K
                        actual_num_preds = min(len(results), num_preds, len(geo_dists))
                        query_db_inliers = torch.zeros(actual_num_preds, dtype=torch.float32)
                        for i in range(actual_num_preds):
                            query_db_inliers[i] = float(results[i]["num_inliers"])
                        
                        # Sort by inliers (descending) and reorder distances
                        _, indices = torch.sort(query_db_inliers, descending=True)
                        ranking_dists = geo_dists[indices]
                except Exception:
                    # If error loading, fall back to retrieval-only
                    ranking_dists = geo_dists
        
        # Check if top-1 is correct (distance <= threshold)
        if len(ranking_dists) > 0 and ranking_dists[0] <= positive_dist_threshold:
            correct_at_1 += 1
        
        processed += 1
    
    recall_at_1 = (correct_at_1 / processed * 100.0) if processed > 0 else 0.0
    classification_accuracy = (correct_classifications / processed * 100.0) if processed > 0 else 0.0
    
    if debug and len(all_probs) > 0:
        all_probs = np.array(all_probs)
        print(f"  Probability stats (LR predictions):")
        print(f"    Min: {all_probs.min():.4f}")
        print(f"    Max: {all_probs.max():.4f}")
        print(f"    Mean: {all_probs.mean():.4f}")
        print(f"    Median: {np.median(all_probs):.4f}")
        print(f"    % with prob >= 0.5: {(all_probs >= 0.5).mean() * 100:.1f}%")
        print(f"    % with prob >= 0.7: {(all_probs >= 0.7).mean() * 100:.1f}%")
        print(f"    % with prob >= 0.9: {(all_probs >= 0.9).mean() * 100:.1f}%")
        print(f"  Threshold = {threshold:.2f}")
        print(f"  Expected easy (prob >= {threshold:.2f}): {(all_probs >= threshold).sum()} ({(all_probs >= threshold).mean() * 100:.1f}%)")
        print(f"  Expected hard (prob < {threshold:.2f}): {(all_probs < threshold).sum()} ({(all_probs < threshold).mean() * 100:.1f}%)")
        print(f"  Actual easy: {num_easy} ({num_easy/processed*100:.1f}%)")
        print(f"  Actual hard: {num_hard} ({num_hard/processed*100:.1f}%)")
        print(f"  Processed: {processed} queries")
        print(f"  Skipped (no CSV match): {skipped_no_csv_match}")
        print(f"  Skipped (no distances): {skipped_no_distances}")
        if example_queries:
            print(f"  Example queries (first 5):")
            for ex in example_queries:
                print(f"    Query {ex['query_id']}: inliers={ex['inliers_top1']:.1f}, prob={ex['prob_correct']:.4f}, "
                      f"is_easy={ex['is_easy']}, is_top1_correct={ex['is_top1_correct']}, threshold={ex['threshold']:.2f}")
    
    if processed == 0:
        print(f"  WARNING: No queries were processed!")
        print(f"    Skipped (no CSV match): {skipped_no_csv_match}")
        print(f"    Skipped (no distances): {skipped_no_distances}")
        print(f"    Total txt files: {len(txt_files)}")
    
    return recall_at_1, classification_accuracy, num_easy, num_hard


def find_best_C(train_csv, val_csv, C_values, feature_col="inliers_top1", label_col="is_top1_correct"):
    """Find best C value based on ROC-AUC."""
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    
    X_train = train_df[[feature_col]].values.astype(np.float32)
    y_train = train_df[label_col].values.astype(np.int32)
    X_val = val_df[[feature_col]].values.astype(np.float32)
    y_val = val_df[label_col].values.astype(np.int32)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    best_C = None
    best_auc = -np.inf
    best_model = None
    best_scaler = None
    
    for C in C_values:
        logreg = LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=42,
            C=C,
        )
        logreg.fit(X_train_scaled, y_train)

        y_val_proba = logreg.predict_proba(X_val_scaled)[:, 1]
        try:
            val_auc = roc_auc_score(y_val, y_val_proba)
            if val_auc > best_auc:
                best_auc = val_auc
                best_C = C
                best_model = logreg
                best_scaler = scaler
        except ValueError:
            continue
    
    if best_C is None:
        # Fallback: use first C value
        best_C = C_values[0]
        best_model = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42, C=best_C)
        best_model.fit(X_train_scaled, y_train)
        best_scaler = scaler
    
    return best_C, best_model, best_scaler


def main():
    parser = argparse.ArgumentParser(
        description="Tune LR hyperparameters (C and threshold) for adaptive re-ranking with R@1 optimization."
    )
    parser.add_argument(
        "--csv-folder",
        type=str,
        required=True,
        help="Folder containing training CSVs (sun, night, combined) and validation CSV.",
    )
    parser.add_argument(
        "--val-preds-dir",
        type=str,
        required=True,
        help="Directory with validation .txt prediction files.",
    )
    parser.add_argument(
        "--val-top20-inliers-dir",
        type=str,
        required=True,
        help="Directory with validation top-20 inlier .torch files.",
    )
    parser.add_argument(
        "--C-values",
        nargs="+",
        type=float,
        default=[0.01, 0.1, 0.3, 1.0, 3.0, 10.0],
        help="C values to try (regularization strength).",
    )
    parser.add_argument(
        "--force-C",
        type=float,
        default=None,
        help="Force a specific C value (skip C tuning, use this value directly).",
    )
    parser.add_argument(
        "--threshold-values",
        nargs="+",
        type=float,
        default=None,
        help="Threshold values to try. If None, uses range [0.0, 0.05, ..., 1.0].",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory. If None, creates 'tuning_results' in csv-folder.",
    )
    parser.add_argument(
        "--num-preds",
        type=int,
        default=20,
        help="Number of predictions to consider for re-ranking (K).",
    )
    parser.add_argument(
        "--positive-dist-threshold",
        type=int,
        default=25,
        help="Distance threshold in meters for positive match.",
    )

    args = parser.parse_args()

    csv_folder = Path(args.csv_folder)
    if not csv_folder.exists():
        raise FileNotFoundError(f"CSV folder does not exist: {csv_folder}")

    # Auto-detect CSV files
    csv_files = list(csv_folder.glob("*.csv"))
    
    # Find training CSVs
    train_sun_csv = None
    train_night_csv = None
    train_combined_csv = None
    val_csv = None
    
    for csv_file in csv_files:
        name = csv_file.name
        if "_svox_train_sun.csv" in name:
            train_sun_csv = csv_file
        elif "_svox_train_night.csv" in name:
            train_night_csv = csv_file
        elif "_svox_train.csv" in name and "_sun" not in name and "_night" not in name:
            train_combined_csv = csv_file
        elif "_sf_xs_val.csv" in name:
            val_csv = csv_file
    
    if train_sun_csv is None or train_night_csv is None or train_combined_csv is None:
        raise FileNotFoundError(
            f"Could not find all required training CSVs in {csv_folder}. "
            f"Need: *_svox_train_sun.csv, *_svox_train_night.csv, *_svox_train.csv"
        )
    if val_csv is None:
        raise FileNotFoundError(f"Could not find validation CSV (*_sf_xs_val.csv) in {csv_folder}")

    print("=" * 80)
    print("Found CSV files:")
    print(f"  Sun-only:    {train_sun_csv.name}")
    print(f"  Night-only:  {train_night_csv.name}")
    print(f"  Combined:    {train_combined_csv.name}")
    print(f"  Validation:  {val_csv.name}")
    print("=" * 80)

    # Set up output directory
    if args.output_dir is None:
        output_dir = csv_folder / "tuning_results"
    else:
        output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set threshold values
    if args.threshold_values is None:
        threshold_values = np.arange(0.0, 1.01, 0.05).tolist()
    else:
        threshold_values = args.threshold_values

    print(f"\nThreshold values: {threshold_values}")
    print(f"Number of predictions (K): {args.num_preds}")

    # Train 3 models and evaluate thresholds
    training_configs = [
        ("sun", train_sun_csv),
        ("night", train_night_csv),
        ("combined", train_combined_csv),
    ]

    all_results = {}

    for config_name, train_csv in training_configs:
        print("\n" + "=" * 80)
        print(f"Training LR model: {config_name.upper()}")
        print("=" * 80)

        # Step 1: Find best C (or use forced C)
        if args.force_C is not None:
            print(f"\nUsing forced C = {args.force_C:.2f} for {config_name}...")
            best_C = args.force_C
            model_bundle = train_lr_model(train_csv, best_C)
            model = model_bundle["model"]
            scaler = model_bundle["scaler"]
        else:
            print(f"\nFinding best C for {config_name}...")
            best_C, model, scaler = find_best_C(train_csv, val_csv, args.C_values)
            print(f"Best C: {best_C:.2f}")

        # Step 2: Evaluate R@1 for each threshold
        print(f"\nEvaluating R@1 for {len(threshold_values)} thresholds...")
        threshold_r1_results = []
        
        for idx, threshold in enumerate(threshold_values):
            r1, cls_acc, num_easy, num_hard = compute_r1_for_threshold(
                val_csv=val_csv,
                val_preds_dir=args.val_preds_dir,
                val_top20_inliers_dir=args.val_top20_inliers_dir,
                model=model,
                scaler=scaler,
                threshold=threshold,
                num_preds=args.num_preds,
                positive_dist_threshold=args.positive_dist_threshold,
                debug=(idx == 0),  # Only debug for first threshold
            )
            
            threshold_r1_results.append({
                "threshold": threshold,
                "recall_at_1": r1,
                "classification_accuracy": cls_acc,
                "num_easy": num_easy,
                "num_hard": num_hard,
                "pct_easy": (num_easy / (num_easy + num_hard) * 100) if (num_easy + num_hard) > 0 else 0,
                "pct_hard": (num_hard / (num_easy + num_hard) * 100) if (num_easy + num_hard) > 0 else 0,
        })
        
            # Print detailed info for key thresholds
            if idx == 0 or idx == len(threshold_values) - 1 or threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
                print(f"  Threshold {threshold:.2f}: R@1={r1:.2f}%, ClsAcc={cls_acc:.2f}%, Easy={num_easy} ({num_easy/(num_easy+num_hard)*100:.1f}%), Hard={num_hard} ({num_hard/(num_easy+num_hard)*100:.1f}%)")
            
            if len(threshold_r1_results) % 5 == 0 and idx not in [0, len(threshold_values) - 1]:
                print(f"  Processed {len(threshold_r1_results)}/{len(threshold_values)} thresholds...")

        # Find best threshold (maximizes classification accuracy, not R@1)
        # This is because validation sets like SF-XS val are too easy, leading to threshold=0.0
        # Classification accuracy better reflects the adaptive strategy's ability to distinguish easy vs hard queries
        best_threshold_result = max(threshold_r1_results, key=lambda x: x["classification_accuracy"])
        
        # Get baseline (threshold 0.0) and full re-ranking (threshold 1.0) results
        baseline_result = threshold_r1_results[0]  # threshold 0.0
        full_rerank_result = threshold_r1_results[-1]  # threshold 1.0
        
        print(f"\n{'='*80}")
        print(f"Results Summary for {config_name.upper()} model:")
        print(f"{'='*80}")
        print(f"Baseline (all easy, threshold=0.00):")
        print(f"  R@1: {baseline_result['recall_at_1']:.2f}%")
        print(f"  Classification Accuracy: {baseline_result['classification_accuracy']:.2f}%")
        print(f"  Easy: {baseline_result['num_easy']} ({baseline_result['pct_easy']:.1f}%), Hard: {baseline_result['num_hard']} ({baseline_result['pct_hard']:.1f}%)")
        print(f"\nFull re-ranking (all hard, threshold=1.00):")
        print(f"  R@1: {full_rerank_result['recall_at_1']:.2f}%")
        print(f"  Classification Accuracy: {full_rerank_result['classification_accuracy']:.2f}%")
        print(f"  Easy: {full_rerank_result['num_easy']} ({full_rerank_result['pct_easy']:.1f}%), Hard: {full_rerank_result['num_hard']} ({full_rerank_result['pct_hard']:.1f}%)")
        print(f"\nBest threshold (optimized for classification accuracy): {best_threshold_result['threshold']:.2f}")
        print(f"  R@1: {best_threshold_result['recall_at_1']:.2f}%")
        print(f"  Classification Accuracy: {best_threshold_result['classification_accuracy']:.2f}%")
        print(f"  Easy queries: {best_threshold_result['num_easy']} ({best_threshold_result['pct_easy']:.1f}%)")
        print(f"  Hard queries: {best_threshold_result['num_hard']} ({best_threshold_result['pct_hard']:.1f}%)")
        print(f"{'='*80}")

        # Save model
        model_path = output_dir / f"lr_model_{config_name}_C{best_C:.2f}.pkl"
        model_bundle = {
            "scaler": scaler,
            "model": model,
            "best_C": best_C,
            "best_threshold": best_threshold_result["threshold"],
            "best_r1": best_threshold_result["recall_at_1"],
            "best_cls_acc": best_threshold_result["classification_accuracy"],
            "config_name": config_name,
        }
        joblib.dump(model_bundle, model_path)
        print(f"Saved model to {model_path}")

        all_results[config_name] = {
            "best_C": best_C,
            "best_threshold": best_threshold_result["threshold"],
            "best_r1": best_threshold_result["recall_at_1"],
            "best_cls_acc": best_threshold_result["classification_accuracy"],
            "threshold_results": threshold_r1_results,
        }

    # Generate plot
    print("\n" + "=" * 80)
    print("Generating plot: Threshold vs R@1")
    print("=" * 80)
    
    plt.figure(figsize=(10, 6))
    colors = {"sun": "orange", "night": "darkblue", "combined": "green"}
    
    for config_name, results in all_results.items():
        thresholds = [r["threshold"] for r in results["threshold_results"]]
        r1_values = [r["recall_at_1"] for r in results["threshold_results"]]
        
        plt.plot(
            thresholds,
            r1_values,
            marker="o",
            label=f"{config_name.capitalize()} (best: {results['best_threshold']:.2f}, R@1: {results['best_r1']:.2f}%)",
            color=colors[config_name],
            linewidth=2,
            markersize=4,
        )
        
        # Mark best threshold
        best_idx = thresholds.index(results["best_threshold"])
        plt.plot(
            results["best_threshold"],
            results["best_r1"],
            marker="*",
            markersize=15,
            color=colors[config_name],
            zorder=5,
        )

    plt.xlabel("Threshold", fontsize=12)
    plt.ylabel("Recall@1 (%)", fontsize=12)
    plt.title("Threshold vs R@1 on Validation Set\n(Threshold optimized for classification accuracy)", fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_path = output_dir / "threshold_vs_r1_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to {plot_path}")

    # Save summary
    summary_path = output_dir / "tuning_summary.txt"
    with open(summary_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("LR Hyperparameter Tuning Summary (Classification Accuracy Optimization)\n")
        f.write("=" * 80 + "\n\n")
        f.write("NOTE: Threshold selection optimizes classification accuracy (how well LR predicts\n")
        f.write("if top-1 is correct), not R@1. This is because validation sets like SF-XS val\n")
        f.write("are often too easy, leading to threshold=0.0 (no re-ranking) when optimizing R@1.\n")
        f.write("Classification accuracy better reflects the adaptive strategy's ability to\n")
        f.write("distinguish easy vs hard queries. R@1 curves are still shown for analysis.\n\n")
        f.write(f"CSV Folder: {csv_folder}\n")
        f.write(f"Validation Predictions: {args.val_preds_dir}\n")
        f.write(f"Validation Top-20 Inliers: {args.val_top20_inliers_dir}\n\n")
        
        # Check baseline performance
        baseline_r1 = all_results[list(all_results.keys())[0]]["threshold_results"][0]["recall_at_1"]
        full_rerank_r1 = all_results[list(all_results.keys())[0]]["threshold_results"][-1]["recall_at_1"]
        
        f.write("NOTE: Validation Set Analysis\n")
        f.write(f"  Baseline (no re-ranking): {baseline_r1:.2f}% R@1\n")
        f.write(f"  Full re-ranking: {full_rerank_r1:.2f}% R@1\n")
        if baseline_r1 > full_rerank_r1:
            f.write("  WARNING: Validation set is very easy (high baseline R@1).\n")
            f.write("           Optimal threshold may favor minimal re-ranking.\n")
            f.write("           Re-ranking may help more on harder test sets.\n")
        f.write("\n")
        
        for config_name, results in all_results.items():
            f.write(f"{config_name.upper()} Model:\n")
            f.write(f"  Best C: {results['best_C']:.2f}\n")
            f.write(f"  Best Threshold (optimized for classification accuracy): {results['best_threshold']:.2f}\n")
            f.write(f"  Classification Accuracy at best threshold: {results['best_cls_acc']:.2f}%\n")
            f.write(f"  R@1 at best threshold: {results['best_r1']:.2f}%\n")
            # Get the result for the best threshold (based on classification accuracy)
            best_result = next(r for r in results["threshold_results"] if r["threshold"] == results["best_threshold"])
            f.write(f"  Easy queries at best threshold: {best_result['num_easy']} ({best_result['pct_easy']:.1f}%)\n")
            f.write(f"  Hard queries at best threshold: {best_result['num_hard']} ({best_result['pct_hard']:.1f}%)\n\n")
    
    print(f"Saved summary to {summary_path}")
    print("\n" + "=" * 80)
    print("TUNING COMPLETE!")
    print("=" * 80)
    print(f"Results saved in: {output_dir}")


if __name__ == "__main__":
    main()
