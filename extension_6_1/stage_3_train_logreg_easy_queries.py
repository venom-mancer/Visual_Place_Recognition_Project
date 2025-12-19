"""
Stage 3 (Improved): Train Logistic Regression to predict EASY queries with optimal threshold.

This version:
- Predicts "easy" queries (label = 1 if correct, 0 if wrong)
- Uses 8 improved features (available before image matching)
- Automatically finds optimal threshold on validation set
- No hard thresholding - threshold is learned/optimized
"""

import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import joblib
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
    """
    Build (X, y) from a feature dictionary.
    X: 8 improved features (available before image matching)
    y: easy_score (1 = easy/correct, 0 = hard/wrong) - Top-1 correctness
    """
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
    # This is Top-1 correctness: 1 = correct (within threshold), 0 = wrong (outside threshold)
    labels = features_dict["labels"]  # 1 = correct, 0 = wrong
    easy_score = labels.astype("float32")  # 1 = easy/correct, 0 = hard/wrong
    
    return X, easy_score, feature_names


def find_optimal_threshold(y_true, y_probs, method="f1"):
    """
    Find optimal threshold on validation set.
    
    Args:
        y_true: True labels (1 = easy, 0 = hard)
        y_probs: Predicted probabilities (probability of being easy)
        method: "f1" (maximize F1) or "recall" (target recall rate)
    
    Returns:
        optimal_threshold: Best threshold value
        best_score: Best F1 or recall score
    """
    thresholds = np.arange(0.1, 0.95, 0.01)
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in thresholds:
        y_pred = (y_probs >= threshold).astype(int)
        
        if method == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif method == "recall":
            # Target: maximize recall of easy queries (we want to catch all easy queries)
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def main(args):
    # Load training features
    train_features = load_feature_file(args.train_features)
    X_train, y_train, feature_names = build_feature_matrix(train_features)
    
    # Handle NaNs
    valid_mask = ~np.isnan(X_train).any(axis=1)
    X_train = X_train[valid_mask]
    y_train = y_train[valid_mask]
    
    if (~valid_mask).sum() > 0:
        print(f"Removed {(~valid_mask).sum()} queries with NaN features")
    
    print(f"\nTraining logistic regression on {X_train.shape[0]} queries.")
    print(f"  Features: {', '.join(feature_names)} ({len(feature_names)} features)")
    print(f"  Target: easy_score (1 = easy/correct, 0 = hard/wrong) - Top-1 correctness")
    
    # Statistics
    num_easy = y_train.sum()
    num_hard = (1 - y_train).sum()
    print(f"\nTarget (easy_score) statistics:")
    print(f"  Easy queries (skip re-ranking): {int(num_easy)} ({100*num_easy/len(y_train):.1f}%)")
    print(f"  Hard queries (apply re-ranking): {int(num_hard)} ({100*num_hard/len(y_train):.1f}%)")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Tune C (regularization parameter) on validation set if available
    optimal_C = 1.0  # Default
    if args.val_features is not None:
        print(f"\n{'='*70}")
        print(f"HYPERPARAMETER TUNING: Finding optimal C (regularization)")
        print(f"{'='*70}")
        
        val_features = load_feature_file(args.val_features)
        X_val, y_val, _ = build_feature_matrix(val_features)
        
        # Handle NaNs
        val_valid_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[val_valid_mask]
        y_val = y_val[val_valid_mask]
        
        # Test different C values
        C_values = [0.01, 0.1, 1.0, 10.0, 100.0]
        best_C = 1.0
        best_val_f1 = 0.0
        
        print(f"Testing C values: {C_values}")
        for C in tqdm(C_values, desc="Tuning C"):
            # Train with this C
            logreg_temp = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=1000,
                class_weight='balanced',
            )
            logreg_temp.fit(X_train_scaled, y_train)
            
            # Evaluate on validation set
            X_val_scaled = scaler.transform(X_val)
            y_val_probs = logreg_temp.predict_proba(X_val_scaled)[:, 1]
            
            # Find optimal threshold for this C
            threshold_temp, _ = find_optimal_threshold(y_val, y_val_probs, method=args.threshold_method)
            y_val_pred = (y_val_probs >= threshold_temp).astype(int)
            val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
            
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_C = C
        
        optimal_C = best_C
        print(f"\nOptimal C: {optimal_C} (validation F1: {best_val_f1:.4f})")
        print(f"{'='*70}\n")
    
    # Train logistic regression with optimal C
    logreg = LogisticRegression(
        C=optimal_C,
        solver="lbfgs",
        max_iter=1000,
        class_weight='balanced',  # Handle class imbalance
    )
    logreg.fit(X_train_scaled, y_train)
    
    # Training predictions
    y_train_probs = logreg.predict_proba(X_train_scaled)[:, 1]
    y_train_pred = (y_train_probs >= 0.5).astype(int)
    
    train_accuracy = (y_train_pred == y_train).mean()
    train_f1 = f1_score(y_train, y_train_pred, zero_division=0)
    train_roc_auc = roc_auc_score(y_train, y_train_probs)
    
    print(f"\nTraining metrics:")
    print(f"  Accuracy: {100*train_accuracy:.1f}%")
    print(f"  F1-Score: {train_f1:.4f}")
    print(f"  ROC-AUC: {train_roc_auc:.4f}")
    
    # Optional validation - find optimal threshold and evaluate performance
    # IMPORTANT: We use validation set for:
    # 1. Tuning C (regularization parameter) - done above if val_features provided
    # 2. Finding optimal threshold (by trying different thresholds)
    # 3. Evaluating model performance (using the optimal threshold)
    # This is the correct approach - both C and threshold are learned from validation, not training
    optimal_threshold = 0.5
    if args.val_features is not None:
        # Use separate validation set (SF-XS val) for threshold selection AND evaluation
        # Note: If C tuning was done above, validation set was already loaded, but we reload here
        # for clarity (could be optimized to reuse, but this is clearer)
        val_features = load_feature_file(args.val_features)
        X_val, y_val, _ = build_feature_matrix(val_features)
        
        # Handle NaNs
        val_valid_mask = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[val_valid_mask]
        y_val = y_val[val_valid_mask]
        
        # Scale and predict (using scaler fitted on training data)
        X_val_scaled = scaler.transform(X_val)
        y_val_probs = logreg.predict_proba(X_val_scaled)[:, 1]  # Model trained with optimal_C
        
        print(f"\n{'='*70}")
        print(f"VALIDATION PHASE: Threshold Selection + Performance Evaluation")
        print(f"{'='*70}")
        print(f"Step 1: Find optimal threshold on validation set (model trained with C={optimal_C})")
        print(f"  - Trying different thresholds (0.1 to 0.95)")
        print(f"  - Selecting threshold that maximizes {args.threshold_method} score")
        print(f"  - This prevents overfitting (threshold not selected on training data)")
        
        # Find optimal threshold on VALIDATION set (prevents overfitting)
        optimal_threshold, best_score = find_optimal_threshold(
            y_val, y_val_probs, method=args.threshold_method
        )
        
        print(f"\nStep 2: Evaluate model performance using optimal threshold")
        print(f"  - Optimal threshold: {optimal_threshold:.3f} (best {args.threshold_method}: {best_score:.4f})")
        print(f"  - Now evaluating accuracy, F1, precision, recall with this threshold")
        
        # Validation metrics with optimal threshold
        y_val_pred = (y_val_probs >= optimal_threshold).astype(int)
        val_accuracy = (y_val_pred == y_val).mean()
        val_f1 = f1_score(y_val, y_val_pred, zero_division=0)
        val_precision = precision_score(y_val, y_val_pred, zero_division=0)
        val_recall = recall_score(y_val, y_val_pred, zero_division=0)
        val_roc_auc = roc_auc_score(y_val, y_val_probs)
        
        print(f"\nValidation metrics (with optimal threshold {optimal_threshold:.3f}):")
        print(f"  Accuracy: {100*val_accuracy:.1f}%")
        print(f"  F1-Score: {val_f1:.4f}")
        print(f"  Precision: {val_precision:.4f}")
        print(f"  Recall: {val_recall:.4f}")
        print(f"  ROC-AUC: {val_roc_auc:.4f}")
        
        # Show distribution
        num_easy_pred = y_val_pred.sum()
        num_hard_pred = (1 - y_val_pred).sum()
        num_easy_actual = y_val.sum()
        num_hard_actual = (1 - y_val).sum()
        
        print(f"\nValidation predictions (threshold {optimal_threshold:.3f}):")
        print(f"  Easy queries (predicted): {num_easy_pred} ({100*num_easy_pred/len(y_val):.1f}%)")
        print(f"  Hard queries (predicted): {num_hard_pred} ({100*num_hard_pred/len(y_val):.1f}%)")
        print(f"\nActual distribution:")
        print(f"  Easy queries (actual): {num_easy_actual} ({100*num_easy_actual/len(y_val):.1f}%)")
        print(f"  Hard queries (actual): {num_hard_actual} ({100*num_hard_actual/len(y_val):.1f}%)")
    
    # Save model
    model_bundle = {
        "model": logreg,
        "scaler": scaler,
        "feature_names": feature_names,
        "optimal_threshold": optimal_threshold,
        "threshold_method": args.threshold_method,
        "target_type": "easy_score",  # Predict easy queries
        "optimal_C": optimal_C,  # Save the tuned C value
    }
    
    output_path = Path(args.output_model)
    joblib.dump(model_bundle, output_path)
    print(f"\nSaved logistic regression model to {output_path}")
    print(f"  Optimal threshold: {optimal_threshold:.3f}")
    print(f"  Threshold method: {args.threshold_method}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train logistic regression to predict EASY queries with optimal threshold"
    )
    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to training feature file (.npz)",
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default=None,
        help="Path to validation feature file (.npz) for threshold optimization. "
             "IMPORTANT: Threshold is learned from validation set only (not training) to prevent overfitting.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="logreg_easy_queries.pkl",
        help="Path to save the trained model",
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="f1",
        choices=["f1", "recall"],
        help="Method to find optimal threshold: 'f1' (maximize F1) or 'recall' (maximize recall)",
    )
    
    args = parser.parse_args()
    main(args)

