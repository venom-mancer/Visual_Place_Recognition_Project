"""
Solution 2: Temperature Scaling for Probability Calibration

Temperature scaling adjusts model confidence without changing predictions.
It makes probabilities more reliable across different datasets.
"""

import argparse
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import f1_score, precision_score, recall_score
import sys

sys.path.append(str(Path(__file__).parent))

from extension_6_1.stage_4_apply_logreg_easy_queries import load_feature_file, build_feature_matrix, find_optimal_threshold


def find_optimal_temperature(y_true, y_probs, temperature_range=(0.1, 10.0), num_temps=100):
    """
    Find optimal temperature for probability calibration.
    
    Temperature scaling: calibrated_probs = sigmoid(logit / temperature)
    For logistic regression: calibrated_probs = 1 / (1 + exp(-logit / temperature))
    
    Lower temperature -> more confident (probabilities spread out)
    Higher temperature -> less confident (probabilities closer to 0.5)
    """
    # Convert probabilities to logits
    logits = np.log(y_probs / (1 - y_probs + 1e-10))
    
    temperatures = np.linspace(temperature_range[0], temperature_range[1], num_temps)
    best_temperature = 1.0
    best_f1 = 0.0
    
    for temp in temperatures:
        # Apply temperature scaling
        scaled_logits = logits / temp
        scaled_probs = 1 / (1 + np.exp(-scaled_logits))
        
        # Find optimal threshold on scaled probabilities
        threshold, _ = find_optimal_threshold(y_true, scaled_probs, method="f1")
        y_pred = (scaled_probs >= threshold).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_temperature = temp
    
    return best_temperature, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Apply temperature scaling for probability calibration"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to test features"
    )
    parser.add_argument(
        "--output-path",
        type=str,
        required=True,
        help="Path to save calibrated predictions"
    )
    parser.add_argument(
        "--hard-queries-output",
        type=str,
        help="Path to save hard query indices"
    )
    
    args = parser.parse_args()
    
    # Load model
    bundle = joblib.load(Path(args.model_path))
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    saved_threshold = bundle.get("optimal_threshold", 0.5)
    
    # Load features
    features = load_feature_file(args.feature_path)
    X = build_feature_matrix(features, feature_names)
    y_true = features["labels"].astype("float32")
    
    # Handle NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    y_true = y_true[valid_mask]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    y_probs = model.predict_proba(X_scaled)[:, 1]
    
    print(f"Original probabilities: min={y_probs.min():.3f}, mean={y_probs.mean():.3f}, max={y_probs.max():.3f}")
    print(f"Original threshold: {saved_threshold:.3f}")
    original_hard_rate = (y_probs < saved_threshold).mean()
    print(f"Original hard query rate: {original_hard_rate:.1%}")
    
    # Find optimal temperature
    print(f"\nFinding optimal temperature...")
    optimal_temp, best_f1 = find_optimal_temperature(y_true, y_probs)
    print(f"Optimal temperature: {optimal_temp:.3f} (F1: {best_f1:.4f})")
    
    # Apply temperature scaling
    logits = np.log(y_probs / (1 - y_probs + 1e-10))
    scaled_logits = logits / optimal_temp
    calibrated_probs = 1 / (1 + np.exp(-scaled_logits))
    
    print(f"\nCalibrated probabilities: min={calibrated_probs.min():.3f}, mean={calibrated_probs.mean():.3f}, max={calibrated_probs.max():.3f}")
    
    # Find optimal threshold on calibrated probabilities
    optimal_threshold, _ = find_optimal_threshold(y_true, calibrated_probs, method="f1")
    print(f"Optimal threshold on calibrated probs: {optimal_threshold:.3f}")
    
    # Apply threshold
    y_pred = (calibrated_probs >= optimal_threshold).astype(int)
    calibrated_hard_rate = (calibrated_probs < optimal_threshold).mean()
    
    print(f"\nResults:")
    print(f"  Original hard rate: {original_hard_rate:.1%}")
    print(f"  Calibrated hard rate: {calibrated_hard_rate:.1%}")
    print(f"  Actual hard rate: {(1-y_true.mean()):.1%}")
    
    # Save results
    save_dict = {
        "probs": calibrated_probs,
        "is_easy": y_pred.astype(bool),
        "is_hard": (~y_pred.astype(bool)),
        "labels": y_true,
        "threshold": optimal_threshold,
        "temperature": optimal_temp,
        "accuracy": (y_pred == y_true).mean(),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
    }
    
    np.savez_compressed(args.output_path, **save_dict)
    print(f"\nSaved calibrated predictions to: {args.output_path}")
    
    # Save hard queries
    if args.hard_queries_output:
        hard_query_indices = np.where(~y_pred.astype(bool))[0]
        with open(args.hard_queries_output, "w") as f:
            for idx in hard_query_indices:
                f.write(f"{idx}\n")
        print(f"Saved {len(hard_query_indices)} hard query indices to: {args.hard_queries_output}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())


