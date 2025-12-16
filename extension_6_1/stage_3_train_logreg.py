import argparse
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib

from extension_6_1.stage_2_feature_io import load_feature_file


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train-features",
        type=str,
        required=True,
        help="Path to training feature file (.npz) produced by extract_features.py (e.g. features_svox_train.npz)",
    )
    parser.add_argument(
        "--val-features",
        type=str,
        required=False,
        help="Optional path to validation feature file (.npz), e.g. features_sf_xs_val.npz",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="logreg_model.pkl",
        help="Where to save the trained logistic regression model",
    )

    return parser.parse_args()


def build_feature_matrix(features_dict: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Build (X, y) from a feature dictionary loaded via load_feature_file.
    X: shape (num_queries, num_features)
    y: shape (num_queries,)
    """
    labels = features_dict["labels"]
    num_inliers = features_dict["num_inliers"]
    top1_distance = features_dict["top1_distance"]
    peakiness = features_dict["peakiness"]
    sue_score = features_dict["sue_score"]

    # Stack features column-wise into a single matrix
    X = np.stack(
        [num_inliers, top1_distance, peakiness, sue_score],
        axis=1,
    ).astype("float32")
    y = labels.astype("float32")

    return X, y


def main(args):
    # Load training features
    train_features = load_feature_file(args.train_features)
    X_train, y_train = build_feature_matrix(train_features)

    # Normalize features with StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Define and train logistic regression
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
    )
    logreg.fit(X_train_scaled, y_train)

    print(f"Trained logistic regression on {X_train.shape[0]} queries.")

    # Optional validation
    if args.val_features is not None:
        val_features = load_feature_file(args.val_features)
        X_val, y_val = build_feature_matrix(val_features)
        X_val_scaled = scaler.transform(X_val)

        # Predict probabilities for the positive class
        y_val_proba = logreg.predict_proba(X_val_scaled)[:, 1]

        # Compute ROC-AUC as a sanity check
        try:
            auc = roc_auc_score(y_val, y_val_proba)
            print(f"Validation ROC-AUC: {auc:.4f}")
        except ValueError:
            print("Could not compute ROC-AUC on validation set (perhaps only one class present).")

    # Save model and scaler together
    output_path = Path(args.output_model)
    model_bundle = {
        "scaler": scaler,
        "model": logreg,
        "feature_names": ["num_inliers", "top1_distance", "peakiness", "sue_score"],
    }
    joblib.dump(model_bundle, output_path)
    print(f"Saved logistic regression model to {output_path}")


if __name__ == "__main__":
    parsed_args = parse_arguments()
    main(parsed_args)


