import argparse
from pathlib import Path

import joblib
import numpy as np

from vpr_uncertainty.feature_io import load_feature_file
from vpr_uncertainty.train_logreg import build_feature_matrix


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained logistic regression model (.pkl) saved by train_logreg.py",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        required=True,
        help="Path to feature file (.npz) produced by extract_features.py",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="logreg_outputs.npz",
        help="Where to save probabilities and easy/hard flags",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold for classifying queries as easy vs hard",
    )

    return parser.parse_args()


def main(args):
    # Load model and scaler
    bundle = joblib.load(Path(args.model_path))
    scaler = bundle["scaler"]
    model = bundle["model"]

    # Load features and build matrix
    features = load_feature_file(args.feature_path)
    X, y = build_feature_matrix(features)

    # Scale features and compute probabilities
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]  # p(Top-1 correct)

    # Simple decision rule
    is_easy = probs >= args.threshold
    is_hard = ~is_easy

    # Save outputs
    np.savez_compressed(
        args.output_path,
        probs=probs.astype("float32"),
        is_easy=is_easy,
        is_hard=is_hard,
        labels=y.astype("float32"),
    )

    num_queries = probs.shape[0]
    print(f"Processed {num_queries} queries.")
    print(f"  Easy (skip re-ranking): {int(is_easy.sum())}")
    print(f"  Hard (apply re-ranking): {int(is_hard.sum())}")
    print(f"Saved results to {args.output_path}")


if __name__ == "__main__":
    parsed = parse_arguments()
    main(parsed)


