import numpy as np
from pathlib import Path
from typing import Dict


def load_feature_file(path: str) -> Dict[str, np.ndarray]:
    """
    Load a single .npz feature file produced by extract_features.py.

    Returns a dictionary with keys:
      - 'labels'
      - 'num_inliers'
      - 'top1_distance'
      - 'peakiness'
      - 'sue_score'
    """
    path_obj = Path(path)
    data = np.load(path_obj)

    return {
        "labels": data["labels"].astype("float32"),
        "num_inliers": data["num_inliers"].astype("float32"),
        "top1_distance": data["top1_distance"].astype("float32"),
        "peakiness": data["peakiness"].astype("float32"),
        "sue_score": data["sue_score"].astype("float32"),
    }


def describe_feature_file(path: str) -> None:
    """
    Print a short summary of a feature file (number of queries and basic stats).
    Useful for quickly checking that train / val / test feature files look reasonable.
    """
    features = load_feature_file(path)
    labels = features["labels"]

    num_queries = labels.shape[0]
    num_positives = int(labels.sum())
    num_negatives = int(num_queries - num_positives)

    print(f"Feature file: {path}")
    print(f"  #queries: {num_queries}")
    print(f"  #positives (correct Top-1): {num_positives}")
    print(f"  #negatives (wrong Top-1):   {num_negatives}")


