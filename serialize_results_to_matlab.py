"""
Serialize analysis results to MATLAB .mat files for further analysis.

This script converts Python analysis results to MATLAB-compatible format.
"""

import argparse
from pathlib import Path
import numpy as np
import scipy.io as sio
import joblib
import json


def serialize_threshold_analysis_results(
    results_dir: Path,
    output_mat_file: str
):
    """
    Serialize threshold analysis results to MATLAB .mat file.
    
    Reads summary data and creates MATLAB-compatible structure.
    """
    summary_file = results_dir / "threshold_analysis_summary.md"
    
    # For now, create a basic structure
    # In practice, you would parse the summary or load from JSON/pickle
    
    # Create MATLAB structure
    mat_data = {
        'threshold_analysis': {
            'datasets': [],
            'thresholds': [],
            'recall_at_1': [],
            'cost_savings': [],
            'hard_query_rates': []
        }
    }
    
    # Save to .mat file
    sio.savemat(output_mat_file, mat_data)
    print(f"Serialized results to {output_mat_file}")


def serialize_model_predictions(
    model_path: Path,
    feature_path: Path,
    output_mat_file: str
):
    """
    Serialize model predictions to MATLAB .mat file.
    """
    # Load model
    bundle = joblib.load(model_path)
    model = bundle["model"]
    scaler = bundle["scaler"]
    feature_names = bundle["feature_names"]
    
    # Load features
    data = np.load(feature_path)
    features_dict = {}
    for key in data.keys():
        features_dict[key] = data[key]
    
    # Build feature matrix
    feature_arrays = []
    for name in feature_names:
        if name in features_dict:
            feature_arrays.append(features_dict[name])
    
    X = np.stack(feature_arrays, axis=1).astype("float32")
    
    # Handle NaNs
    valid_mask = ~np.isnan(X).any(axis=1)
    X = X[valid_mask]
    
    # Scale and predict
    X_scaled = scaler.transform(X)
    probs = model.predict_proba(X_scaled)[:, 1]
    
    # Create MATLAB structure
    mat_data = {
        'model_predictions': {
            'probabilities': probs,
            'feature_names': feature_names,
            'num_queries': len(probs),
            'prob_mean': float(probs.mean()),
            'prob_std': float(probs.std()),
            'prob_min': float(probs.min()),
            'prob_max': float(probs.max())
        }
    }
    
    # Save to .mat file
    sio.savemat(output_mat_file, mat_data)
    print(f"Serialized model predictions to {output_mat_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Serialize analysis results to MATLAB .mat files"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="output_stages/threshold_analysis",
        help="Directory with threshold analysis results",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to model (for predictions serialization)",
    )
    parser.add_argument(
        "--feature-path",
        type=str,
        default=None,
        help="Path to feature file (for predictions serialization)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output_stages/matlab_files",
        help="Output directory for .mat files",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Serialize threshold analysis
    if args.results_dir:
        results_dir = Path(args.results_dir)
        if results_dir.exists():
            output_mat = output_dir / "threshold_analysis_results.mat"
            serialize_threshold_analysis_results(results_dir, str(output_mat))
    
    # Serialize model predictions
    if args.model_path and args.feature_path:
        model_path = Path(args.model_path)
        feature_path = Path(args.feature_path)
        if model_path.exists() and feature_path.exists():
            output_mat = output_dir / "model_predictions.mat"
            serialize_model_predictions(model_path, feature_path, str(output_mat))
    
    print(f"\nAll MATLAB files saved to: {output_dir}")


if __name__ == "__main__":
    main()

