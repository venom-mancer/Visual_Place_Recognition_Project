"""
Train 3 models with inliers that predict HARD queries:
1. Model 1: SVOX night + sun (with inliers)
2. Model 2: SVOX night only (with inliers)
3. Model 3: SVOX sun only (with inliers)

All models predict hard queries (hard_score).
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_model_with_inliers(
    train_features: str,
    val_features: str,
    output_model: str,
    model_name: str,
    threshold_method: str = "f1",
):
    """Train a single model with inliers that predicts hard queries."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print(f"  Training features: {train_features}")
    print(f"  Validation features: {val_features}")
    print(f"  Output model: {output_model}")
    print(f"  Target: hard_score (predicts hard queries)")
    print(f"  Features: 9 (8 retrieval + num_inliers_top1)")
    
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_3_train_logreg_easy_queries",
        "--train-features", train_features,
        "--val-features", val_features,
        "--output-model", output_model,
        "--threshold-method", threshold_method,
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {model_name}")
        print(result.stderr)
        return False
    
    print(f"[OK] {model_name} trained successfully")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Train 3 models with inliers that predict HARD queries"
    )
    parser.add_argument(
        "--svox-train-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_with_inliers.npz",
        help="Path to SVOX train features with inliers (night + sun)"
    )
    parser.add_argument(
        "--svox-night-features",
        type=str,
        help="Path to SVOX night-only features with inliers (will be created if not provided)"
    )
    parser.add_argument(
        "--svox-sun-features",
        type=str,
        help="Path to SVOX sun-only features with inliers (will be created if not provided)"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_with_inliers.npz",
        help="Path to validation features with inliers (SF-XS val)"
    )
    parser.add_argument(
        "--svox-train-inliers-dir",
        type=str,
        default="logs/log_svox_train/2025-12-16_17-08-46/preds_superpoint-lg",
        help="Path to SVOX train inliers directory"
    )
    parser.add_argument(
        "--val-inliers-dir",
        type=str,
        default="logs/log_sf_xs_val/2025-12-16_21-55-53/preds_superpoint-lg",
        help="Path to validation inliers directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_three_way_comparison_with_inliers",
        help="Output directory for models"
    )
    parser.add_argument(
        "--threshold-method",
        type=str,
        default="f1",
        choices=["f1", "recall"],
        help="Threshold selection method during validation (matches stage_3_train_logreg_easy_queries.py)",
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if base feature files exist
    if not Path(args.svox_train_features).exists():
        print(f"ERROR: SVOX train features with inliers not found: {args.svox_train_features}")
        print("  Please run add_inliers_to_features.py first")
        return 1
    
    if not Path(args.val_features).exists():
        print(f"ERROR: Validation features with inliers not found: {args.val_features}")
        print("  Please run add_inliers_to_features.py first")
        return 1
    
    # Create night and sun feature files with inliers if not provided
    svox_night_features = args.svox_night_features
    svox_sun_features = args.svox_sun_features
    
    if not svox_night_features or not Path(svox_night_features).exists():
        print(f"\nCreating SVOX night features with inliers...")
        # First, we need the base night features
        base_night_features = "data/features_and_predictions/features_svox_train_night_improved.npz"
        if not Path(base_night_features).exists():
            print(f"ERROR: Base night features not found: {base_night_features}")
            print("  Please run filter_svox_features_by_subset.py first with --subset night")
            return 1
        
        svox_night_features = "data/features_and_predictions/features_svox_train_night_with_inliers.npz"
        print(f"  Adding inliers to: {base_night_features}")
        print(f"  Output: {svox_night_features}")
        
        cmd = [
            sys.executable,
            "add_inliers_to_features.py",
            "--feature-file", base_night_features,
            "--preds-dir", str(Path(args.svox_train_inliers_dir).parent / "preds"),
            "--inliers-dir", args.svox_train_inliers_dir,
            "--output-file", svox_night_features
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to add inliers to night features")
            print(result.stderr)
            return 1
    
    if not svox_sun_features or not Path(svox_sun_features).exists():
        print(f"\nCreating SVOX sun features with inliers...")
        # First, we need the base sun features
        base_sun_features = "data/features_and_predictions/features_svox_train_sun_improved.npz"
        if not Path(base_sun_features).exists():
            print(f"ERROR: Base sun features not found: {base_sun_features}")
            print("  Please run filter_svox_features_by_subset.py first with --subset sun")
            return 1
        
        svox_sun_features = "data/features_and_predictions/features_svox_train_sun_with_inliers.npz"
        print(f"  Adding inliers to: {base_sun_features}")
        print(f"  Output: {svox_sun_features}")
        
        cmd = [
            sys.executable,
            "add_inliers_to_features.py",
            "--feature-file", base_sun_features,
            "--preds-dir", str(Path(args.svox_train_inliers_dir).parent / "preds"),
            "--inliers-dir", args.svox_train_inliers_dir,
            "--output-file", svox_sun_features
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"ERROR: Failed to add inliers to sun features")
            print(result.stderr)
            return 1
    
    # Train Model 1: Night + Sun (with inliers, predicts hard)
    model1_path = output_dir / "logreg_hard_night_sun_with_inliers.pkl"
    if not train_model_with_inliers(
        args.svox_train_features,
        args.val_features,
        str(model1_path),
        "Model 1: SVOX Night + Sun (with inliers, predicts hard)",
        threshold_method=args.threshold_method,
    ):
        return 1
    
    # Train Model 2: Night only (with inliers, predicts hard)
    model2_path = output_dir / "logreg_hard_night_only_with_inliers.pkl"
    if not train_model_with_inliers(
        svox_night_features,
        args.val_features,
        str(model2_path),
        "Model 2: SVOX Night Only (with inliers, predicts hard)",
        threshold_method=args.threshold_method,
    ):
        return 1
    
    # Train Model 3: Sun only (with inliers, predicts hard)
    model3_path = output_dir / "logreg_hard_sun_only_with_inliers.pkl"
    if not train_model_with_inliers(
        svox_sun_features,
        args.val_features,
        str(model3_path),
        "Model 3: SVOX Sun Only (with inliers, predicts hard)",
        threshold_method=args.threshold_method,
    ):
        return 1
    
    print(f"\n{'='*70}")
    print("All 3 models trained successfully!")
    print(f"{'='*70}")
    print(f"\nModels saved in: {output_dir}")
    print(f"  - Model 1 (Night + Sun): {model1_path}")
    print(f"  - Model 2 (Night Only): {model2_path}")
    print(f"  - Model 3 (Sun Only): {model3_path}")
    print(f"\nAll models:")
    print(f"  - Use 9 features (8 retrieval + num_inliers_top1)")
    print(f"  - Predict hard queries (hard_score)")
    print(f"\nNext step: Run plot_recall1_vs_threshold_validation.py with these models")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

