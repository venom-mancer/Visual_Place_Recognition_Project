"""
Train 3 models for comparison:
1. Model 1: SVOX night + sun (current)
2. Model 2: SVOX night only
3. Model 3: SVOX sun only

Then generate comparison charts showing:
- Chart 1: Metrics used for threshold selection (F1, precision, recall)
- Chart 2: Recall@1 vs threshold for all models
"""

import argparse
import subprocess
import sys
from pathlib import Path


def train_model(
    train_features: str,
    val_features: str,
    output_model: str,
    model_name: str
):
    """Train a single model."""
    print(f"\n{'='*70}")
    print(f"Training {model_name}")
    print(f"{'='*70}")
    print(f"  Training features: {train_features}")
    print(f"  Validation features: {val_features}")
    print(f"  Output model: {output_model}")
    
    cmd = [
        sys.executable,
        "-m", "extension_6_1.stage_3_train_logreg_easy_queries",
        "--train-features", train_features,
        "--val-features", val_features,
        "--output-model", output_model,
        "--threshold-method", "f1"
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
        description="Train 3 models (night+sun, night only, sun only) and prepare for comparison"
    )
    parser.add_argument(
        "--svox-train-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_improved.npz",
        help="Path to SVOX train features (night + sun)"
    )
    parser.add_argument(
        "--svox-night-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_night_improved.npz",
        help="Path to SVOX night-only features"
    )
    parser.add_argument(
        "--svox-sun-features",
        type=str,
        default="data/features_and_predictions/features_svox_train_sun_improved.npz",
        help="Path to SVOX sun-only features"
    )
    parser.add_argument(
        "--val-features",
        type=str,
        default="data/features_and_predictions/features_sf_xs_val_improved.npz",
        help="Path to validation features (SF-XS val)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models_three_way_comparison",
        help="Output directory for models"
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if filtered feature files exist
    if not Path(args.svox_night_features).exists():
        print(f"ERROR: Night features not found: {args.svox_night_features}")
        print("  Please run filter_svox_features_by_subset.py first with --subset night")
        return 1
    
    if not Path(args.svox_sun_features).exists():
        print(f"ERROR: Sun features not found: {args.svox_sun_features}")
        print("  Please run filter_svox_features_by_subset.py first with --subset sun")
        return 1
    
    # Train Model 1: Night + Sun (current)
    model1_path = output_dir / "logreg_easy_night_sun.pkl"
    if not train_model(
        args.svox_train_features,
        args.val_features,
        str(model1_path),
        "Model 1: SVOX Night + Sun"
    ):
        return 1
    
    # Train Model 2: Night only
    model2_path = output_dir / "logreg_easy_night_only.pkl"
    if not train_model(
        args.svox_night_features,
        args.val_features,
        str(model2_path),
        "Model 2: SVOX Night Only"
    ):
        return 1
    
    # Train Model 3: Sun only
    model3_path = output_dir / "logreg_easy_sun_only.pkl"
    if not train_model(
        args.svox_sun_features,
        args.val_features,
        str(model3_path),
        "Model 3: SVOX Sun Only"
    ):
        return 1
    
    print(f"\n{'='*70}")
    print("All 3 models trained successfully!")
    print(f"{'='*70}")
    print(f"\nModels saved in: {output_dir}")
    print(f"  - Model 1 (Night + Sun): {model1_path}")
    print(f"  - Model 2 (Night Only): {model2_path}")
    print(f"  - Model 3 (Sun Only): {model3_path}")
    print(f"\nNext step: Run compare_three_models.py to generate comparison charts")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

