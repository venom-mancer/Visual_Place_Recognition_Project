"""
Build all 3 training CSVs (sun-only, night-only, combined) for a VPR+matcher combination.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Build all 3 training CSVs (sun, night, combined) for a VPR+matcher combination."
    )
    parser.add_argument(
        "--vpr-method",
        required=True,
        choices=["cosplace", "mixvpr"],
        help="VPR method name.",
    )
    parser.add_argument(
        "--matcher-method",
        required=True,
        choices=["loftr", "superpoint-lg"],
        help="Matcher method name.",
    )
    parser.add_argument(
        "--preds-dir-sun",
        required=True,
        help="Preds directory for svox_train_sun.",
    )
    parser.add_argument(
        "--inliers-dir-sun",
        required=True,
        help="Inliers directory for svox_train_sun.",
    )
    parser.add_argument(
        "--preds-dir-night",
        required=True,
        help="Preds directory for svox_train_night.",
    )
    parser.add_argument(
        "--inliers-dir-night",
        required=True,
        help="Inliers directory for svox_train_night.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="csv_files",
        help="Output directory for CSVs (default: csv_files).",
    )

    args = parser.parse_args()

    # Get script directory
    script_dir = Path(__file__).parent
    build_script = script_dir / "build_lr_dataset.py"
    
    # Create subfolder name: e.g., "Cosplace_Loftr", "Mixvpr_Superpoint-lg"
    vpr_capitalized = args.vpr_method.capitalize()
    # Handle matcher names like "superpoint-lg" -> "Superpoint-lg"
    matcher_parts = args.matcher_method.split("-")
    matcher_capitalized = "-".join([p.capitalize() for p in matcher_parts])
    subfolder_name = f"{vpr_capitalized}_{matcher_capitalized}"
    
    # Output directory: csv_files/Cosplace_Loftr/
    base_output_dir = script_dir / args.output_dir
    output_dir = base_output_dir / subfolder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Base name for output files
    base_name = f"lr_data_{args.vpr_method}_{args.matcher_method}_svox_train"

    print("=" * 80)
    print(f"Building training CSVs for {args.vpr_method} + {args.matcher_method}")
    print("=" * 80)

    # 1. Sun-only CSV
    print("\n[1/3] Building sun-only CSV...")
    sun_csv = output_dir / f"{base_name}_sun.csv"
    cmd_sun = [
        sys.executable,
        str(build_script),
        "--preds-dirs",
        args.preds_dir_sun,
        "--inliers-dirs",
        args.inliers_dir_sun,
        "--dataset-names",
        "svox_train_sun",
        "--vpr-method",
        args.vpr_method,
        "--matcher-method",
        args.matcher_method,
        "--out-csv",
        str(sun_csv),
    ]
    subprocess.run(cmd_sun, check=True)
    print(f"✓ Created: {sun_csv}")

    # 2. Night-only CSV
    print("\n[2/3] Building night-only CSV...")
    night_csv = output_dir / f"{base_name}_night.csv"
    cmd_night = [
        sys.executable,
        str(build_script),
        "--preds-dirs",
        args.preds_dir_night,
        "--inliers-dirs",
        args.inliers_dir_night,
        "--dataset-names",
        "svox_train_night",
        "--vpr-method",
        args.vpr_method,
        "--matcher-method",
        args.matcher_method,
        "--out-csv",
        str(night_csv),
    ]
    subprocess.run(cmd_night, check=True)
    print(f"✓ Created: {night_csv}")

    # 3. Combined CSV (both sun and night)
    print("\n[3/3] Building combined CSV (sun + night)...")
    combined_csv = output_dir / f"{base_name}.csv"
    cmd_combined = [
        sys.executable,
        str(build_script),
        "--preds-dirs",
        args.preds_dir_sun,
        args.preds_dir_night,
        "--inliers-dirs",
        args.inliers_dir_sun,
        args.inliers_dir_night,
        "--dataset-names",
        "svox_train_sun",
        "svox_train_night",
        "--vpr-method",
        args.vpr_method,
        "--matcher-method",
        args.matcher_method,
        "--out-csv",
        str(combined_csv),
    ]
    subprocess.run(cmd_combined, check=True)
    print(f"✓ Created: {combined_csv}")

    print("\n" + "=" * 80)
    print("All 3 training CSVs created successfully!")
    print("=" * 80)
    print(f"  Saved in: {output_dir}")
    print(f"  • Sun-only:   {sun_csv.name}")
    print(f"  • Night-only: {night_csv.name}")
    print(f"  • Combined:    {combined_csv.name}")


if __name__ == "__main__":
    main()

