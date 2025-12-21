"""
Serialize list of all .md documentation files created for the project.
This creates a .mat file with the list of documentation files.
"""

import argparse
from pathlib import Path
import scipy.io as sio
import os


def find_md_files(docs_dir: Path) -> dict:
    """Find all .md files in the docs directory and subdirectories."""
    md_files = []
    
    # Find all .md files recursively
    for md_file in docs_dir.rglob("*.md"):
        # Get relative path from docs directory
        rel_path = md_file.relative_to(docs_dir)
        md_files.append({
            "name": md_file.name,
            "path": str(rel_path),
            "full_path": str(md_file),
            "size_bytes": md_file.stat().st_size,
        })
    
    # Sort by name
    md_files.sort(key=lambda x: x["name"])
    
    return {
        "documentation_files": md_files,
        "total_files": len(md_files),
        "docs_directory": str(docs_dir),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Serialize list of .md documentation files to MATLAB .mat file"
    )
    parser.add_argument(
        "--docs-dir",
        type=str,
        default="docs",
        help="Path to docs directory (default: docs)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="documentation_files_list.mat",
        help="Output .mat file path (default: documentation_files_list.mat)",
    )
    
    args = parser.parse_args()
    
    docs_dir = Path(args.docs_dir)
    if not docs_dir.exists():
        raise ValueError(f"Docs directory does not exist: {docs_dir}")
    
    print(f"Scanning documentation files in: {docs_dir}")
    data = find_md_files(docs_dir)
    
    print(f"\nFound {data['total_files']} documentation files:")
    for i, md_file in enumerate(data["documentation_files"], 1):
        print(f"  {i:2d}. {md_file['path']} ({md_file['size_bytes']} bytes)")
    
    # Save to MATLAB .mat file
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to MATLAB-compatible format
    mat_data = {
        "documentation_files": data["documentation_files"],
        "total_files": data["total_files"],
        "docs_directory": data["docs_directory"],
    }
    
    sio.savemat(str(output_path), mat_data, oned_as='column')
    print(f"\n[SUCCESS] Saved documentation list to: {output_path}")
    print(f"   File contains {data['total_files']} .md files")


if __name__ == "__main__":
    main()

