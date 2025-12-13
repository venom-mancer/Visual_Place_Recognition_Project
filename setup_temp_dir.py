"""
Utility module to set up project-local temporary directory.
This ensures all temporary files are created in the project directory (e.g., E drive)
instead of the system temp directory (e.g., C drive).
"""

import os
import tempfile
from pathlib import Path


def setup_project_temp_directory(project_root=None):
    """
    Set up temporary directory within the project folder to avoid filling C drive.
    This function sets environment variables and creates the temp directory.
    
    Parameters
    ----------
    project_root : Path, optional
        Path to project root. If None, will try to auto-detect.
    
    Returns
    -------
    Path
        Path to the created temporary directory
    """
    if project_root is None:
        # Try to find project root by looking for common project markers
        # Start from current file location
        current_file = Path(__file__).resolve()
        project_root = current_file.parent
        
        # Navigate up to find project root
        for _ in range(10):  # Limit search depth
            # Look for project root indicators
            if (project_root / "VPR-methods-evaluation").exists() or \
               (project_root / "image-matching-models").exists() or \
               project_root.name == "Visual-Place-Recognition-Project":
                break
            if project_root.parent == project_root:  # Reached filesystem root
                # Fall back to current file's parent
                project_root = Path(__file__).parent
                break
            project_root = project_root.parent
    
    project_root = Path(project_root).absolute()
    temp_dir = project_root / ".temp"
    
    # Create temp directory if it doesn't exist
    temp_dir.mkdir(exist_ok=True)
    
    # Set environment variables for temp directories (affects tempfile, torch, etc.)
    temp_str = str(temp_dir)
    os.environ["TEMP"] = temp_str
    os.environ["TMP"] = temp_str
    os.environ["TMPDIR"] = temp_str  # For Unix-like systems
    
    # Also set for Python's tempfile module
    tempfile.tempdir = temp_str
    
    return temp_dir


# Auto-setup when imported (optional, can be called explicitly)
if __name__ != "__main__":
    # Only auto-setup if not being run as a script
    try:
        setup_project_temp_directory()
    except Exception:
        # Silently fail if there are issues, let individual scripts handle it
        pass

