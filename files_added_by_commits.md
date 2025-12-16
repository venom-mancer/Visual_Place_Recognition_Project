# Files Added by Your Commits to Main Branch

This document lists all files that were added/modified by your commits (Extension 6.1 implementation) compared to the original template state (commit `3660ad3`).

## Extension 6.1 Implementation Files (Your Work)

### Core Extension 6.1 Scripts
- `vpr_uncertainty/extract_features.py` - Stage 1: Feature extraction
- `vpr_uncertainty/train_logreg.py` - Stage 3: Train logistic regression model
- `vpr_uncertainty/apply_logreg.py` - Stage 4: Apply logistic regression
- `vpr_uncertainty/adaptive_reranking_eval.py` - Stage 5: Adaptive re-ranking evaluation
- `vpr_uncertainty/feature_io.py` - Helper for loading feature files
- `vpr_uncertainty/baselines.py` - Uncertainty baseline implementations
- `vpr_uncertainty/eval.py` - VPR evaluation script

### Supporting Scripts
- `match_queries_preds.py` - Image matching script (SuperPoint + LightGlue)
- `util.py` - Utility functions
- `reranking.py` - Re-ranking utilities
- `clear_gpu_memory.py` - GPU memory management
- `download_datasets.py` - Dataset download script
- `setup_temp_dir.py` - Temporary directory setup

### Documentation Files
- `docs_extension_6_1_overview.md` - Extension 6.1 overview
- `stage_1.md` - Stage 1 documentation
- `stage_3.md` - Stage 3 documentation
- `stage_4.md` - Stage 4 documentation
- `stage_5.md` - Stage 5 documentation
- `stage_6.md` - Stage 6 documentation
- `execution.md` - Execution plan
- `pdf.txt` - Project requirements (from PDF)

### Project Files
- `README.md` - Modified README
- `start_your_project.ipynb` - Jupyter notebook starter

## VPR Methods Evaluation (Template/Base Code)
These files are part of the base VPR evaluation framework:
- `VPR-methods-evaluation/main.py` - Main VPR evaluation script
- `VPR-methods-evaluation/parser.py` - Argument parser
- `VPR-methods-evaluation/visualizations.py` - Visualization utilities
- `VPR-methods-evaluation/test_dataset.py` - Dataset testing
- `VPR-methods-evaluation/vpr_models/` - VPR model implementations (CosPlace, MixVPR, etc.)
- `VPR-methods-evaluation/requirements.txt` - Dependencies

## Summary
**Total Extension 6.1 specific files:** ~15 Python scripts + 7 documentation files
**Total files added (including base VPR framework):** ~100+ files

The main branch now contains all these files in a single commit: "Project files and extension 6.1 implementation"

