# Documentation Index

Welcome to the comprehensive documentation for the **Visual Place Recognition - Adaptive Re-ranking Project**.

---

## 📚 Documentation Structure

### 1. [Project Overview](01_PROJECT_OVERVIEW.md)
- Complete project introduction
- Key achievements and results
- Quick start guide
- Project structure
- Key concepts

### 2. [Pipeline Guide](02_PIPELINE_GUIDE.md)
- Complete pipeline walkthrough (Stages 0-6)
- Step-by-step instructions
- Command examples
- Comparison with baseline methods

### 3. [Methodology](03_METHODOLOGY.md)
- Evolution of approaches
- Detailed comparison (Regressor vs Logistic Regression)
- Feature engineering evolution
- Threshold selection strategy
- Model training details

### 4. [Results](04_RESULTS.md)
- SF-XS test results
- Tokyo-XS test results
- SVOX test results
- All datasets summary
- Efficiency analysis
- Recommendations

### 5. [Technical Details](05_TECHNICAL_DETAILS.md)
- Feature engineering details
- Bug fixes (sue_score underflow)
- Model implementation
- Pipeline implementation
- Data flow
- Key design decisions

### 6. [Stage Details](06_STAGE_DETAILS.md)
- Comprehensive reference for each pipeline stage
- Detailed stage-by-stage documentation
- Integrated from original stage.md files

### 7. [Colab Execution Guide](07_COLAB_EXECUTION_GUIDE.md)
- How to run the pipeline in Google Colab
- Free Colab vs Colab Pro comparison
- Step-by-step Colab setup
- Time estimates and tips

### 8. [Validation Features](08_VALIDATION_FEATURES.md)
- Complete list of 8 features used during validation
- Feature descriptions and meanings
- Why num_inliers is not available
- How features are used in validation

### 9. [Dataset Constraints](09_DATASET_CONSTRAINTS.md)
- Dataset usage rules and constraints
- Training/validation/test split requirements

### 10. [Improving Generalization](10_IMPROVING_GENERALIZATION.md)
- Strategies for improving model generalization
- Threshold calibration approaches
- Cross-dataset performance

### 11. [Threshold Analysis](11_THRESHOLD_ANALYSIS.md)
- R@1 vs threshold plots for different datasets
- Dataset influence on threshold computation
- Cost savings calculation and visualization

### 12. [Threshold Analysis Quick Start](12_THRESHOLD_ANALYSIS_QUICK_START.md)
- Quick guide for threshold analysis
- Essential commands and examples

### 13. [Threshold Analysis Workflow](13_THRESHOLD_ANALYSIS_WORKFLOW.md)
- Complete workflow for threshold analysis
- Step-by-step process

### 14. [Threshold Calibration Guide](14_THRESHOLD_CALIBRATION_GUIDE.md)
- Solving dataset distribution shift problem
- How to calibrate thresholds for different test datasets
- F1-maximization vs target rate calibration

### 15. [Threshold Calibration Explained](15_THRESHOLD_CALIBRATION_EXPLAINED.md)
- Detailed explanation of threshold calibration
- Technical details and methodology

### 16. [Threshold Calibration Simple](16_THRESHOLD_CALIBRATION_SIMPLE.md)
- Simple, beginner-friendly explanation
- Easy-to-understand examples

### 17. [Threshold Generalization Analysis](17_THRESHOLD_GENERALIZATION_ANALYSIS.md)
- Analysis of threshold generalization across datasets
- Problem identification and solutions

### 18. [Documentation Structure](18_DOCUMENTATION_STRUCTURE.md)
- Overview of documentation organization
- How to navigate the documentation

---

## 🚀 Quick Navigation

### For New Users
1. Start with [Project Overview](01_PROJECT_OVERVIEW.md) to understand the project
2. Follow [Pipeline Guide](02_PIPELINE_GUIDE.md) to run the pipeline
3. Check [Results](04_RESULTS.md) to see experimental results

### For Understanding the Approach
1. Read [Methodology](03_METHODOLOGY.md) for different approaches
2. Check [Technical Details](05_TECHNICAL_DETAILS.md) for implementation details

### For Reproducing Results
1. Follow [Pipeline Guide](02_PIPELINE_GUIDE.md) step-by-step
2. Refer to [Technical Details](05_TECHNICAL_DETAILS.md) for implementation specifics
3. Check [Results](04_RESULTS.md) for expected outcomes

---

## 📊 Key Results Summary

| Dataset | Method | R@1 | Time Savings | R@1 Gain |
|---------|--------|-----|--------------|-----------|
| **SF-XS test** | Baseline | 63.1% | 100% | - |
| **SF-XS test** | Full Re-ranking | 77.4% | 0% | +14.3% |
| **SF-XS test** | **Adaptive (LogReg Easy)** | **69.8%** | **74.6%** | **+6.7%** |

*See [Results](04_RESULTS.md) for complete results across all datasets.*

---

## 🔗 External Resources

- **Main README**: [../README.md](../README.md)
- **Code**: [../extension_6_1/](../extension_6_1/)
- **Results**: [../output_stages/results/](../output_stages/results/)

---

*Last updated: 2025-12-18*

