# Documentation Index

Welcome to the comprehensive documentation for the **Visual Place Recognition - Adaptive Re-ranking Project**.

---

## 📚 Documentation Structure

### 1. [Project Overview](PROJECT_OVERVIEW.md)
- Complete project introduction
- Key achievements and results
- Quick start guide
- Project structure
- Key concepts

### 2. [Pipeline Guide](PIPELINE_GUIDE.md)
- Complete pipeline walkthrough (Stages 0-6)
- Step-by-step instructions
- Command examples
- Comparison with baseline methods

### 3. [Methodology](METHODOLOGY.md)
- Evolution of approaches
- Detailed comparison (Regressor vs Logistic Regression)
- Feature engineering evolution
- Threshold selection strategy
- Model training details

### 4. [Results](RESULTS.md)
- SF-XS test results
- Tokyo-XS test results
- SVOX test results
- All datasets summary
- Efficiency analysis
- Recommendations

### 5. [Technical Details](TECHNICAL_DETAILS.md)
- Feature engineering details
- Bug fixes (sue_score underflow)
- Model implementation
- Pipeline implementation
- Data flow
- Key design decisions

### 6. [Stage Details](STAGE_DETAILS.md)
- Comprehensive reference for each pipeline stage
- Detailed stage-by-stage documentation
- Integrated from original stage.md files

### 7. [Threshold Calibration Guide](THRESHOLD_CALIBRATION_GUIDE.md)
- Solving dataset distribution shift problem
- How to calibrate thresholds for different test datasets
- F1-maximization vs target rate calibration

### 8. [Threshold Analysis](THRESHOLD_ANALYSIS.md)
- R@1 vs threshold plots for different datasets
- Dataset influence on threshold computation
- Cost savings calculation and visualization

---

## 🚀 Quick Navigation

### For New Users
1. Start with [Project Overview](PROJECT_OVERVIEW.md) to understand the project
2. Follow [Pipeline Guide](PIPELINE_GUIDE.md) to run the pipeline
3. Check [Results](RESULTS.md) to see experimental results

### For Understanding the Approach
1. Read [Methodology](METHODOLOGY.md) for different approaches
2. Check [Technical Details](TECHNICAL_DETAILS.md) for implementation details

### For Reproducing Results
1. Follow [Pipeline Guide](PIPELINE_GUIDE.md) step-by-step
2. Refer to [Technical Details](TECHNICAL_DETAILS.md) for implementation specifics
3. Check [Results](RESULTS.md) for expected outcomes

---

## 📊 Key Results Summary

| Dataset | Method | R@1 | Time Savings | R@1 Gain |
|---------|--------|-----|--------------|-----------|
| **SF-XS test** | Baseline | 63.1% | 100% | - |
| **SF-XS test** | Full Re-ranking | 77.4% | 0% | +14.3% |
| **SF-XS test** | **Adaptive (LogReg Easy)** | **69.8%** | **74.6%** | **+6.7%** |

*See [Results](RESULTS.md) for complete results across all datasets.*

---

## 🔗 External Resources

- **Main README**: [../README.md](../README.md)
- **Code**: [../extension_6_1/](../extension_6_1/)
- **Results**: [../output_stages/results/](../output_stages/results/)

---

*Last updated: 2025-12-18*

