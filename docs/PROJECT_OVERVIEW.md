# Visual Place Recognition - Adaptive Re-ranking Project

## Project Overview

This project implements an **adaptive re-ranking strategy** for Visual Place Recognition (VPR) that selectively applies expensive image matching only to "hard" queries, achieving significant time savings while maintaining good accuracy.

### Core Problem

- **VPR Task**: Given a query image, find its location by retrieving similar images from a database
- **Challenge**: Full re-ranking (image matching) is computationally expensive but improves accuracy
- **Solution**: Predict which queries are "hard" before image matching, apply re-ranking only to those

---

## Key Achievements

### Performance Results (SF-XS Test)

| Method | R@1 | Time Savings | R@1 Gain vs Baseline |
|--------|-----|--------------|----------------------|
| **Baseline** (Retrieval-only) | 63.1% | 100% | - |
| **Full Re-ranking** | 77.4% | 0% | +14.3% |
| **Adaptive (LogReg Easy)** | 69.8% | **74.6%** | **+6.7%** |

### Best Approach: Logistic Regression (Easy Queries)

- ✅ **92.5% validation accuracy**
- ✅ **74.6% time savings** (only 25.4% queries re-ranked)
- ✅ **+6.7% R@1 improvement** over baseline
- ✅ **Optimal threshold learned** from validation data (0.410)

---

## Project Structure

```
Visual_Place_Recognition_Project/
├── README.md                          # Main project README
├── docs/                              # Comprehensive documentation
│   ├── PROJECT_OVERVIEW.md           # This file
│   ├── PIPELINE_GUIDE.md             # Complete pipeline walkthrough
│   ├── METHODOLOGY.md                # Approaches and strategies
│   └── RESULTS.md                    # All experimental results
├── extension_6_1/                     # Adaptive re-ranking implementation
│   ├── stage_1_extract_features_no_inliers.py
│   ├── stage_2_feature_io.py
│   ├── stage_3_train_logreg_easy_queries.py
│   ├── stage_4_apply_logreg_easy_queries.py
│   └── stage_5_adaptive_reranking_eval.py
├── VPR-methods-evaluation/            # VPR baseline evaluation
├── data/                              # Datasets
│   ├── sf_xs/                        # San Francisco XS
│   ├── tokyo_xs/                     # Tokyo XS
│   └── svox/                         # SVOX dataset
└── output_stages/                     # Analysis and results
    ├── results/                      # Experimental results
    ├── analysis/                     # Feature analysis
    ├── fixes/                        # Bug fixes documentation
    └── execution/                    # Execution logs
```

---

## Quick Start

### 1. Installation

```bash
cd image-matching-models
pip install -e .[all]
pip install faiss-cpu
cd ..
```

### 2. Download Datasets

```bash
python download_datasets.py
```

### 3. Run VPR Evaluation

```bash
python VPR-methods-evaluation/main.py \
  --num_workers 4 \
  --batch_size 32 \
  --log_dir log_sf_xs_test \
  --method=cosplace --backbone=ResNet18 --descriptors_dimension=512 \
  --image_size 512 512 \
  --database_folder data/sf_xs/test/gallery \
  --queries_folder data/sf_xs/test/queries \
  --num_preds_to_save 20 \
  --recall_values 1 5 10 20 \
  --save_for_uncertainty \
  --device cuda
```

### 4. Extract Features

```bash
python -m extension_6_1.stage_1_extract_features_no_inliers \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --z-data-path logs/log_sf_xs_test/[timestamp]/z_data.torch \
  --output-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --positive-dist-threshold 25
```

### 5. Train Model (if needed)

```bash
python -m extension_6_1.stage_3_train_logreg_easy_queries \
  --train-features data/features_and_predictions/features_svox_train_improved.npz \
  --val-features data/features_and_predictions/features_sf_xs_val_improved.npz \
  --output-model logreg_easy_queries_optimal.pkl
```

### 6. Apply Model

```bash
python -m extension_6_1.stage_4_apply_logreg_easy_queries \
  --model-path logreg_easy_queries_optimal.pkl \
  --feature-path data/features_and_predictions/features_sf_xs_test_improved.npz \
  --output-path logreg_easy_test.npz \
  --hard-queries-output data/features_and_predictions/hard_queries_test.txt
```

### 7. Adaptive Image Matching

```bash
python match_queries_preds_adaptive.py \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --hard-queries-list data/features_and_predictions/hard_queries_test.txt \
  --out-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --matcher superpoint-lg \
  --device cuda \
  --num-preds 20
```

### 8. Evaluate Adaptive Re-ranking

```bash
python -m extension_6_1.stage_5_adaptive_reranking_eval \
  --preds-dir logs/log_sf_xs_test/[timestamp]/preds \
  --inliers-dir logs/log_sf_xs_test/[timestamp]/preds_superpoint-lg_adaptive \
  --logreg-output logreg_easy_test.npz \
  --num-preds 20 \
  --positive-dist-threshold 25 \
  --recall-values 1 5 10 20
```

---

## Documentation Navigation

- **[Pipeline Guide](PIPELINE_GUIDE.md)**: Step-by-step pipeline explanation
- **[Methodology](METHODOLOGY.md)**: Different approaches and strategies
- **[Results](RESULTS.md)**: Complete experimental results
- **[Technical Details](TECHNICAL_DETAILS.md)**: Implementation details and fixes

---

## Key Concepts

### Visual Place Recognition (VPR)
- Task: Recognize location from query images
- Method: Global descriptor matching (e.g., CosPlace, NetVLAD)
- Metric: Recall@N (percentage of queries with correct match in top-N)

### Re-ranking with Image Matching
- Method: Local feature matching (SuperPoint + LightGlue)
- Process: Extract keypoints, match, filter with RANSAC
- Result: Number of inliers used to re-rank candidates
- Cost: ~9.5 seconds per query (expensive!)

### Adaptive Re-ranking
- Goal: Apply re-ranking only to "hard" queries
- Strategy: Predict query difficulty before image matching
- Benefit: Save computation time while maintaining accuracy

### Hard vs Easy Queries
- **Easy**: Top-1 retrieval is correct (within 25m threshold)
- **Hard**: Top-1 retrieval is wrong (needs re-ranking)
- **Challenge**: Identify hard queries without running image matching

---

## Datasets

| Dataset | Split | Queries | Gallery | Baseline R@1 |
|---------|-------|---------|---------|--------------|
| **SF-XS** | Train | - | - | - |
| **SF-XS** | Val | 7,993 | - | - |
| **SF-XS** | Test | 1,000 | - | 63.1% |
| **Tokyo-XS** | Test | 315 | - | 65.1% |
| **SVOX** | Train | 1,414 | - | - |
| **SVOX** | Test | 14,278 | 17,166 | 96.3% |

---

## Main Features

### 8 Retrieval-Based Features (Available Before Image Matching)

1. **`top1_distance`**: Descriptor distance of Top-1 retrieved image
2. **`peakiness`**: Ratio of Top-1 to Top-2 descriptor distances
3. **`sue_score`**: Spatial Uncertainty Estimate from top-K neighbors
4. **`topk_distance_spread`**: Standard deviation of top-5 distances
5. **`top1_top2_similarity`**: Distance ratio (Top2/Top1)
6. **`top1_top3_ratio`**: Distance ratio (Top1/Top3)
7. **`top2_top3_ratio`**: Distance ratio (Top2/Top3)
8. **`geographic_clustering`**: Average pairwise distance of top-K positions

---

*Last updated: 2025-12-18*

