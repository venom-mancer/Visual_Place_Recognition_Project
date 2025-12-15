## Extension 6.1 – Execution Plan (Step-by-step)

You will run **11 Python commands** (after installing requirements).  
We will execute them **one by one**: I’ll give you a command, you run it, then tell me when it finishes so we move to the next.

All commands are assumed to be run from the project root:

```text
D:\regres\Visual_Place_Recognition_Project
```

### 0. One-time setup (before commands 1–11)

Install dependencies:

```bash
pip install -r requirements.txt
pip install scikit-learn joblib
```

---

### 1) Train logistic regression on SVOX train

**Command 1/11 – Extract SVOX train features**

```bash
python -m vpr_uncertainty.extract_features --preds-dir PATH_SVOX_TRAIN_PREDS --inliers-dir PATH_SVOX_TRAIN_INLIERS --z-data-path PATH_SVOX_TRAIN_Z_DATA --output-path features_svox_train.npz
```

**Command 2/11 – Train logistic regression**

```bash
python -m vpr_uncertainty.train_logreg --train-features features_svox_train.npz --output-model logreg_svox.pkl
```

---

### 2) SF-XS test (features + logreg + adaptive eval)

**Command 3/11 – Extract SF-XS test features**

```bash
python -m vpr_uncertainty.extract_features --preds-dir PATH_SF_TEST_PREDS --inliers-dir PATH_SF_TEST_INLIERS --z-data-path PATH_SF_TEST_Z_DATA --output-path features_sf_xs_test.npz
```

**Command 4/11 – Apply logistic regression on SF-XS test**

```bash
python -m vpr_uncertainty.apply_logreg --model-path logreg_svox.pkl --feature-path features_sf_xs_test.npz --output-path logreg_sf_xs_test_outputs.npz
``>

**Command 5/11 – Adaptive re-ranking eval on SF-XS test**

```bash
python -m vpr_uncertainty.adaptive_reranking_eval --preds-dir PATH_SF_TEST_PREDS --inliers-dir PATH_SF_TEST_INLIERS --logreg-output logreg_sf_xs_test_outputs.npz --num-preds 100 --positive-dist-threshold 25 --recall-values 1 5 10 20 100
```

---

### 3) Tokyo-XS test

**Command 6/11 – Extract Tokyo-XS test features**

```bash
python -m vpr_uncertainty.extract_features --preds-dir PATH_TOKYO_TEST_PREDS --inliers-dir PATH_TOKYO_TEST_INLIERS --z-data-path PATH_TOKYO_TEST_Z_DATA --output-path features_tokyo_xs_test.npz
```

**Command 7/11 – Apply logistic regression on Tokyo-XS test**

```bash
python -m vpr_uncertainty.apply_logreg --model-path logreg_svox.pkl --feature-path features_tokyo_xs_test.npz --output-path logreg_tokyo_xs_test_outputs.npz
```

**Command 8/11 – Adaptive re-ranking eval on Tokyo-XS test**

```bash
python -m vpr_uncertainty.adaptive_reranking_eval --preds-dir PATH_TOKYO_TEST_PREDS --inliers-dir PATH_TOKYO_TEST_INLIERS --logreg-output logreg_tokyo_xs_test_outputs.npz --num-preds 100 --positive-dist-threshold 25 --recall-values 1 5 10 20 100
```

---

### 4) SVOX test

**Command 9/11 – Extract SVOX test features**

```bash
python -m vpr_uncertainty.extract_features --preds-dir PATH_SVOX_TEST_PREDS --inliers-dir PATH_SVOX_TEST_INLIERS --z-data-path PATH_SVOX_TEST_Z_DATA --output-path features_svox_test.npz
```

**Command 10/11 – Apply logistic regression on SVOX test**

```bash
python -m vpr_uncertainty.apply_logreg --model-path logreg_svox.pkl --feature-path features_svox_test.npz --output-path logreg_svox_test_outputs.npz
```

**Command 11/11 – Adaptive re-ranking eval on SVOX test**

```bash
python -m vpr_uncertainty.adaptive_reranking_eval --preds-dir PATH_SVOX_TEST_PREDS --inliers-dir PATH_SVOX_TEST_INLIERS --logreg-output logreg_svox_test_outputs.npz --num-preds 100 --positive-dist-threshold 25 --recall-values 1 5 10 20 100
```


