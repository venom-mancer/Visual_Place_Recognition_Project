# Extension 6.1 – Dataset Usage

## Summary

Based on `pdf.txt` Section 6.1 (lines 90-106), here is the dataset usage for Extension 6.1:

---

## **Training Dataset**

### **SVOX Train** (Sun and Night subsets)
- **Purpose**: Train the logistic regression model
- **Note**: Exclude GSV-XS from training (GSV-XS is used for general VPR model training, but NOT for Extension 6.1)
- **Location**: `data/svox/images/train/`
  - **Database/Gallery**: `data/svox/images/train/gallery/` (22,232 images - SVOX dataset)
  - **Queries**: Use **Sun and Night subsets only**:
    - `data/svox/images/train/queries_sun/` (712 images - RobotCar dataset)
    - `data/svox/images/train/queries_night/` (702 images - RobotCar dataset)
    - **Total**: 1,414 queries (712 + 702)
  
**Important**: The VPR evaluation script accepts only one `--queries_folder`. You need to combine `queries_sun/` and `queries_night/` into one folder.

**How to combine (Windows PowerShell)**:
```powershell
# Create combined folder
New-Item -ItemType Directory -Path "data/svox/images/train/queries_sun_night" -Force

# Copy sun queries
Copy-Item "data/svox/images/train/queries_sun/*" -Destination "data/svox/images/train/queries_sun_night/"

# Copy night queries
Copy-Item "data/svox/images/train/queries_night/*" -Destination "data/svox/images/train/queries_sun_night/"
```

**How to combine (Linux/Git Bash)**:
```bash
# Create combined folder
mkdir -p data/svox/images/train/queries_sun_night

# Copy both folders
cp data/svox/images/train/queries_sun/* data/svox/images/train/queries_sun_night/
cp data/svox/images/train/queries_night/* data/svox/images/train/queries_sun_night/
```

Then use `--queries_folder data/svox/images/train/queries_sun_night` in your VPR evaluation command.

**Why SVOX Train?**
- SVOX provides a robust cross-domain VPR dataset
- The Sun and Night subsets offer diverse conditions for learning query difficulty
- Explicitly excludes GSV-XS as per Extension 6.1 requirements

---

## **Validation Dataset**

### **SF-XS Validation**
- **Purpose**: Validate hyperparameter selections (e.g., logistic regression threshold)
- **Location**: `data/sf_xs/val/`
  - Database: `data/sf_xs/val/database/` (8,015 images)
  - Queries: `data/sf_xs/val/queries/` (6,420 images)

**Why SF-XS Val?**
- Standard validation set for the project
- Used to tune the threshold for easy/hard query classification
- Helps balance accuracy vs. cost (how many queries to re-rank)

**Note**: SF-XS also has a training set, but it is **NOT used** in this project (ignore it).

---

## **Test Datasets**

Evaluate on **ALL** test sets:

### 1. **SF-XS Test**
- **Location**: `data/sf_xs/test/`
  - Database: `data/sf_xs/test/database/` (4,720 images)
  - Queries: `data/sf_xs/test/queries/` (1,000 images)

### 2. **Tokyo-XS Test**
- **Purpose**: Testing only (no training/validation)
- **Location**: `data/tokyo_xs/test/`
  - Database: `data/tokyo_xs/test/database/` (12,771 images)
  - Queries: `data/tokyo_xs/test/queries/` (315 images)

### 3. **SVOX Test** (Sun and Night subsets)
- **Location**: `data/svox/images/test/`
  - **Database/Gallery**: `data/svox/images/test/gallery/` (17,166 images - SVOX dataset)
  - **Queries**: Use **Sun and Night subsets only**:
    - `data/svox/images/test/queries_sun/` (854 images - RobotCar dataset)
    - `data/svox/images/test/queries_night/` (823 images - RobotCar dataset)
    - **Total**: 1,677 queries (854 + 823)
  
**Important**: Same as training - combine `queries_sun/` and `queries_night/` into one folder (see training section above for commands)

---

## **Excluded Datasets**

### **GSV-XS**
- **Status**: **EXCLUDED** from Extension 6.1 training
- **Reason**: Explicitly stated in Extension 6.1 requirements: "use only the training sets–excluding GSV-XS"
- **Note**: GSV-XS is used for general VPR model training (e.g., CosPlace), but NOT for training the logistic regression in Extension 6.1

---

## **Pipeline Summary**

```
┌─────────────────────────────────────────────────────────────┐
│ Extension 6.1 Pipeline                                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ 1. TRAINING (Stage 1-3)                                     │
│    └─ SVOX Train → Extract features → Train logistic reg.  │
│                                                             │
│ 2. VALIDATION (Stage 4)                                     │
│    └─ SF-XS Val → Apply logreg → Tune threshold            │
│                                                             │
│ 3. TESTING (Stage 5)                                        │
│    ├─ SF-XS Test → Adaptive re-ranking eval                │
│    ├─ Tokyo-XS Test → Adaptive re-ranking eval             │
│    └─ SVOX Test → Adaptive re-ranking eval                 │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## **Reference from pdf.txt**

> **Section 6.1 (lines 104-106):**
> "For training the logistic regressor or for threshold selection, use only the training sets–excluding GSV-XS. Use the validation sets to validate your hyperparameter selections. Then, evaluate on all the test sets."

> **Section 4 (lines 38-51):**
> - GSV-XS: used for training (general VPR, excluded for Extension 6.1)
> - SF-XS: validation (SF-XS val) and testing (SF-XS test)
> - Tokyo-XS: used only for testing
> - SVOX: Sun and Night subsets for training and testing

---

## **Next Steps**

1. **Run VPR evaluation** (CosPlace) on:
   - SVOX train
   - SF-XS val
   - SF-XS test
   - Tokyo-XS test
   - SVOX test

2. **Run image matching** (SuperPoint + LightGlue) on all predictions

3. **Extract features** for each split

4. **Train logistic regression** on SVOX train features

5. **Validate** on SF-XS val

6. **Test** on all test sets

