import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib


def main():
    parser = argparse.ArgumentParser(
        description="Train Logistic Regression model from CSV dataset."
    )
    parser.add_argument(
        "--train-csv",
        type=str,
        required=True,
        help="Path to training CSV file (e.g., lr_data_cosplace_loftr_svox_train.csv).",
    )
    parser.add_argument(
        "--val-csv",
        type=str,
        required=False,
        help="Optional path to validation CSV file.",
    )
    parser.add_argument(
        "--output-model",
        type=str,
        default="logreg_model.pkl",
        help="Path to save the trained model (will save model + scaler).",
    )
    parser.add_argument(
        "--feature-col",
        type=str,
        default="inliers_top1",
        help="Name of the feature column in CSV.",
    )
    parser.add_argument(
        "--label-col",
        type=str,
        default="is_top1_correct",
        help="Name of the label column in CSV.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Regularization strength (inverse of regularization). Smaller C = stronger regularization.",
    )

    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from {args.train_csv}...")
    train_df = pd.read_csv(args.train_csv)
    
    # Extract features and labels
    X_train = train_df[[args.feature_col]].values.astype(np.float32)
    y_train = train_df[args.label_col].values.astype(np.int32)
    
    print(f"Training set: {len(X_train)} samples")
    print(f"  Positive samples (correct): {y_train.sum()}")
    print(f"  Negative samples (wrong): {len(y_train) - y_train.sum()}")
    print(f"  Feature range: [{X_train.min():.1f}, {X_train.max():.1f}]")

    # Normalize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Train logistic regression
    print("\nTraining Logistic Regression model...")
    # C=1.0 is default (L2 regularization)
    # Smaller C = stronger regularization (prevents overfitting)
    # Larger C = weaker regularization (fits training data more closely)
    logreg = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
        C=args.C,  # Regularization strength (from command line)
    )
    logreg.fit(X_train_scaled, y_train)
    
    # Training accuracy
    y_train_pred = logreg.predict(X_train_scaled)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"Training accuracy: {train_acc:.4f}")

    # Optional validation
    if args.val_csv:
        print(f"\nLoading validation data from {args.val_csv}...")
        val_df = pd.read_csv(args.val_csv)
        X_val = val_df[[args.feature_col]].values.astype(np.float32)
        y_val = val_df[args.label_col].values.astype(np.int32)
        
        X_val_scaled = scaler.transform(X_val)
        y_val_pred = logreg.predict(X_val_scaled)
        y_val_proba = logreg.predict_proba(X_val_scaled)[:, 1]
        
        val_acc = accuracy_score(y_val, y_val_pred)
        print(f"Validation accuracy: {val_acc:.4f}")
        
        try:
            val_auc = roc_auc_score(y_val, y_val_proba)
            print(f"Validation ROC-AUC: {val_auc:.4f}")
        except ValueError:
            print("Could not compute ROC-AUC (perhaps only one class present).")
        
        print("\nValidation Classification Report:")
        print(classification_report(y_val, y_val_pred, target_names=["Wrong", "Correct"]))

    # Save model and scaler
    output_path = Path(args.output_model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    model_bundle = {
        "scaler": scaler,
        "model": logreg,
        "feature_col": args.feature_col,
    }
    joblib.dump(model_bundle, output_path)
    print(f"\nSaved model to {output_path}")

    # Print model coefficients for interpretability
    print(f"\nModel coefficients:")
    print(f"  Intercept: {logreg.intercept_[0]:.4f}")
    print(f"  Coefficient for {args.feature_col}: {logreg.coef_[0][0]:.4f}")


if __name__ == "__main__":
    main()

