"""
End-to-end execution script for Milestone 1: Fraud Detection Baseline.

Performs:
1. Data generation / loading
2. Strict temporal out-of-time train/val/test split
3. Domain feature engineering (velocity, balance discrepancies, cyclic time)
4. Model training (Logistic Regression, Random Forest, LightGBM, XGBoost)
5. Business cost optimization and PR-AUC benchmark comparison
"""

import os
import sys
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pandas as pd
import numpy as np

from payment_fraud_detection.synthetic_generator import generate_payment_dataset
from payment_fraud_detection.data_loader import temporal_train_val_test_split, compute_dataset_summary
from payment_fraud_detection.features import FraudFeatureEngineer
from payment_fraud_detection.models import FraudModelTrainer


def main():
    print("=" * 70)
    print("PAYMENT FRAUD DETECTION: BASELINE & COST-SENSITIVE BENCHMARK")
    print("=" * 70)

    raw_data_path = "data/raw/synthetic_payments.parquet"
    results_path = "data/processed/benchmark_results.csv"

    # 1. Dataset Generation / Loading
    if not os.path.exists(raw_data_path):
        print("\n[Step 1] Generating synthetic transaction stream...")
        df = generate_payment_dataset(
            n_transactions=50_000,
            n_users=5_000,
            n_merchants=1_000,
            fraud_rate=0.015,
            random_seed=42,
            output_path=raw_data_path,
        )
    else:
        print(f"\n[Step 1] Loading transactions from {raw_data_path}...")
        df = pd.read_parquet(raw_data_path)

    summary = compute_dataset_summary(df)
    print(f"Total Transactions : {summary['num_transactions']:,}")
    print(f"Total Fraud Cases  : {summary['num_fraud']:,} ({summary['fraud_prevalence_pct']:.2f}%)")
    print(f"Total Volume       : ${summary['total_volume']:,.2f}")

    # 2. Strict Temporal Out-of-Time Split
    print("\n[Step 2] Executing strict temporal train/val/test split (no future leakage)...")
    train_df, val_df, test_df, meta = temporal_train_val_test_split(
        df, time_col="step", train_ratio=0.70, val_ratio=0.15
    )
    print(f"  Training Split   (Steps <= {meta['train_cutoff']:.1f}): {len(train_df):,} rows (Fraud: {meta['train_fraud_rate']*100:.2f}%)")
    print(f"  Validation Split (Steps {meta['train_cutoff']:.1f} - {meta['val_cutoff']:.1f}): {len(val_df):,} rows (Fraud: {meta['val_fraud_rate']*100:.2f}%)")
    print(f"  Out-of-Time Test (Steps > {meta['val_cutoff']:.1f}): {len(test_df):,} rows (Fraud: {meta['test_fraud_rate']*100:.2f}%)")

    # 3. Feature Engineering
    print("\n[Step 3] Engineering risk features (balance error, drain ratio, velocity, cyclic)...")
    fe = FraudFeatureEngineer()
    X_train = fe.fit_transform(train_df)
    X_val = fe.transform(val_df)
    X_test = fe.transform(test_df)

    y_train = train_df["isFraud"].to_numpy()
    y_val = val_df["isFraud"].to_numpy()
    y_test = test_df["isFraud"].to_numpy()

    print(f"  Total engineered features: {len(fe.get_feature_names())}")
    print(f"  Features: {', '.join(fe.get_feature_names()[:8])}...")

    # 4. Model Training
    print("\n[Step 4] Training benchmark models...")
    trainer = FraudModelTrainer(random_state=42)

    print("  -> Training Baseline 1: Logistic Regression (balanced weights)...")
    trainer.train_logistic_regression(X_train, y_train)

    print("  -> Training Baseline 2: Random Forest (balanced subsample)...")
    trainer.train_random_forest(X_train, y_train, n_estimators=80, max_depth=10)

    print("  -> Training Model 3: LightGBM (scale_pos_weight)...")
    trainer.train_lightgbm(X_train, y_train, X_val=X_val, y_val=y_val, n_estimators=150)

    print("  -> Training Model 4: XGBoost (scale_pos_weight)...")
    trainer.train_xgboost(X_train, y_train, n_estimators=150, max_depth=5)

    # 5. Comprehensive Out-of-Time Evaluation
    print("\n[Step 5] Evaluating on Out-Of-Time test set...")
    cost_fn = 200.0  # Loss of missed fraud
    cost_fp = 15.0   # Cost of alert review
    results_df = trainer.evaluate_all(X_test, y_test, cost_fn=cost_fn, cost_fp=cost_fp)

    os.makedirs(os.path.dirname(os.path.abspath(results_path)), exist_ok=True)
    results_df.to_csv(results_path, index=False)

    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (Out-of-Time Test Set)")
    print("=" * 70)
    cols_display = [
        "model_name",
        "pr_auc",
        "roc_auc",
        "recall_at_90_precision",
        "optimal_threshold",
        "optimal_total_cost",
        "pct_savings",
    ]
    print(results_df[cols_display].to_string(index=False))
    print("=" * 70)
    print(f"Results saved to: {results_path}\n")


if __name__ == "__main__":
    main()
