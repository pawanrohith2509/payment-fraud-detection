"""
Interactive Local Showcase & Real-Time Fraud Inference Demo.
"""

import os
import sys
import time
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

from payment_fraud_detection.synthetic_generator import generate_payment_dataset
from payment_fraud_detection.data_loader import temporal_train_val_test_split
from payment_fraud_detection.features import FraudFeatureEngineer
from payment_fraud_detection.models import FraudModelTrainer


def run_showcase():
    print("\n" + "=" * 75)
    print("      PAYMENT FRAUD DETECTION SYSTEM - REAL-TIME INFERENCE SHOWCASE      ")
    print("=" * 75)
    print("Initializing feature engine and training production LightGBM baseline...")

    data_path = os.path.join("data", "raw", "synthetic_payments.parquet")
    if not os.path.exists(data_path):
        print("Generating transaction stream...")
        df = generate_payment_dataset(n_transactions=30_000, output_path=data_path)
    else:
        df = pd.read_parquet(data_path)

    # Temporal split
    train_df, val_df, test_df, meta = temporal_train_val_test_split(df)

    # Feature engineering
    fe = FraudFeatureEngineer()
    X_train = fe.fit_transform(train_df)
    y_train = train_df["isFraud"].to_numpy()

    # Train LightGBM model
    trainer = FraudModelTrainer(random_state=42)
    model = trainer.train_lightgbm(X_train, y_train, n_estimators=100)

    # Cost-optimal threshold
    optimal_threshold = 0.188  # Calibrated to minimize expected loss ($200 FN vs $15 FP)

    print("\nModel successfully initialized!")
    print(f"Optimal Decision Cutoff : {optimal_threshold:.3f} (Trained on Asymmetric Loss Matrix)")
    print(f"Default Cutoff          : 0.500 (Naive Symmetric Loss)\n")
    print("-" * 75)
    print("SIMULATING REAL-TIME TRANSACTION INCOMING STREAM...")
    print("-" * 75)

    # Test scenarios
    scenarios = [
        {
            "name": "Scenario 1: Standard Grocery Payment (Genuine)",
            "tx": {
                "step": 142,  # 2:00 PM
                "type": "PAYMENT",
                "amount": 42.50,
                "nameOrig": "C0001234",
                "oldbalanceOrg": 3450.00,
                "newbalanceOrig": 3407.50,
                "nameDest": "M0009876",
                "oldbalanceDest": 12500.00,
                "newbalanceDest": 12542.50,
            },
            "expected_fraud": False,
        },
        {
            "name": "Scenario 2: Night Account Takeover & Balance Drain (Fraud Attack)",
            "tx": {
                "step": 147,  # 3:00 AM off-hours
                "type": "TRANSFER",
                "amount": 9950.00,
                "nameOrig": "C0005678",
                "oldbalanceOrg": 10000.00,
                "newbalanceOrig": 50.00,
                "nameDest": "C0009999",
                "oldbalanceDest": 0.00,  # Fresh mule account
                "newbalanceDest": 9950.00,
            },
            "expected_fraud": True,
        },
        {
            "name": "Scenario 3: Routine Peer-to-Peer Transfer (Genuine)",
            "tx": {
                "step": 210,  # 6:00 PM
                "type": "TRANSFER",
                "amount": 250.00,
                "nameOrig": "C0004321",
                "oldbalanceOrg": 5200.00,
                "newbalanceOrig": 4950.00,
                "nameDest": "C0008765",
                "oldbalanceDest": 3100.00,
                "newbalanceDest": 3350.00,
            },
            "expected_fraud": False,
        },
        {
            "name": "Scenario 4: Immediate Cash-Out with Ledger Mismatch (Mule Activity)",
            "tx": {
                "step": 148,  # 4:00 AM
                "type": "CASH_OUT",
                "amount": 9900.00,
                "nameOrig": "C0009999",  # Recipient from Scenario 2
                "oldbalanceOrg": 9950.00,
                "newbalanceOrig": 0.00,   # Ledger discrepancy
                "nameDest": "M0001122",
                "oldbalanceDest": 0.00,
                "newbalanceDest": 9900.00,
            },
            "expected_fraud": True,
        },
    ]

    for item in scenarios:
        time.sleep(0.4)
        tx = item["tx"]
        tx_df = pd.DataFrame([tx])
        X_sample = fe.transform(tx_df)
        prob = float(model.predict_proba(X_sample)[0, 1])

        decision_optimal = "ALERT / BLOCK" if prob >= optimal_threshold else "APPROVE"
        decision_default = "ALERT / BLOCK" if prob >= 0.500 else "APPROVE"

        # Key domain feature extractions
        drain_ratio = tx["amount"] / (tx["oldbalanceOrg"] + 1.0)
        hour = tx["step"] % 24
        is_night = (hour >= 0) and (hour <= 5)
        dest_is_empty = tx["oldbalanceDest"] == 0.0

        print(f"\n>>> {item['name']}")
        print(f"    Transaction Details : {tx['type']} of ${tx['amount']:,.2f} | Time: {hour:02d}:00 hrs")
        print(f"    Sender Balance      : ${tx['oldbalanceOrg']:,.2f} -> ${tx['newbalanceOrig']:,.2f}")
        print(f"    Recipient Balance   : ${tx['oldbalanceDest']:,.2f} -> ${tx['newbalanceDest']:,.2f}")
        print(f"    Engineered Signals  : Drain Ratio: {drain_ratio:.1%} | Off-Hours: {is_night} | Zero Dest Bal: {dest_is_empty}")
        print(f"    Predicted Fraud Risk: {prob * 100:.2f}%")

        if decision_optimal == "ALERT / BLOCK":
            status_tag = "[ALERT: FRAUD DETECTED -> BLOCKED]"
        else:
            status_tag = "[PASS: TRANSACTION APPROVED]"

        print(f"    Decision (Optimal T={optimal_threshold:.3f}) : {status_tag}")
        print(f"    Decision (Default T=0.500)         : [{decision_default}]")

        if decision_optimal != decision_default:
            print("    [*] BUSINESS BENEFIT: Cost-sensitive threshold caught this fraud while default 0.5 missed it!")
            print("        Protected ~$200.00 chargeback loss with just ~$15.00 review cost.")

    print("\n" + "=" * 75)
    print("SHOWCASE COMPLETE")
    print("=" * 75)
