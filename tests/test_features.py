"""
Unit tests for domain feature engineering.
"""

import pandas as pd
import numpy as np
import pytest
from payment_fraud_detection.features import FraudFeatureEngineer


@pytest.fixture
def sample_transactions():
    return pd.DataFrame([
        {
            "step": 10,
            "type": "TRANSFER",
            "amount": 1000.0,
            "nameOrig": "C1001",
            "oldbalanceOrg": 1000.0,
            "newbalanceOrig": 0.0,
            "nameDest": "C2001",
            "oldbalanceDest": 0.0,
            "newbalanceDest": 1000.0,
            "isFraud": 1,
        },
        {
            "step": 36,  # 36 % 24 = 12 (noon)
            "type": "PAYMENT",
            "amount": 50.0,
            "nameOrig": "C1002",
            "oldbalanceOrg": 500.0,
            "newbalanceOrig": 450.0,
            "nameDest": "M3001",
            "oldbalanceDest": 1000.0,
            "newbalanceDest": 1050.0,
            "isFraud": 0,
        },
    ])


def test_feature_engineering_columns(sample_transactions):
    fe = FraudFeatureEngineer()
    X = fe.fit_transform(sample_transactions)
    
    expected_cols = [
        "amount",
        "log_amount",
        "amount_to_orig_balance_ratio",
        "is_exact_drain",
        "orig_balance_error",
        "dest_balance_error",
        "is_zero_dest_balance_before",
        "hour_of_day",
        "is_night",
        "is_high_risk_type",
    ]
    for col in expected_cols:
        assert col in X.columns, f"Missing feature: {col}"


def test_exact_drain_detection(sample_transactions):
    fe = FraudFeatureEngineer()
    X = fe.fit_transform(sample_transactions)
    
    # First row is an exact drain of 1000 out of 1000
    assert X.iloc[0]["is_exact_drain"] == 1
    # Second row is not an exact drain
    assert X.iloc[1]["is_exact_drain"] == 0


def test_diurnal_flags(sample_transactions):
    fe = FraudFeatureEngineer()
    X = fe.fit_transform(sample_transactions)
    
    # Step 10: hour 10 (not night: 0-5)
    assert X.iloc[0]["is_night"] == 0
    # Step 36: hour 12 (not night)
    assert X.iloc[1]["hour_of_day"] == 12
