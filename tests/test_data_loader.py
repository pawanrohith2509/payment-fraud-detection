"""
Unit tests for data loading and temporal out-of-time splitting.
"""

import pandas as pd
import numpy as np
import pytest
from payment_fraud_detection.data_loader import temporal_train_val_test_split, compute_dataset_summary


@pytest.fixture
def chronological_dataset():
    # 100 rows across steps 1 to 100
    return pd.DataFrame({
        "step": list(range(1, 101)),
        "amount": [100.0] * 100,
        "isFraud": [0] * 90 + [1] * 10,
        "type": ["PAYMENT"] * 50 + ["TRANSFER"] * 50,
    })


def test_temporal_split_boundaries(chronological_dataset):
    train_df, val_df, test_df, meta = temporal_train_val_test_split(
        chronological_dataset, time_col="step", train_ratio=0.70, val_ratio=0.15
    )
    
    # Check that lengths sum up to total records
    assert len(train_df) + len(val_df) + len(test_df) == len(chronological_dataset)
    
    # Check strict chronological order and non-overlapping time boundaries
    assert train_df["step"].max() <= meta["train_cutoff"]
    assert val_df["step"].min() > meta["train_cutoff"]
    assert val_df["step"].max() <= meta["val_cutoff"]
    assert test_df["step"].min() > meta["val_cutoff"]


def test_dataset_summary(chronological_dataset):
    summary = compute_dataset_summary(chronological_dataset)
    assert summary["num_transactions"] == 100
    assert summary["num_fraud"] == 10
    assert summary["fraud_prevalence_pct"] == 10.0
