"""
Data Loading and Leak-Free Temporal Train/Validation/Test Splitting.

Key Interview Concept:
In payment fraud, standard random k-fold cross-validation or train_test_split causes
catastrophic data leakage because:
1. Future fraud campaigns leak into historical training data.
2. Account-level behavioral patterns are seen ahead of time.
3. Realistic temporal concept drift is hidden.

A production-grade pipeline enforces strict Out-Of-Time (OOT) validation splits.
"""

from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd


def temporal_train_val_test_split(
    df: pd.DataFrame,
    time_col: str = "step",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
    """
    Split transaction dataset chronologically into train, validation, and test sets.
    
    Args:
        df: Input DataFrame containing transactions
        time_col: Column representing time sequence (e.g. step or timestamp)
        train_ratio: Proportion of time window allocated to training (e.g. 0.70)
        val_ratio: Proportion of time window allocated to validation (e.g. 0.15)
        
    Returns:
        Tuple of (train_df, val_df, test_df, metadata_dict)
    """
    # Ensure chronological ordering
    df_sorted = df.sort_values(by=time_col).reset_index(drop=True)
    
    min_time = df_sorted[time_col].min()
    max_time = df_sorted[time_col].max()
    time_range = max_time - min_time
    
    train_cutoff = min_time + (time_range * train_ratio)
    val_cutoff = min_time + (time_range * (train_ratio + val_ratio))
    
    train_df = df_sorted[df_sorted[time_col] <= train_cutoff].copy().reset_index(drop=True)
    val_df = df_sorted[(df_sorted[time_col] > train_cutoff) & (df_sorted[time_col] <= val_cutoff)].copy().reset_index(drop=True)
    test_df = df_sorted[df_sorted[time_col] > val_cutoff].copy().reset_index(drop=True)
    
    metadata = {
        "time_col": time_col,
        "total_records": len(df_sorted),
        "min_time": float(min_time),
        "max_time": float(max_time),
        "train_cutoff": float(train_cutoff),
        "val_cutoff": float(val_cutoff),
        "train_records": len(train_df),
        "val_records": len(val_df),
        "test_records": len(test_df),
        "train_fraud_rate": float(train_df["isFraud"].mean()) if "isFraud" in train_df else None,
        "val_fraud_rate": float(val_df["isFraud"].mean()) if "isFraud" in val_df else None,
        "test_fraud_rate": float(test_df["isFraud"].mean()) if "isFraud" in test_df else None,
    }
    
    return train_df, val_df, test_df, metadata


def compute_dataset_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate key fraud dataset distributions and statistics."""
    summary = {
        "num_transactions": len(df),
        "num_fraud": int(df["isFraud"].sum()) if "isFraud" in df else 0,
        "fraud_prevalence_pct": float(df["isFraud"].mean() * 100) if "isFraud" in df else 0.0,
        "total_volume": float(df["amount"].sum()) if "amount" in df else 0.0,
        "mean_amount": float(df["amount"].mean()) if "amount" in df else 0.0,
        "median_amount": float(df["amount"].median()) if "amount" in df else 0.0,
        "transaction_types": df["type"].value_counts().to_dict() if "type" in df else {},
    }
    if "isFraud" in df and "type" in df:
        summary["fraud_by_type"] = (
            df.groupby("type")["isFraud"].agg(["count", "sum", "mean"]).to_dict(orient="index")
        )
    return summary
