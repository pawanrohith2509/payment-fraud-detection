"""
Feature engineering pipeline for fraud detection.

Implements domain-specific payment risk features:
1. Balance Discrepancy / Math Errors (discrepancy between balance before and after)
2. Balance Drain Indicators (ratio of transfer amount to remaining balance)
3. Diurnal & Time-of-day Patterns (night transactions, velocity)
4. Account Velocity (rolling activity counts and sums within temporal windows)
5. Categorical encodings for transaction channels / types
"""

from typing import List, Tuple, Optional
import numpy as np
import pandas as pd


class FraudFeatureEngineer:
    """
    Stateless and stateful feature engineering for payment transaction logs.
    """

    def __init__(self, high_risk_types: Optional[List[str]] = None):
        self.high_risk_types = high_risk_types or ["TRANSFER", "CASH_OUT"]
        self.feature_names: List[str] = []

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fit on training split and transform."""
        return self.transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate engineered features from raw transaction logs.
        Preserves original DataFrame and returns an enriched feature DataFrame.
        """
        feats = pd.DataFrame(index=df.index)

        # 1. Financial Amount Transformations
        feats["amount"] = df["amount"].astype(float)
        feats["log_amount"] = np.log1p(df["amount"].clip(lower=0))

        # 2. Origin Balance Signals & Ratios
        old_orig = df["oldbalanceOrg"].astype(float)
        new_orig = df["newbalanceOrig"].astype(float)
        feats["oldbalanceOrg"] = old_orig
        feats["newbalanceOrig"] = new_orig
        feats["log_oldbalanceOrg"] = np.log1p(old_orig.clip(lower=0))

        # Ratio of amount to sender balance (1.0 = full drain)
        feats["amount_to_orig_balance_ratio"] = feats["amount"] / (old_orig + 1.0)
        feats["is_exact_drain"] = ((old_orig > 0) & (np.abs(old_orig - feats["amount"]) < 0.01)).astype(int)

        # Balance calculation error (in genuine payments, old - amount ~ new)
        feats["orig_balance_error"] = (old_orig - feats["amount"]) - new_orig

        # 3. Destination Balance Signals
        old_dest = df["oldbalanceDest"].astype(float)
        new_dest = df["newbalanceDest"].astype(float)
        feats["oldbalanceDest"] = old_dest
        feats["newbalanceDest"] = new_dest
        feats["is_zero_dest_balance_before"] = (old_dest == 0).astype(int)
        feats["dest_balance_error"] = (old_dest + feats["amount"]) - new_dest

        # 4. Temporal & Cyclic Signals
        step = df["step"].astype(int)
        feats["hour_of_day"] = step % 24
        feats["day_of_week"] = (step // 24) % 7
        feats["is_night"] = ((feats["hour_of_day"] >= 0) & (feats["hour_of_day"] <= 5)).astype(int)

        # 5. High-Risk Transaction Channel Indicators
        if "type" in df.columns:
            for t in ["TRANSFER", "CASH_OUT", "PAYMENT", "CASH_IN", "DEBIT"]:
                feats[f"type_{t}"] = (df["type"] == t).astype(int)
            feats["is_high_risk_type"] = df["type"].isin(self.high_risk_types).astype(int)

        # 6. Entity Velocity Signals (Fast group-by within chronological dataset)
        # Transactions per sender in the past
        if "nameOrig" in df.columns:
            orig_tx_count = df.groupby("nameOrig").cumcount()
            feats["sender_cumulative_tx_count"] = orig_tx_count

        self.feature_names = list(feats.columns)
        return feats

    def get_feature_names(self) -> List[str]:
        return self.feature_names
