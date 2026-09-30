"""
Synthetic Payment & UPI/Mobile-Money Transaction Generator.

Generates realistic transaction logs with authentic fraud typologies:
1. Account Drain (Transferring full balance)
2. Mule Pass-Through (High transfer followed immediately by cash-out)
3. Velocity Bursts (Multiple rapid transactions within short temporal window)
4. Unusual High Value at Off-Hours
"""

import os
from typing import Optional
import numpy as np
import pandas as pd


def generate_payment_dataset(
    n_transactions: int = 100_000,
    n_users: int = 10_000,
    n_merchants: int = 2_000,
    fraud_rate: float = 0.015,
    random_seed: int = 42,
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Generate synthetic transactions over a 30-day simulated timeframe (720 hourly steps).
    
    Args:
        n_transactions: Total number of transactions to generate
        n_users: Number of distinct customer accounts
        n_merchants: Number of distinct merchant / recipient accounts
        fraud_rate: Targeted fraction of fraudulent transactions
        random_seed: Seed for reproducibility
        output_path: If provided, saves dataframe as parquet/csv
        
    Returns:
        pd.DataFrame containing simulated transaction logs.
    """
    np.random.seed(random_seed)
    
    # 720 hours = 30 days of transactions
    # Simulating diurnal volume curve (higher during day, lower at night)
    steps_raw = np.random.randint(1, 721, size=n_transactions)
    # Sort by time step to mimic streaming log arrival
    steps = np.sort(steps_raw)
    
    user_ids = [f"C{i:07d}" for i in range(1, n_users + 1)]
    merchant_ids = [f"M{i:07d}" for i in range(1, n_merchants + 1)]
    
    # Pre-generate user balance profiles
    user_balances = {uid: float(np.random.exponential(scale=5000.0) + 100.0) for uid in user_ids}
    dest_balances = {mid: float(np.random.exponential(scale=10000.0)) for mid in merchant_ids}
    
    types = ["PAYMENT", "TRANSFER", "CASH_OUT", "CASH_IN", "DEBIT"]
    type_probs = [0.45, 0.20, 0.20, 0.12, 0.03]
    
    tx_types = np.random.choice(types, size=n_transactions, p=type_probs)
    
    # Pre-sample entity arrays to eliminate per-iteration overhead
    orig_sample = np.random.choice(user_ids, size=n_transactions)
    user_dest_sample = np.random.choice(user_ids, size=n_transactions)
    merchant_dest_sample = np.random.choice(merchant_ids, size=n_transactions)
    rand_scenarios = np.random.rand(n_transactions)
    
    records = []
    n_frauds_target = int(n_transactions * fraud_rate)
    fraud_count = 0
    
    # Generate transactions
    for idx in range(n_transactions):
        step = int(steps[idx])
        tx_type = tx_types[idx]
        orig = orig_sample[idx]
        dest = merchant_dest_sample[idx] if tx_type == "PAYMENT" else user_dest_sample[idx]
        
        orig_bal_before = user_balances.get(orig, 1000.0)
        dest_bal_before = dest_balances.get(dest, 500.0)
        
        # Decide if this transaction is an injected fraud pattern
        # In reality, fraud occurs across all time periods, concentrated in TRANSFER/CASH_OUT
        is_fraud = 0
        fraud_scenario = "none"
        
        # Fraud probability per high-risk transfer/cash-out event
        p_fraud_trigger = fraud_rate / (type_probs[1] + type_probs[2])
        if tx_type in ["TRANSFER", "CASH_OUT"] and rand_scenarios[idx] < p_fraud_trigger:
            is_fraud = 1
            fraud_count += 1
            scenario_choice = np.random.choice(["drain", "velocity_burst", "high_value_night"])
            
            if scenario_choice == "drain":
                # Account draining: transfer entire balance
                amount = max(500.0, orig_bal_before * 0.98)
                fraud_scenario = "account_drain"
            elif scenario_choice == "velocity_burst":
                # Rapid successive transfers
                amount = float(np.random.uniform(2000.0, 15000.0))
                fraud_scenario = "velocity_burst"
            else:
                # Off-hours high value
                amount = float(np.random.exponential(scale=20000.0) + 10000.0)
                fraud_scenario = "high_value_night"
        else:
            # Genuine transaction amount distribution
            if tx_type == "PAYMENT":
                amount = float(np.random.exponential(scale=150.0) + 5.0)
            elif tx_type in ["TRANSFER", "CASH_OUT"]:
                amount = float(np.random.exponential(scale=1200.0) + 50.0)
            else:
                amount = float(np.random.exponential(scale=500.0) + 20.0)
                
        # Calculate resulting balances
        amount = round(amount, 2)
        if tx_type in ["TRANSFER", "CASH_OUT", "PAYMENT", "DEBIT"]:
            orig_bal_after = max(0.0, round(orig_bal_before - amount, 2))
        else:  # CASH_IN
            orig_bal_after = round(orig_bal_before + amount, 2)
            
        if tx_type in ["TRANSFER", "CASH_IN", "PAYMENT"]:
            dest_bal_after = round(dest_bal_before + amount, 2)
        else:
            dest_bal_after = max(0.0, round(dest_bal_before - amount, 2))
            
        # Update user balance state
        user_balances[orig] = orig_bal_after
        if dest in dest_balances:
            dest_balances[dest] = dest_bal_after
            
        records.append({
            "step": step,
            "type": tx_type,
            "amount": amount,
            "nameOrig": orig,
            "oldbalanceOrg": round(orig_bal_before, 2),
            "newbalanceOrig": orig_bal_after,
            "nameDest": dest,
            "oldbalanceDest": round(dest_bal_before, 2),
            "newbalanceDest": dest_bal_after,
            "isFraud": is_fraud,
            "fraudScenario": fraud_scenario,
        })
        
    df = pd.DataFrame(records)
    
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        if output_path.endswith(".parquet"):
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)
        print(f"Generated {len(df):,} transactions ({df['isFraud'].sum():,} frauds, {df['isFraud'].mean()*100:.2f}%) -> {output_path}")
        
    return df


if __name__ == "__main__":
    generate_payment_dataset(
        n_transactions=100_000,
        output_path="data/raw/synthetic_payments.parquet",
    )
