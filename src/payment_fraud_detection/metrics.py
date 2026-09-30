"""
Business-aligned and statistical evaluation metrics for imbalanced fraud detection.

In fraud detection, headline accuracy is meaningless due to extreme class imbalance
(e.g., 0.1% to 2% fraud rate). This module provides:
1. Precision-Recall AUC (PR-AUC / Average Precision)
2. ROC-AUC
3. Recall at fixed precision (e.g., Recall @ 90% Precision)
4. Cost-sensitive evaluation: Expected business loss optimization
   Cost = C_FN * False_Negatives + C_FP * False_Positives
"""

from typing import Dict, Any, Tuple
import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    brier_score_loss,
)


def compute_pr_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Precision-Recall Curve (Average Precision)."""
    return float(average_precision_score(y_true, y_prob))


def compute_roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Area Under the Receiver Operating Characteristic Curve."""
    return float(roc_auc_score(y_true, y_prob))


def compute_brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Compute Brier Score (lower is better, measures probability calibration)."""
    return float(brier_score_loss(y_true, y_prob))


def recall_at_min_precision(
    y_true: np.ndarray, y_prob: np.ndarray, min_precision: float = 0.90
) -> Tuple[float, float]:
    """
    Find maximum recall achievable while maintaining precision >= min_precision.
    
    Returns:
        Tuple of (achieved_recall, operating_threshold).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Filter thresholds where precision meets requirement
    valid_indices = np.where(precisions >= min_precision)[0]
    
    if len(valid_indices) == 0:
        return 0.0, 1.0
    
    # Select index with highest recall among valid points
    best_idx = valid_indices[np.argmax(recalls[valid_indices])]
    
    # precision_recall_curve returns thresholds of len(precisions) - 1
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]
    best_recall = float(recalls[best_idx])
    
    return best_recall, float(best_threshold)


def calculate_business_cost(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    cost_fn: float = 150.0,
    cost_fp: float = 10.0,
) -> Dict[str, Any]:
    """
    Calculate financial cost of model decisions at a specific decision threshold.
    
    Args:
        y_true: Ground truth binary labels (1=fraud, 0=legit)
        y_prob: Predicted fraud probabilities
        threshold: Classification cutoff [0, 1]
        cost_fn: Business loss of missing a fraudulent transaction (Chargeback / loss)
        cost_fp: Operational cost of investigating a legitimate transaction falsely flagged
    
    Returns:
        Dictionary containing total cost, cost breakdown, and confusion matrix components.
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    
    total_cost = (fn * cost_fn) + (fp * cost_fp)
    total_transactions = len(y_true)
    cost_per_tx = total_cost / total_transactions if total_transactions > 0 else 0.0
    
    return {
        "threshold": float(threshold),
        "total_cost": float(total_cost),
        "cost_per_tx": float(cost_per_tx),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0,
        "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0,
    }


def find_optimal_cost_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    cost_fn: float = 150.0,
    cost_fp: float = 10.0,
    num_steps: int = 100,
) -> Dict[str, Any]:
    """
    Find threshold that minimizes total business cost.
    
    Rather than defaulting to 0.5 (which assumes equal cost for false positives and false negatives),
    this searches for the cost-optimal decision point.
    """
    thresholds = np.linspace(0.01, 0.99, num_steps)
    best_result = None
    min_cost = float("inf")
    
    for thresh in thresholds:
        result = calculate_business_cost(y_true, y_prob, thresh, cost_fn, cost_fp)
        if result["total_cost"] < min_cost:
            min_cost = result["total_cost"]
            best_result = result
            
    # Also evaluate the default 0.5 threshold for comparison
    default_result = calculate_business_cost(y_true, y_prob, 0.5, cost_fn, cost_fp)
    
    savings = default_result["total_cost"] - best_result["total_cost"]
    pct_savings = (savings / default_result["total_cost"] * 100) if default_result["total_cost"] > 0 else 0.0
    
    return {
        "optimal": best_result,
        "default_0_5": default_result,
        "cost_savings": float(savings),
        "pct_cost_savings": float(pct_savings),
    }


def evaluate_model_comprehensive(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "Model",
    cost_fn: float = 150.0,
    cost_fp: float = 10.0,
) -> Dict[str, Any]:
    """
    Generate complete evaluation suite for fraud model reporting.
    """
    pr_auc = compute_pr_auc(y_true, y_prob)
    roc_auc = compute_roc_auc(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    rec_at_90p, thresh_90p = recall_at_min_precision(y_true, y_prob, min_precision=0.90)
    rec_at_95p, thresh_95p = recall_at_min_precision(y_true, y_prob, min_precision=0.95)
    cost_analysis = find_optimal_cost_threshold(y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp)
    
    return {
        "model_name": model_name,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "brier_score": brier,
        "recall_at_90_precision": rec_at_90p,
        "threshold_for_90_precision": thresh_90p,
        "recall_at_95_precision": rec_at_95p,
        "threshold_for_95_precision": thresh_95p,
        "optimal_threshold": cost_analysis["optimal"]["threshold"],
        "optimal_total_cost": cost_analysis["optimal"]["total_cost"],
        "default_0_5_cost": cost_analysis["default_0_5"]["total_cost"],
        "cost_savings_over_default": cost_analysis["cost_savings"],
        "pct_savings": cost_analysis["pct_cost_savings"],
    }
