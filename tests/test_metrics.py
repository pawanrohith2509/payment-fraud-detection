"""
Unit tests for fraud evaluation and cost metrics.
"""

import numpy as np
import pytest
from payment_fraud_detection.metrics import (
    compute_pr_auc,
    compute_roc_auc,
    compute_brier_score,
    recall_at_min_precision,
    calculate_business_cost,
    find_optimal_cost_threshold,
    evaluate_model_comprehensive,
)


@pytest.fixture
def sample_predictions():
    # Synthetic imbalanced test batch: 1000 items, 20 positive cases (2% fraud)
    np.random.seed(42)
    y_true = np.zeros(1000, dtype=int)
    fraud_indices = np.random.choice(1000, size=20, replace=False)
    y_true[fraud_indices] = 1
    
    # Model assigns higher scores to frauds with some noise
    y_prob = np.random.beta(1, 15, size=1000)
    y_prob[fraud_indices] = np.random.beta(8, 2, size=20)
    return y_true, y_prob


def test_auc_metrics(sample_predictions):
    y_true, y_prob = sample_predictions
    pr_auc = compute_pr_auc(y_true, y_prob)
    roc_auc = compute_roc_auc(y_true, y_prob)
    brier = compute_brier_score(y_true, y_prob)
    
    assert 0.0 <= pr_auc <= 1.0
    assert 0.5 <= roc_auc <= 1.0
    assert 0.0 <= brier <= 1.0


def test_recall_at_min_precision(sample_predictions):
    y_true, y_prob = sample_predictions
    recall, thresh = recall_at_min_precision(y_true, y_prob, min_precision=0.80)
    
    assert 0.0 <= recall <= 1.0
    assert 0.0 <= thresh <= 1.0


def test_cost_optimization(sample_predictions):
    y_true, y_prob = sample_predictions
    cost_fn = 200.0  # Missed fraud is costly
    cost_fp = 10.0   # False alert investigation cost
    
    analysis = find_optimal_cost_threshold(y_true, y_prob, cost_fn=cost_fn, cost_fp=cost_fp)
    
    optimal = analysis["optimal"]
    default = analysis["default_0_5"]
    
    assert optimal["total_cost"] <= default["total_cost"]
    assert analysis["cost_savings"] >= 0.0


def test_comprehensive_evaluation(sample_predictions):
    y_true, y_prob = sample_predictions
    report = evaluate_model_comprehensive(y_true, y_prob, model_name="LightGBM_Test")
    
    assert report["model_name"] == "LightGBM_Test"
    assert "pr_auc" in report
    assert "roc_auc" in report
    assert "optimal_threshold" in report
    assert "cost_savings_over_default" in report
