"""
Model training, imbalance handling, and evaluation pipelines for fraud detection.

Implements:
1. Logistic Regression (StandardScaler + class_weight='balanced')
2. Random Forest (balanced_subsample)
3. LightGBM (scale_pos_weight)
4. XGBoost (scale_pos_weight)
5. Model Probability Calibration (Isotonic / Sigmoid)
6. SHAP TreeExplainer for feature attribution
"""

from typing import Dict, Any, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
import lightgbm as lgb
import xgboost as xgb
import shap

from payment_fraud_detection.metrics import evaluate_model_comprehensive


class FraudModelTrainer:
    """
    Unified trainer supporting classical ML baselines and gradient boosted trees.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models: Dict[str, Any] = {}
        self.results: Dict[str, Dict[str, Any]] = {}

    def _get_scale_pos_weight(self, y_train: np.ndarray) -> float:
        """Compute negative to positive class ratio for imbalanced weighting."""
        n_pos = np.sum(y_train == 1)
        n_neg = np.sum(y_train == 0)
        return float(n_neg / n_pos) if n_pos > 0 else 1.0

    def train_logistic_regression(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        class_weight: str = "balanced",
        max_iter: int = 1000,
    ) -> Pipeline:
        """Baseline 1: Scaled Logistic Regression."""
        model = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                class_weight=class_weight,
                max_iter=max_iter,
                random_state=self.random_state,
                solver="lbfgs",
            )),
        ])
        model.fit(X_train, y_train)
        self.models["LogisticRegression"] = model
        return model

    def train_random_forest(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        n_estimators: int = 100,
        max_depth: int = 12,
    ) -> RandomForestClassifier:
        """Baseline 2: Random Forest with balanced subsampling."""
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced_subsample",
            random_state=self.random_state,
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models["RandomForest"] = model
        return model

    def train_lightgbm(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[np.ndarray] = None,
        use_class_weights: bool = True,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
    ) -> lgb.LGBMClassifier:
        """Production Baseline: LightGBM with scale_pos_weight."""
        scale_pos = self._get_scale_pos_weight(y_train) if use_class_weights else 1.0
        
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )
        
        if X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20, verbose=False)],
            )
        else:
            model.fit(X_train, y_train)
            
        self.models["LightGBM"] = model
        return model

    def train_xgboost(
        self,
        X_train: pd.DataFrame,
        y_train: np.ndarray,
        use_class_weights: bool = True,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
    ) -> xgb.XGBClassifier:
        """Gradient Boosted Trees: XGBoost with scale_pos_weight."""
        scale_pos = self._get_scale_pos_weight(y_train) if use_class_weights else 1.0
        
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            scale_pos_weight=scale_pos,
            random_state=self.random_state,
            eval_metric="aucpr",
            n_jobs=-1,
        )
        model.fit(X_train, y_train)
        self.models["XGBoost"] = model
        return model

    def calibrate_model(
        self,
        base_model_name: str,
        X_calib: pd.DataFrame,
        y_calib: np.ndarray,
        method: str = "isotonic",
    ) -> CalibratedClassifierCV:
        """Calibrate probabilities using hold-out validation data."""
        base_model = self.models[base_model_name]
        calibrated = CalibratedClassifierCV(estimator=base_model, method=method, cv="prefit")
        calibrated.fit(X_calib, y_calib)
        self.models[f"{base_model_name}_Calibrated"] = calibrated
        return calibrated

    def evaluate_all(
        self,
        X_test: pd.DataFrame,
        y_test: np.ndarray,
        cost_fn: float = 150.0,
        cost_fp: float = 10.0,
    ) -> pd.DataFrame:
        """Evaluate all trained models and format into comparison table."""
        records = []
        for name, model in self.models.items():
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_model_comprehensive(
                y_test, y_prob, model_name=name, cost_fn=cost_fn, cost_fp=cost_fp
            )
            records.append(metrics)
            self.results[name] = metrics
            
        df_results = pd.DataFrame(records)
        return df_results.sort_values(by="pr_auc", ascending=False).reset_index(drop=True)

    def explain_with_shap(
        self,
        model_name: str,
        X_sample: pd.DataFrame,
    ) -> Tuple[np.ndarray, shap.Explainer]:
        """Compute SHAP feature importances using TreeExplainer."""
        model = self.models[model_name]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
        return shap_values, explainer
