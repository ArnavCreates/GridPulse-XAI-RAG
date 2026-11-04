"""LightGBM gradient-boosted tabular anomaly detector.

Trains a gradient-boosted decision tree ensemble on engineered telemetry features
and outputs calibrated anomaly probability scores for downstream hybrid ensembling.
"""

from typing import Any, Dict, Optional
import lightgbm as lgb
import numpy as np


class LGBMAnomalyDetector:
    """Gradient boosted tabular classifier for grid telemetry anomaly detection."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, **kwargs: Any):
        default_params: Dict[str, Any] = {
            "n_estimators": 120,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "scale_pos_weight": 3.5,
            "objective": "binary",
            "metric": "binary_logloss",
            "boost_from_average": True,
            "random_state": 42,
            "n_jobs": 2,
            "verbose": -1,
        }
        if config:
            default_params.update(config)
        default_params.update(kwargs)

        self.params = default_params
        self.model = lgb.LGBMClassifier(**self.params)
        self.is_fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LGBMAnomalyDetector":
        """Fit the LightGBM classifier on engineered features."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict anomaly probabilities for input feature vectors."""
        if not self.is_fitted:
            raise RuntimeError("LGBMAnomalyDetector must be fitted before predict_proba.")
        # Ensure 2D
        if X.ndim == 1:
            X = np.expand_dims(X, axis=0)
        probs = self.model.predict_proba(X)
        return probs[:, 1]  # Return probability of anomaly class

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Predict binary anomaly decisions based on decision threshold."""
        scores = self.predict_proba(X)
        return (scores >= threshold).astype(int)

    def get_model(self) -> lgb.LGBMClassifier:
        """Return the underlying LightGBM classifier for SHAP explanation."""
        return self.model
