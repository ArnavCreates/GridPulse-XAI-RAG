"""Stacked logistic anomaly ensemble for tabular, sequence, and topology scores.

Combines scores from LightGBM, an LSTM autoencoder, and NetworkX topology with
a logistic meta-classifier over the three scores and their maximum:
    s_hybrid(x) = sigma(w0 + w1*s_lgbm + w2*s_lstm + w3*s_graph + w4*max(s_lgbm, s_lstm, s_graph))
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class HybridEnsembleDetector:
    """Combine detector outputs with a fitted logistic meta-classifier."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        weight_lgbm: float = 0.35,
        weight_lstm_ae: float = 0.30,
        weight_topology: float = 0.35,
        alert_threshold: float = 0.50,
        critical_threshold: float = 0.80,
        high_threshold: float = 0.65,
        medium_threshold: float = 0.50,
        prior_blend_weight: float = 0.20,
    ):
        if config:
            self.w_lgbm = float(config.get("weight_lgbm", weight_lgbm))
            self.w_lstm = float(config.get("weight_lstm_ae", weight_lstm_ae))
            self.w_graph = float(config.get("weight_topology", weight_topology))
            self.alert_threshold = float(config.get("alert_threshold", alert_threshold))
            self.critical_threshold = float(
                config.get("critical_severity_threshold", critical_threshold)
            )
            self.high_threshold = float(config.get("high_severity_threshold", high_threshold))
            self.medium_threshold = float(config.get("medium_severity_threshold", medium_threshold))
            self.prior_blend_weight = float(config.get("prior_blend_weight", prior_blend_weight))
            self.min_active_modalities = int(config.get("min_active_modalities", 1))
            self.random_state = int(config.get("random_state", 42))
        else:
            self.w_lgbm = weight_lgbm
            self.w_lstm = weight_lstm_ae
            self.w_graph = weight_topology
            self.alert_threshold = alert_threshold
            self.critical_threshold = critical_threshold
            self.high_threshold = high_threshold
            self.medium_threshold = medium_threshold
            self.prior_blend_weight = prior_blend_weight
            self.min_active_modalities = 1
            self.random_state = 42

        if not 1 <= self.min_active_modalities <= 3:
            raise ValueError("min_active_modalities must be between 1 and 3")
        if sum(weight > 0 for weight in (self.w_lgbm, self.w_lstm, self.w_graph)) < self.min_active_modalities:
            raise ValueError("Configured modality weights do not meet min_active_modalities")

        # Logistic stacking weights: [w0, w1, w2, w3, w4]
        # w0: intercept, w1: lgbm, w2: lstm, w3: graph, w4: max_boost
        self.w0 = -4.20
        self.w1 = 3.20
        self.w2 = 2.80
        self.w3 = 2.80
        self.w4 = 4.20
        self.meta_model: Optional[LogisticRegression] = None
        self.is_fitted = False

    def _build_meta_features(
        self,
        s_lgbm: np.ndarray,
        s_lstm: np.ndarray,
        s_graph: np.ndarray,
    ) -> np.ndarray:
        """Construct [s_lgbm, s_lstm, s_graph, max(s_lgbm, s_lstm, s_graph)] feature matrix."""
        s1 = np.asarray(s_lgbm, dtype=np.float32)
        s2 = np.asarray(s_lstm, dtype=np.float32)
        s3 = np.asarray(s_graph, dtype=np.float32)
        s_max = np.maximum(np.maximum(s1, s2), s3)
        return np.column_stack([s1, s2, s3, s_max])

    def fit(
        self,
        s_lgbm_train: np.ndarray,
        s_lstm_train: np.ndarray,
        s_graph_train: np.ndarray,
        y_train: np.ndarray,
    ) -> "HybridEnsembleDetector":
        """Fit the logistic meta-classifier and calibrate its decision threshold."""
        X_meta = self._build_meta_features(s_lgbm_train, s_lstm_train, s_graph_train)
        lr = LogisticRegression(C=5.0, random_state=self.random_state, max_iter=200)
        lr.fit(X_meta, y_train)

        self.meta_model = lr
        self.w0 = float(lr.intercept_[0])
        self.w1 = float(lr.coef_[0][0])
        self.w2 = float(lr.coef_[0][1])
        self.w3 = float(lr.coef_[0][2])
        self.w4 = float(lr.coef_[0][3])

        # Calibrate optimal decision threshold on training set
        train_probs = lr.predict_proba(X_meta)[:, 1]
        best_t = 0.50
        best_f1 = 0.0
        for t in np.linspace(0.15, 0.75, 61):
            f = f1_score(y_train, (train_probs >= t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_t = float(t)

        self.alert_threshold = round(best_t, 2)
        self.is_fitted = True
        return self

    def combine_scores(
        self,
        s_lgbm: Union[np.ndarray, float],
        s_lstm: Union[np.ndarray, float],
        s_graph: Union[np.ndarray, float],
    ) -> Union[np.ndarray, float]:
        """Combine component anomaly scores with the fitted logistic model."""
        is_scalar = np.isscalar(s_lgbm) and np.isscalar(s_lstm) and np.isscalar(s_graph)

        s1 = np.asarray(s_lgbm, dtype=np.float32)
        s2 = np.asarray(s_lstm, dtype=np.float32)
        s3 = np.asarray(s_graph, dtype=np.float32)
        s_max = np.maximum(np.maximum(s1, s2), s3)

        if self.meta_model is not None:
            stacked = self.meta_model.predict_proba(self._build_meta_features(s1, s2, s3))[:, 1]
        else:
            logit = self.w0 + self.w1 * s1 + self.w2 * s2 + self.w3 * s3 + self.w4 * s_max
            stacked = 1.0 / (1.0 + np.exp(-np.clip(logit, -15.0, 15.0)))

        prior_weights = np.asarray([self.w_lgbm, self.w_lstm, self.w_graph], dtype=np.float32)
        weight_sum = float(prior_weights.sum())
        if weight_sum <= 0.0:
            prior_weights = np.full(3, 1.0 / 3.0, dtype=np.float32)
        else:
            prior_weights = prior_weights / weight_sum
        prior = prior_weights[0] * s1 + prior_weights[1] * s2 + prior_weights[2] * s3
        beta = float(np.clip(self.prior_blend_weight, 0.0, 1.0))
        hybrid = (1.0 - beta) * stacked + beta * prior

        if is_scalar:
            return float(np.clip(hybrid.item(), 0.0, 1.0))
        return np.clip(hybrid, 0.0, 1.0)

    def classify_severity(self, hybrid_score: float) -> Tuple[bool, str]:
        """Classify binary alert status and severity level."""
        is_anomaly = hybrid_score >= self.alert_threshold
        if hybrid_score >= self.critical_threshold:
            severity = "CRITICAL"
        elif hybrid_score >= self.high_threshold:
            severity = "HIGH"
        elif hybrid_score >= self.medium_threshold:
            severity = "MEDIUM"
        else:
            severity = "NOMINAL"
        return is_anomaly, severity

    @staticmethod
    def compute_component_metrics(
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float = 0.5,
    ) -> Dict[str, float]:
        """Compute standard empirical discrimination and detection metrics."""
        y_pred = (scores >= threshold).astype(int)
        roc_auc = float(roc_auc_score(y_true, scores))
        pr_auc = float(average_precision_score(y_true, scores))
        f1 = float(f1_score(y_true, y_pred, zero_division=0))
        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))

        return {
            "roc_auc": round(roc_auc, 5),
            "pr_auc": round(pr_auc, 5),
            "f1_score": round(f1, 5),
            "precision": round(prec, 5),
            "recall": round(rec, 5),
        }

    def evaluate_all(
        self,
        y_true: np.ndarray,
        s_lgbm: np.ndarray,
        s_lstm: np.ndarray,
        s_graph: np.ndarray,
    ) -> Dict[str, Dict[str, float]]:
        """Evaluate and return comparative metrics for all components and the ensemble."""
        s_hybrid = self.combine_scores(s_lgbm, s_lstm, s_graph)

        return {
            "LightGBM": self.compute_component_metrics(y_true, s_lgbm, 0.50),
            "LSTM_Autoencoder": self.compute_component_metrics(y_true, s_lstm, 0.50),
            "NetworkX_Topology": self.compute_component_metrics(y_true, s_graph, 0.50),
            "Hybrid_Ensemble": self.compute_component_metrics(y_true, s_hybrid, self.alert_threshold),
        }
