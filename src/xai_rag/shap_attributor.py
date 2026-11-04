"""Per-alert SHAP feature attributor using TreeExplainer on LightGBM.

Isolates root-cause telemetry channels driving anomalies with signed contribution
values, observed sensor metrics, and directional indicators.
"""

from typing import Any, Dict, List, Optional
import warnings
import numpy as np
import pandas as pd
from ..data.feature_engineering import FEATURE_NAMES


class ShapAttributor:
    """Computes exact SHAP attributions for individual grid telemetry events."""

    def __init__(
        self,
        lgbm_model: Any,
        feature_names: Optional[List[str]] = None,
        top_k: int = 4,
        check_additivity: bool = False,
        explainer: Any = None,
    ):
        self.lgbm_model = lgbm_model
        self.feature_names = feature_names or FEATURE_NAMES
        self.top_k = top_k
        self.check_additivity = check_additivity
        # Initialize SHAP TreeExplainer
        if explainer is None:
            import shap

            explainer = shap.TreeExplainer(
                self.lgbm_model,
                feature_names=self.feature_names,
                model_output="raw",
            )
        self.explainer = explainer

    def _shap_values(self, features: np.ndarray) -> Any:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="LightGBM binary classifier with TreeExplainer shap values output has changed to a list of ndarray",
                category=UserWarning,
            )
            return self.explainer.shap_values(
                features,
                check_additivity=self.check_additivity,
            )

    def explain_instance(
        self,
        features_1d: np.ndarray | pd.Series,
        observed_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Explain a single telemetry event and return top-k root-cause channels.

        Args:
            features_1d: 1D array of 14 engineered features.
            observed_dict: Optional dictionary with human-readable observed values.

        Returns:
            List of top-k dictionaries with:
            - feature: feature name
            - observed_value: raw or engineered value
            - shap_impact: signed SHAP value
            - direction: '+' (anomaly-driving) or '-' (suppressing)
            - abs_impact: absolute magnitude
        """
        if isinstance(features_1d, pd.Series):
            features_arr = features_1d.values.reshape(1, -1)
        elif isinstance(features_1d, np.ndarray):
            features_arr = (
                features_1d.reshape(1, -1) if features_1d.ndim == 1 else features_1d
            )
        else:
            features_arr = np.array(features_1d).reshape(1, -1)

        shap_values = self._shap_values(features_arr)

        # In binary LightGBM, shap_values can be a list [class0, class1] or 2D array
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = shap_values[1][0]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[0, :, 1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 2:
            vals = shap_values[0]
        else:
            vals = np.array(shap_values).flatten()

        raw_feat_vals = features_arr[0]

        attributions = []
        for i, feat_name in enumerate(self.feature_names):
            impact = float(vals[i])
            obs_val = (
                observed_dict.get(feat_name, float(raw_feat_vals[i]))
                if observed_dict
                else float(raw_feat_vals[i])
            )
            attributions.append({
                "feature": feat_name,
                "observed_value": round(float(obs_val), 4),
                "shap_impact": round(impact, 4),
                "impact_units": "log-odds",
                "direction": "+" if impact > 0 else "-",
                "abs_impact": abs(impact),
            })

        # Rank by absolute impact descending
        attributions.sort(key=lambda x: x["abs_impact"], reverse=True)
        return attributions[: self.top_k]

    def compute_global_importance(self, X_sample: np.ndarray) -> pd.DataFrame:
        """Compute global mean absolute SHAP feature importance over a sample."""
        shap_values = self._shap_values(X_sample)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            vals = shap_values[1]
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            vals = shap_values[:, :, 1]
        else:
            vals = np.array(shap_values)

        mean_abs_impact = np.mean(np.abs(vals), axis=0)
        df_imp = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs_impact,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_imp
