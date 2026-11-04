"""Feature engineering pipeline for distributed grid telemetry.

Transforms 10-column raw grid telemetry into 14 engineered kinematic,
ratio, queue-surge, and protocol-encoded features with sequence windowing.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

FEATURE_NAMES: List[str] = [
    "cpu_load_delta",
    "queue_depth_delta",
    "queue_surge_ratio",
    "cpu_load_ratio",
    "bandwidth_pressure",
    "queue_to_payload_ratio",
    "payload_mb",
    "src_cpu_load_curr",
    "dst_queue_depth_curr",
    "protocol_xrootd",
    "protocol_gridftp",
    "protocol_heartbeat",
    "protocol_rpc",
    "protocol_replication",
]

# Continuous features used for LSTM Autoencoder sequence modeling
LSTM_FEATURE_INDICES = [0, 1, 2, 4, 6, 7]  # 6 continuous features


class TelemetryFeatureExtractor:
    """Extracts 14 engineered tabular features and sliding temporal sequence windows."""

    def __init__(
        self,
        eps: float = 1e-4,
        use_scaler: bool = True,
        config: Optional[Dict[str, Any]] = None,
        schema_config: Optional[Dict[str, Any]] = None,
    ):
        self.config = dict(config or {})
        self.schema_config = dict(schema_config or {})
        self.eps = float(self.config.get("eps", eps))
        self.use_scaler = bool(self.config.get("robust_scaling", use_scaler))
        self.window_length = int(
            self.schema_config.get(
                "sequence_length",
                self.config.get("rolling_window_size", 6),
            )
        )
        self.window_stride = int(self.schema_config.get("window_stride", 1))
        self.expected_feature_count = int(self.config.get("num_engineered_features", len(FEATURE_NAMES)))
        self.lower_clip = float(self.config.get("clip_bounds_lower", -np.inf))
        self.upper_clip = float(self.config.get("clip_bounds_upper", np.inf))
        if self.window_length != int(self.config.get("rolling_window_size", self.window_length)):
            raise ValueError("rolling_window_size and telemetry sequence_length must match")
        self.validate_feature_list(FEATURE_NAMES)
        self.scaler = RobustScaler() if self.use_scaler else None
        self.is_fitted = False
        self.feature_names = FEATURE_NAMES

    def validate_feature_list(self, feature_names: List[str]) -> None:
        if len(feature_names) != self.expected_feature_count:
            raise ValueError(
                f"Expected {self.expected_feature_count} engineered features, got {len(feature_names)}."
            )
        if len(set(feature_names)) != len(feature_names):
            raise ValueError("Engineered feature names must be unique.")

    def transform_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer 14 tabular features from a raw 10-column telemetry DataFrame."""
        schema = self.schema_config
        cpu_curr = df[schema.get("src_cpu_curr_col", "src_cpu_load_curr")].astype(float)
        cpu_prev = df[schema.get("src_cpu_prev_col", "src_cpu_load_prev")].astype(float)
        queue_curr = df[schema.get("dst_queue_curr_col", "dst_queue_depth_curr")].astype(float)
        queue_prev = df[schema.get("dst_queue_prev_col", "dst_queue_depth_prev")].astype(float)
        payload = df[schema.get("payload_col", "payload_mb")].astype(float)
        proto = df["protocol_type"].astype(str)

        cpu_load_delta = cpu_curr - cpu_prev if self.config.get("compute_load_delta", True) else 0.0
        queue_depth_delta = queue_curr - queue_prev
        queue_surge_ratio = (
            (queue_curr + self.eps) / (queue_prev + self.eps)
            if self.config.get("compute_queue_surge_ratio", True)
            else 1.0
        )
        cpu_load_ratio = (
            (cpu_curr + self.eps) / (cpu_prev + self.eps)
            if self.config.get("compute_load_ratio", True)
            else 1.0
        )
        bandwidth_pressure = (
            payload * (cpu_curr / 100.0)
            if self.config.get("compute_bandwidth_pressure", True)
            else 0.0
        )
        queue_to_payload_ratio = (
            queue_curr / (payload + self.eps)
            if self.config.get("compute_queue_to_payload", True)
            else 0.0
        )

        protocol_xrootd = (proto == "XROOTD_STREAM").astype(float)
        protocol_gridftp = (proto == "GRID_FTP").astype(float)
        protocol_heartbeat = (proto == "HEARTBEAT_SYNC").astype(float)
        protocol_rpc = (proto == "RPC_CONTROL").astype(float)
        protocol_replication = (proto == "CACHE_REPLICATION").astype(float)

        if not self.config.get("include_raw_features", True):
            payload = payload * 0.0
            cpu_curr = cpu_curr * 0.0
            queue_curr = queue_curr * 0.0

        feat_df = pd.DataFrame(
            {
                "cpu_load_delta": cpu_load_delta,
                "queue_depth_delta": queue_depth_delta,
                "queue_surge_ratio": queue_surge_ratio,
                "cpu_load_ratio": cpu_load_ratio,
                "bandwidth_pressure": bandwidth_pressure,
                "queue_to_payload_ratio": queue_to_payload_ratio,
                "payload_mb": payload,
                "src_cpu_load_curr": cpu_curr,
                "dst_queue_depth_curr": queue_curr,
                "protocol_xrootd": protocol_xrootd,
                "protocol_gridftp": protocol_gridftp,
                "protocol_heartbeat": protocol_heartbeat,
                "protocol_rpc": protocol_rpc,
                "protocol_replication": protocol_replication,
            },
            index=df.index,
        )
        feat_df = feat_df.clip(lower=self.lower_clip, upper=self.upper_clip)
        self.validate_feature_list(list(feat_df.columns))
        return feat_df

    def transform_single(self, event: Dict[str, Any]) -> pd.DataFrame:
        """Transform a single 10-column dictionary event into a 1-row feature DataFrame."""
        df = pd.DataFrame([event])
        return self.transform_df(df)

    def fit(self, X: pd.DataFrame | np.ndarray) -> "TelemetryFeatureExtractor":
        """Fit scaler on tabular feature matrix."""
        if self.use_scaler and self.scaler is not None:
            self.scaler.fit(X)
        self.is_fitted = True
        return self

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit scaler and return normalized feature array."""
        feat_df = self.transform_df(X)
        if self.use_scaler and self.scaler is not None:
            arr = self.scaler.fit_transform(feat_df.values)
        else:
            arr = feat_df.values
        self.is_fitted = True
        return arr

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform and normalize feature DataFrame."""
        feat_df = self.transform_df(X)
        if self.use_scaler and self.scaler is not None and self.is_fitted:
            return self.scaler.transform(feat_df.values)
        return feat_df.values

    @staticmethod
    def build_sequences(
        df: pd.DataFrame | np.ndarray,
        feature_matrix: Optional[np.ndarray] = None,
        seq_len: Optional[int] = None,
        window_stride: int = 1,
        source_col: str = "source_node",
        time_col: str = "window_step",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build sliding window sequence tensors for PyTorch LSTM Autoencoder.

        Args:
            df: Source-node and timestamp columns for the input feature rows. An array
                may be passed alone for backwards-compatible single-stream use.
            feature_matrix: (N, D) array of continuous features.
            seq_len: Length of each temporal sequence window.

        Returns:
            Tuple of one sequence per input row and the corresponding row indices.
        """
        if feature_matrix is None:
            feature_matrix = np.asarray(df, dtype=np.float32)
            frame = pd.DataFrame({"source_node": "__single_stream__", "window_step": np.arange(len(feature_matrix))})
        else:
            frame = df.reset_index(drop=True) if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
            feature_matrix = np.asarray(feature_matrix, dtype=np.float32)

        n_samples, feature_count = feature_matrix.shape
        if len(frame) != n_samples:
            raise ValueError("DataFrame rows and feature_matrix rows must have equal length.")
        seq_len = int(seq_len or 6)
        window_stride = max(1, int(window_stride))
        if n_samples == 0:
            return np.empty((0, seq_len, feature_count), dtype=np.float32), np.array([], dtype=int)

        sources = frame[source_col].astype(str) if source_col in frame else pd.Series("__single_stream__", index=frame.index)
        times = frame[time_col] if time_col in frame else pd.Series(np.arange(n_samples), index=frame.index)
        ordered = pd.DataFrame({"source": sources, "time": times, "position": np.arange(n_samples)})
        ordered = ordered.sort_values(["source", "time", "position"], kind="mergesort")
        sequences = np.empty((n_samples, seq_len, feature_count), dtype=np.float32)

        for _, group in ordered.groupby("source", sort=False):
            positions = group["position"].to_numpy(dtype=int)
            for rank, target_position in enumerate(positions):
                history_ranks = list(range(rank, -1, -window_stride))[:seq_len]
                history_positions = positions[np.asarray(history_ranks[::-1], dtype=int)]
                history = feature_matrix[history_positions]
                pad_count = seq_len - len(history)
                if pad_count:
                    padding = np.repeat(history[:1], pad_count, axis=0)
                    history = np.concatenate([padding, history], axis=0)
                sequences[target_position] = history

        return sequences, np.arange(n_samples, dtype=int)
