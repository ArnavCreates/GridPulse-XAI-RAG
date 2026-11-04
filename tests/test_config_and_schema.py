"""Unit tests for centralized configuration and 10-column grid telemetry schema."""

from pathlib import Path
import pandas as pd
import pytest
from pydantic import ValidationError

from src.config_loader import count_leaf_parameters, load_config
from src.data.feature_engineering import FEATURE_NAMES, TelemetryFeatureExtractor
from src.data.telemetry_generator import generate_single_event, generate_telemetry_dataset
from src.api.schemas import TelemetryEventInput


def test_config_loader_and_leaf_parameters():
    """Verify config sections and centralized parameter coverage."""
    cfg = load_config()
    assert isinstance(cfg, dict)

    # Check mandatory top-level sections
    mandatory = [
        "system",
        "telemetry_schema",
        "feature_engineering",
        "lightgbm",
        "lstm_autoencoder",
        "topology_graph",
        "hybrid_ensemble",
        "shap_xai",
        "rag_diagnostics",
        "api",
    ]
    for section in mandatory:
        assert section in cfg, f"Missing section: {section}"

    leaf_count = count_leaf_parameters(cfg)
    assert len(cfg) == 10
    assert leaf_count == 114, f"Expected 114 leaf parameters, got {leaf_count}"


def test_telemetry_schema_validation():
    """Verify Pydantic schema enforces 10-column constraints and bounds."""
    # Valid nominal event
    valid_event = generate_single_event(is_anomaly=False)
    model = TelemetryEventInput(**valid_event)
    assert model.window_step == valid_event["window_step"]
    assert model.protocol_type == "XROOTD_STREAM"
    assert model.payload_mb >= 0.0

    # Invalid event: negative payload
    invalid_event = dict(valid_event)
    invalid_event["payload_mb"] = -10.0
    with pytest.raises(ValidationError):
        TelemetryEventInput(**invalid_event)

    # Invalid event: out-of-range CPU load (>100%)
    invalid_event2 = dict(valid_event)
    invalid_event2["src_cpu_load_curr"] = 125.0
    with pytest.raises(ValidationError):
        TelemetryEventInput(**invalid_event2)


def test_telemetry_generator_shapes_and_modes():
    """Verify synthetic dataset generator produces exact 10 columns and valid modes."""
    df = generate_telemetry_dataset(n_samples=200, seed=123)
    assert df.shape == (200, 10)
    expected_cols = [
        "window_step",
        "protocol_type",
        "payload_mb",
        "source_node",
        "src_cpu_load_prev",
        "src_cpu_load_curr",
        "target_node",
        "dst_queue_depth_prev",
        "dst_queue_depth_curr",
        "is_anomaly",
    ]
    assert list(df.columns) == expected_cols
    assert set(df["is_anomaly"].unique()).issubset({0, 1})
    assert (df["is_anomaly"] == 1).sum() > 0

    with_modes = generate_telemetry_dataset(n_samples=200, seed=123, return_modes=True)
    mode_b = with_modes[with_modes["_mode"] == "MODE_B"]
    assert len(mode_b) == 6
    assert mode_b["source_node"].nunique() == 1
    assert mode_b["window_step"].sort_values().diff().dropna().eq(1).all()


def test_feature_engineering_14_channels():
    """Verify feature extractor computes exact 14 engineered features."""
    df = generate_telemetry_dataset(n_samples=50, seed=42)
    extractor = TelemetryFeatureExtractor(eps=1e-4, use_scaler=False)
    feat_df = extractor.transform_df(df)

    assert feat_df.shape == (50, 14)
    assert list(feat_df.columns) == FEATURE_NAMES
    assert not feat_df.isna().any().any()

    # Check non-negative surge ratios
    assert (feat_df["queue_surge_ratio"] >= 0.0).all()
    assert (feat_df["bandwidth_pressure"] >= 0.0).all()

    sequence_frame = pd.DataFrame({
        "source_node": ["NODE_A", "NODE_B", "NODE_A", "NODE_B"],
        "window_step": [1, 1, 2, 2],
    })
    sequence_values = [[1.0], [10.0], [2.0], [20.0]]
    sequences, positions = TelemetryFeatureExtractor.build_sequences(
        sequence_frame,
        sequence_values,
        seq_len=3,
    )
    assert positions.tolist() == [0, 1, 2, 3]
    assert sequences[2, :, 0].tolist() == [1.0, 1.0, 2.0]
    assert sequences[3, :, 0].tolist() == [10.0, 10.0, 20.0]
