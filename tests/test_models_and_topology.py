"""Unit tests for LightGBM, PyTorch LSTM Autoencoder, and NetworkX Topology analyzer."""

import numpy as np
import pytest
import torch

from src.data.feature_engineering import TelemetryFeatureExtractor
from src.data.telemetry_generator import generate_telemetry_dataset
from src.models.hybrid_ensemble import HybridEnsembleDetector
from src.models.lgbm_detector import LGBMAnomalyDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector, LSTMAutoencoderNet
from src.models.topology_analyzer import GridTopologyAnalyzer


@pytest.fixture(scope="module")
def sample_telemetry_data():
    """Fixture generating sample dataset and engineered features."""
    df = generate_telemetry_dataset(n_samples=300, seed=42)
    extractor = TelemetryFeatureExtractor(eps=1e-4, use_scaler=False)
    X = extractor.transform_df(df).values
    y = df["is_anomaly"].values
    return df, X, y, extractor


def test_lgbm_detector(sample_telemetry_data):
    """Verify LightGBM detector fits and outputs valid probability scores in [0, 1]."""
    _, X, y, _ = sample_telemetry_data
    detector = LGBMAnomalyDetector(n_estimators=30, num_leaves=15, max_depth=4)
    detector.fit(X, y)

    probs = detector.predict_proba(X[:10])
    assert probs.shape == (10,)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    preds = detector.predict(X[:10], threshold=0.5)
    assert preds.shape == (10,)
    assert set(preds).issubset({0, 1})


def test_lstm_autoencoder_architecture():
    """Verify PyTorch LSTM Autoencoder forward pass shape and reconstruction."""
    net = LSTMAutoencoderNet(input_dim=6, hidden_dim=16, latent_dim=8, seq_len=6)
    dummy_input = torch.randn(4, 6, 6)
    reconstructed, z = net(dummy_input)

    assert reconstructed.shape == (4, 6, 6)
    assert z.shape == (4, 8)


def test_lstm_autoencoder_detector(sample_telemetry_data):
    """Verify LSTM Autoencoder detector training and score computation."""
    df, X, y, extractor = sample_telemetry_data
    # Train only on nominal sequences
    nom_mask = y == 0
    X_nom = X[nom_mask][:, [0, 1, 2, 4, 6, 7]]
    seqs, _ = extractor.build_sequences(X_nom, seq_len=6)

    detector = LSTMAutoencoderDetector(
        input_dim=6,
        hidden_dim=16,
        latent_dim=8,
        seq_len=6,
        epochs=3,
        batch_size=32,
    )
    detector.fit(seqs)

    # Score sample sequences
    test_seqs = seqs[:5]
    scores = detector.predict_score(test_seqs)
    assert scores.shape == (5,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_networkx_topology_analyzer(sample_telemetry_data):
    """Verify NetworkX directed graph builds, detects cycles and bottleneck degrees."""
    df, _, _, _ = sample_telemetry_data
    analyzer = GridTopologyAnalyzer(
        max_cycle_length=4,
        fan_in_threshold=5,
        fan_out_threshold=5,
    )
    analyzer.build_graph_from_events(df)

    assert len(analyzer.graph.nodes) > 10
    assert len(analyzer.graph.edges) > 10

    # Test scoring an event
    score, meta = analyzer.analyze_event(
        source_node="NODE_T1_005",
        target_node="NODE_T1_006",
        payload_mb=50.0,
        protocol="RPC_CONTROL",
    )
    assert 0.0 <= score <= 1.0
    assert "in_routing_cycle" in meta
    assert "bottleneck_regime" in meta
    assert isinstance(meta["target_fan_in_degree"], int)
    assert isinstance(meta["source_fan_out_degree"], int)


def test_hybrid_ensemble_combiner():
    """Verify hybrid combiner weights, thresholding, and metric calculations."""
    ensemble = HybridEnsembleDetector(
        weight_lgbm=0.5,
        weight_lstm_ae=0.3,
        weight_topology=0.2,
        alert_threshold=0.5,
    )
    s_lgbm = np.array([0.8, 0.2])
    s_lstm = np.array([0.9, 0.1])
    s_graph = np.array([0.7, 0.3])

    hybrid = ensemble.combine_scores(s_lgbm, s_lstm, s_graph)
    assert hybrid.shape == (2,)
    assert hybrid[0] > 0.7
    assert hybrid[1] < 0.3

    is_anom, sev = ensemble.classify_severity(0.85)
    assert is_anom is True
    assert sev == "CRITICAL"

    is_anom2, sev2 = ensemble.classify_severity(0.20)
    assert is_anom2 is False
    assert sev2 == "NOMINAL"

    # Test metric computation
    y_true = np.array([1, 0, 1, 0])
    scores = np.array([0.9, 0.1, 0.8, 0.2])
    metrics = ensemble.compute_component_metrics(y_true, scores)
    assert metrics["roc_auc"] == 1.0
    assert metrics["f1_score"] == 1.0
