"""Asynchronous FastAPI service for GridPulse-XAI-RAG.

Exposes real-time distributed grid anomaly diagnosis, per-alert SHAP explanations,
ChromaDB runbook recommendations, cryptographic audit logging, and Prometheus metrics.
"""

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from contextlib import asynccontextmanager
from collections import deque
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional
import uuid
import numpy as np

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, Summary, generate_latest

from ..config_loader import apply_deterministic_seeds, load_config
from ..data.feature_engineering import LSTM_FEATURE_INDICES, TelemetryFeatureExtractor
from ..data.telemetry_generator import generate_telemetry_dataset
from ..models.hybrid_ensemble import HybridEnsembleDetector
from ..models.lgbm_detector import LGBMAnomalyDetector
from ..models.lstm_autoencoder import LSTMAutoencoderDetector
from ..models.topology_analyzer import GridTopologyAnalyzer
from ..xai_rag.audit_logger import AuditLogger
from ..xai_rag.runbook_rag import RunbookRAGEngine
from ..xai_rag.shap_attributor import ShapAttributor
from .schemas import (
    DiagnosticResponse,
    HealthResponse,
    ModelScores,
    RootResponse,
    RunbookRecommendation,
    ShapDriver,
    TelemetryBatchInput,
    TelemetryEventInput,
    TopologyMetadata,
)

# Enforce offline mode for Hugging Face Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

_BOOT_CONFIG = load_config()
_API_CONFIG = _BOOT_CONFIG["api"]
_METRICS_PREFIX = str(_API_CONFIG.get("metrics_prefix", "gridpulse_"))

# Prometheus Metrics Definitions
METRIC_REQUESTS = Counter(
    f"{_METRICS_PREFIX}requests_total",
    "Total number of telemetry diagnosis requests processed",
    ["status", "endpoint"],
)
METRIC_ANOMALIES = Counter(
    f"{_METRICS_PREFIX}anomalies_detected_total",
    "Total number of grid anomalies detected",
    ["severity", "regime"],
)
METRIC_LATENCY = Histogram(
    f"{_METRICS_PREFIX}inference_latency_seconds",
    "End-to-end inference and RAG retrieval latency in seconds",
    buckets=_API_CONFIG.get("histogram_buckets", [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5]),
)
METRIC_HYBRID_SCORE = Summary(
    f"{_METRICS_PREFIX}hybrid_score_summary",
    "Summary distribution of composite tri-modal anomaly scores",
)


class AppState:
    """Singleton holding initialized models and engines during application lifespan."""

    def __init__(self):
        self.config: Dict[str, Any] = {}
        self.feature_extractor: Optional[TelemetryFeatureExtractor] = None
        self.lgbm_model: Optional[LGBMAnomalyDetector] = None
        self.lstm_model: Optional[LSTMAutoencoderDetector] = None
        self.topology_analyzer: Optional[GridTopologyAnalyzer] = None
        self.hybrid_ensemble: Optional[HybridEnsembleDetector] = None
        self.shap_attributor: Optional[ShapAttributor] = None
        self.rag_engine: Optional[RunbookRAGEngine] = None
        self.audit_logger: Optional[AuditLogger] = None
        self.runbooks_indexed_count: int = 0
        self.node_history: Dict[str, deque] = {}
        self.history_lock = threading.Lock()
        self.is_ready: bool = False


state = AppState()


def deterministic_timestamp(window_step: int) -> str:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (epoch + timedelta(seconds=max(0, int(window_step)))).isoformat(timespec="milliseconds")


def initialize_gridpulse_system() -> None:
    """Initialize configurations, models, RAG engine, and loggers."""
    if state.is_ready:
        return

    state.config = load_config()
    seed = apply_deterministic_seeds(state.config)
    schema_cfg = state.config["telemetry_schema"]
    feature_cfg = state.config["feature_engineering"]
    seq_len = int(schema_cfg["sequence_length"])
    if seq_len != int(feature_cfg["rolling_window_size"]):
        raise ValueError("rolling_window_size and sequence_length must match")

    state.feature_extractor = TelemetryFeatureExtractor(
        config=feature_cfg,
        schema_config=schema_cfg,
    )

    train_df = generate_telemetry_dataset(
        n_samples=500,
        seed=seed,
        config=state.config,
    )
    X_train = state.feature_extractor.transform_df(train_df).values
    y_train = train_df["is_anomaly"].values

    lgbm_cfg = dict(state.config.get("lightgbm", {}))
    lgbm_cfg["n_estimators"] = min(30, int(lgbm_cfg.get("n_estimators", 30)))
    lgbm_cfg["random_state"] = seed
    state.lgbm_model = LGBMAnomalyDetector(config=lgbm_cfg)
    state.lgbm_model.fit(X_train, y_train)

    lstm_cfg = dict(state.config.get("lstm_autoencoder", {}))
    lstm_cfg["epochs"] = min(4, int(lstm_cfg.get("epochs", 4)))
    lstm_cfg["device"] = state.config["system"].get("device", "cpu")
    state.lstm_model = LSTMAutoencoderDetector(config=lstm_cfg)
    nominal_df = train_df[train_df["is_anomaly"] == 0].reset_index(drop=True)
    X_nom_feat = state.feature_extractor.transform_df(nominal_df).values
    X_nom_cont = X_nom_feat[:, LSTM_FEATURE_INDICES]
    seqs, _ = state.feature_extractor.build_sequences(
        nominal_df,
        X_nom_cont,
        seq_len=seq_len,
        window_stride=int(schema_cfg["window_stride"]),
        source_col=schema_cfg["src_node_col"],
        time_col=schema_cfg["time_col"],
    )
    state.lstm_model.fit(seqs)

    top_cfg = state.config.get("topology_graph", {})
    state.topology_analyzer = GridTopologyAnalyzer(config=top_cfg)
    state.topology_analyzer.build_graph_from_events(train_df)

    X_train_cont = X_train[:, LSTM_FEATURE_INDICES]
    train_sequences, _ = state.feature_extractor.build_sequences(
        train_df,
        X_train_cont,
        seq_len=seq_len,
        window_stride=int(schema_cfg["window_stride"]),
        source_col=schema_cfg["src_node_col"],
        time_col=schema_cfg["time_col"],
    )
    s_lgbm_train = state.lgbm_model.predict_proba(X_train)
    s_lstm_train = state.lstm_model.predict_score(train_sequences)
    s_graph_train, _ = state.topology_analyzer.score_batch(train_df)

    ens_cfg = state.config.get("hybrid_ensemble", {})
    ens_cfg["random_state"] = seed
    state.hybrid_ensemble = HybridEnsembleDetector(config=ens_cfg)
    state.hybrid_ensemble.fit(s_lgbm_train, s_lstm_train, s_graph_train, y_train)

    shap_cfg = state.config.get("shap_xai", {})
    state.shap_attributor = ShapAttributor(
        lgbm_model=state.lgbm_model.get_model(),
        feature_names=state.feature_extractor.feature_names,
        top_k=int(shap_cfg.get("top_k_features", 4)),
        check_additivity=shap_cfg.get("check_additivity", False),
    )

    rag_cfg = state.config.get("rag_diagnostics", {})
    state.rag_engine = RunbookRAGEngine(config=rag_cfg)
    state.runbooks_indexed_count = state.rag_engine.index_runbooks()

    api_cfg = state.config.get("api", {})
    log_path = api_cfg.get("audit_log_path", "logs/diagnostic_audit.jsonl")
    state.audit_logger = AuditLogger(
        log_path=log_path,
        deterministic_mode=bool(state.config["system"].get("deterministic_mode", False)),
    )
    state.node_history = {}

    state.is_ready = True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager initializing system components on startup."""
    initialize_gridpulse_system()
    yield


app = FastAPI(
    title="GridPulse-XAI-RAG API",
    description="Tri-Modal Distributed Computing Grid Anomaly Detection and Runbook RAG",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=_API_CONFIG.get("cors_origins", ["*"]),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/", response_model=RootResponse)
async def root():
    """Service status and schema summary."""
    return RootResponse(
        service=str(_API_CONFIG.get("service_name", "GridPulse-XAI-RAG")),
        version=str(_BOOT_CONFIG.get("system", {}).get("version", "1.0.0")),
        status="operational" if state.is_ready else "initializing",
        schema_columns=[
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
        ],
        endpoints=["/health", "/diagnose", "/diagnose/batch", "/predict", "/metrics"],
    )


@app.get("/health", response_model=HealthResponse)
async def health():
    """Health readiness probe confirming all models and ChromaDB indexing status."""
    return HealthResponse(
        status="healthy" if state.is_ready else "degraded",
        version=str(_BOOT_CONFIG.get("system", {}).get("version", "1.0.0")),
        models_loaded={
            "lightgbm": state.lgbm_model is not None and state.lgbm_model.is_fitted,
            "lstm_autoencoder": state.lstm_model is not None and state.lstm_model.is_fitted,
            "networkx_topology": state.topology_analyzer is not None,
            "shap_attributor": state.shap_attributor is not None,
            "chromadb_rag": state.rag_engine is not None and state.rag_engine.is_indexed,
            "audit_logger": state.audit_logger is not None,
        },
        runbooks_indexed_count=state.runbooks_indexed_count,
        device=str(state.config.get("system", {}).get("device", "cpu")),
    )


@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint exposing real-time operational telemetry."""
    if not _API_CONFIG.get("prometheus_enabled", True):
        raise HTTPException(status_code=404, detail="Metrics endpoint is disabled")
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def _process_single_diagnostic(event: TelemetryEventInput) -> DiagnosticResponse:
    """Internal synchronous worker executing tri-modal detection, SHAP, and RAG."""
    t0 = time.perf_counter()
    event_dict = event.model_dump()
    deterministic_mode = bool(state.config.get("system", {}).get("deterministic_mode", False))
    if event.event_id:
        event_id = event.event_id
    elif deterministic_mode:
        canonical_event = json.dumps(event_dict, sort_keys=True, separators=(",", ":"))
        event_id = f"evt_{hashlib.sha256(canonical_event.encode('utf-8')).hexdigest()[:12]}"
    else:
        event_id = f"evt_{uuid.uuid4().hex[:10]}"

    # 1. Feature Engineering
    feat_df = state.feature_extractor.transform_single(event_dict)
    feat_arr = feat_df.values

    # 2. LightGBM Tabular Probability
    s_lgbm = float(state.lgbm_model.predict_proba(feat_arr)[0])

    # 3. PyTorch LSTM Autoencoder Temporal Sequence Score
    cont_features = feat_arr[0, LSTM_FEATURE_INDICES]
    seq_len = int(state.config["telemetry_schema"]["sequence_length"])
    with state.history_lock:
        history = state.node_history.get(event.source_node)
        if history is None:
            history = deque(maxlen=seq_len)
            state.node_history[event.source_node] = history
        history.append(cont_features.copy())
        history_rows = np.asarray(list(history), dtype=np.float32)
    if len(history_rows) < seq_len:
        history_rows = np.concatenate(
            [np.repeat(history_rows[:1], seq_len - len(history_rows), axis=0), history_rows],
            axis=0,
        )
    s_lstm = float(state.lstm_model.predict_score(history_rows[None, :, :])[0])

    # 4. NetworkX Graph Topology Score & Metadata
    s_graph, topo_meta = state.topology_analyzer.analyze_event(
        source_node=event.source_node,
        target_node=event.target_node,
        payload_mb=event.payload_mb,
        protocol=event.protocol_type,
    )

    # 5. Hybrid Ensemble Score and Severity
    hybrid_score = float(state.hybrid_ensemble.combine_scores(s_lgbm, s_lstm, s_graph))
    is_anomaly, severity = state.hybrid_ensemble.classify_severity(hybrid_score)

    # 6. SHAP Root-Cause Channel Attribution
    shap_raw = state.shap_attributor.explain_instance(
        features_1d=feat_arr[0],
        observed_dict=event_dict,
    )
    top_shap_drivers = [
        ShapDriver(
            feature=d["feature"],
            observed_value=float(d["observed_value"]),
            shap_impact=float(d["shap_impact"]),
            impact_units=str(d.get("impact_units", "log-odds")),
            direction=str(d["direction"]),
        )
        for d in shap_raw
    ]

    # 7. ChromaDB Runbook RAG Retrieval
    rag_raw = state.rag_engine.retrieve_recommendations(
        protocol_type=event.protocol_type,
        shap_drivers=shap_raw,
        bottleneck_regime=topo_meta["bottleneck_regime"],
        temporal_drift_detected=(s_lstm >= 0.50),
        top_k=2,
    )
    runbook_recs = [
        RunbookRecommendation(
            runbook_id=r["runbook_id"],
            section_title=r["section_title"],
            severity=r["severity"],
            failure_mode=r["failure_mode"],
            similarity_score=float(r["similarity_score"]),
            recommended_action=r["recommended_action"],
            full_section_snippet=r.get("full_section_snippet"),
        )
        for r in rag_raw
    ]

    exec_time_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    timestamp_utc = (
        deterministic_timestamp(event.window_step)
        if deterministic_mode
        else datetime.now(timezone.utc).isoformat()
    )

    # 8. Cryptographic Hash-Chained Audit Logging
    state.audit_logger.log_diagnostic_event(
        event_id=event_id,
        source_node=event.source_node,
        target_node=event.target_node,
        hybrid_score=hybrid_score,
        is_anomaly=is_anomaly,
        severity=severity,
        top_shap_drivers=[d.feature for d in top_shap_drivers],
        retrieved_runbook_ids=[r.runbook_id for r in runbook_recs],
        timestamp_utc=timestamp_utc,
        window_step=event.window_step,
    )

    # 9. Prometheus Telemetry Recording
    METRIC_REQUESTS.labels(status="success", endpoint="/diagnose").inc()
    if is_anomaly:
        METRIC_ANOMALIES.labels(severity=severity, regime=topo_meta["bottleneck_regime"]).inc()
    METRIC_LATENCY.observe(exec_time_ms / 1000.0)
    METRIC_HYBRID_SCORE.observe(hybrid_score)

    return DiagnosticResponse(
        event_id=event_id,
        timestamp_utc=timestamp_utc,
        is_anomaly=is_anomaly,
        severity=severity,
        hybrid_score=round(hybrid_score, 4),
        model_scores=ModelScores(
            lgbm_score=round(s_lgbm, 4),
            lstm_ae_score=round(s_lstm, 4),
            topology_score=round(s_graph, 4),
            hybrid_score=round(hybrid_score, 4),
        ),
        top_shap_drivers=top_shap_drivers,
        topology_metadata=TopologyMetadata(
            in_routing_cycle=topo_meta["in_routing_cycle"],
            cycle_path=topo_meta["cycle_path"],
            target_fan_in_degree=topo_meta["target_fan_in_degree"],
            source_fan_out_degree=topo_meta["source_fan_out_degree"],
            bottleneck_regime=topo_meta["bottleneck_regime"],
            target_pagerank=topo_meta.get("target_pagerank", 0.0),
            source_pagerank=topo_meta.get("source_pagerank", 0.0),
        ),
        runbook_recommendations=runbook_recs,
        execution_time_ms=exec_time_ms,
    )


@app.post("/diagnose", response_model=DiagnosticResponse)
async def diagnose(event: TelemetryEventInput):
    """Diagnose a single 10-column telemetry event with tri-modal scoring, SHAP, and RAG."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing. Please retry.")
    try:
        return _process_single_diagnostic(event)
    except Exception as e:
        METRIC_REQUESTS.labels(status="error", endpoint="/diagnose").inc()
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=DiagnosticResponse)
async def predict_alias(event: TelemetryEventInput):
    """Alias for /diagnose endpoint."""
    return await diagnose(event)


@app.post("/diagnose/batch", response_model=List[DiagnosticResponse])
async def diagnose_batch(batch: TelemetryBatchInput):
    """Diagnose a batch of distributed grid telemetry events."""
    if not state.is_ready:
        raise HTTPException(status_code=503, detail="System initializing. Please retry.")
    results = []
    for evt in batch.events:
        results.append(_process_single_diagnostic(evt))
    return results
