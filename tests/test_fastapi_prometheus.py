"""End-to-end tests for FastAPI endpoints and Prometheus metrics."""

import asyncio
import selectors
import time
from typing import Any

import httpx
import pytest

import src.api.main as api_main
from src.config_loader import load_config
from src.data.feature_engineering import TelemetryFeatureExtractor
from src.data.telemetry_generator import generate_single_event
from src.xai_rag.audit_logger import AuditLogger


class DirectASGIClient:
    def __init__(self, app):
        self.app = app
        self.loop = InMemoryEventLoop()

    def request(self, method: str, path: str, payload: Any = None):
        async def send_request():
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
                return await client.request(method.upper(), path, json=payload)

        return self.loop.run_until_complete(send_request())

    def get(self, path: str):
        return self.request("GET", path)

    def post(self, path: str, json: Any = None):
        return self.request("POST", path, json)

    def close(self):
        self.loop.close()


class InMemoryEventLoop(asyncio.SelectorEventLoop):
    def __init__(self):
        super().__init__(selector=NoSocketSelector())

    def _make_self_pipe(self):
        self._ssock = None
        self._csock = None
        self._internal_fds = 0

    def _close_self_pipe(self):
        return None

    def _write_to_self(self):
        return None


class NoSocketSelector(selectors.SelectSelector):
    def select(self, timeout=None):
        if not self.get_map():
            time.sleep(0.001 if timeout is None else max(0.0, timeout))
            return []
        return super().select(timeout)


class FakeLGBM:
    is_fitted = True

    def predict_proba(self, features):
        return [0.95 if float(features[0, 6]) > 100 else 0.10]


class FakeLSTM:
    is_fitted = True

    def predict_score(self, sequences):
        value = 0.90 if float(sequences[:, :, 4].max()) > 100 else 0.20
        return [value]


class FakeTopology:
    def analyze_event(self, **kwargs):
        return 0.15, {
            "in_routing_cycle": False,
            "cycle_path": [],
            "target_fan_in_degree": 2,
            "source_fan_out_degree": 1,
            "bottleneck_regime": "NOMINAL",
            "target_pagerank": 0.02,
            "source_pagerank": 0.01,
        }


class FakeEnsemble:
    alert_threshold = 0.50

    def combine_scores(self, lgbm, lstm, topology):
        return float((lgbm + lstm + topology) / 3.0)

    def classify_severity(self, score):
        if score >= 0.80:
            return True, "CRITICAL"
        if score >= 0.65:
            return True, "HIGH"
        if score >= 0.50:
            return True, "MEDIUM"
        return False, "NOMINAL"


class FakeShap:
    def explain_instance(self, features_1d, observed_dict=None):
        names = ["payload_mb", "src_cpu_load_curr", "bandwidth_pressure", "queue_depth_delta"]
        return [
            {
                "feature": name,
                "observed_value": float(observed_dict.get(name, 0.0)),
                "shap_impact": 0.4 - index / 10,
                "direction": "+",
            }
            for index, name in enumerate(names)
        ]


class FakeRAG:
    is_indexed = True

    def retrieve_recommendations(self, **kwargs):
        return [
            {
                "runbook_id": f"RB-GRID-{index:03d}",
                "section_title": "Diagnostic Verification",
                "severity": "HIGH",
                "failure_mode": "Synthetic telemetry anomaly",
                "similarity_score": 0.90 - index / 10,
                "recommended_action": "Check the node queue and verify recovery.",
                "full_section_snippet": "Verify telemetry recovery.",
            }
            for index in (1, 2)
        ]


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    log_path = tmp_path_factory.mktemp("gridpulse_api") / "diagnostic_audit.jsonl"
    api_main.state.config = load_config()
    api_main.state.feature_extractor = TelemetryFeatureExtractor(
        config=api_main.state.config["feature_engineering"],
        schema_config=api_main.state.config["telemetry_schema"],
    )
    api_main.state.lgbm_model = FakeLGBM()
    api_main.state.lstm_model = FakeLSTM()
    api_main.state.topology_analyzer = FakeTopology()
    api_main.state.hybrid_ensemble = FakeEnsemble()
    api_main.state.shap_attributor = FakeShap()
    api_main.state.rag_engine = FakeRAG()
    api_main.state.runbooks_indexed_count = 18
    api_main.state.audit_logger = AuditLogger(
        log_path=log_path,
        deterministic_mode=bool(api_main.state.config["system"].get("deterministic_mode", False)),
    )
    api_main.state.node_history = {}
    api_main.state.is_ready = True
    test_client = DirectASGIClient(api_main.app)
    yield test_client
    test_client.close()


def test_root_endpoint(client):
    """Verify GET / and request validation errors."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "gridpulse-xai-service"
    assert data["status"] == "operational"
    assert len(data["schema_columns"]) == 10
    invalid = client.post("/diagnose", json={})
    assert invalid.status_code == 422


def test_health_endpoint(client):
    """Verify GET /health confirms model states and ChromaDB indexing."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["runbooks_indexed_count"] >= 6
    assert data["models_loaded"]["lightgbm"] is True
    assert data["models_loaded"]["lstm_autoencoder"] is True
    assert data["models_loaded"]["networkx_topology"] is True
    assert data["models_loaded"]["chromadb_rag"] is True


def test_diagnose_nominal_event(client):
    """Verify POST /diagnose correctly evaluates nominal telemetry."""
    nominal_event = generate_single_event(is_anomaly=False, seed=2026)
    response = client.post("/diagnose", json=nominal_event)
    assert response.status_code == 200
    data = response.json()
    assert "event_id" in data
    assert data["timestamp_utc"].startswith("2026-01-01T")
    assert "hybrid_score" in data
    assert "model_scores" in data
    assert len(data["top_shap_drivers"]) == 4
    assert data["top_shap_drivers"][0]["impact_units"] == "log-odds"
    assert len(data["runbook_recommendations"]) == 2
    assert "topology_metadata" in data
    assert data["execution_time_ms"] > 0.0
    history = api_main.state.node_history[nominal_event["source_node"]]
    assert len(history) == 1
    next_event = dict(nominal_event)
    next_event["window_step"] += 1
    next_event["payload_mb"] += 1.0
    assert client.post("/diagnose", json=next_event).status_code == 200
    history = api_main.state.node_history[nominal_event["source_node"]]
    assert len(history) == 2
    assert history[0].tolist() != history[1].tolist()


def test_diagnose_anomaly_event(client):
    """Verify POST /diagnose identifies anomaly drivers and produces stable identity."""
    anomaly_event = generate_single_event(is_anomaly=True, mode="MODE_A")
    response = client.post("/diagnose", json=anomaly_event)
    assert response.status_code == 200
    data = response.json()
    assert data["is_anomaly"] is True
    assert data["severity"] in ("CRITICAL", "HIGH", "MEDIUM")
    assert data["hybrid_score"] >= 0.50
    assert data["event_id"].startswith("evt_")
    driver_names = [d["feature"] for d in data["top_shap_drivers"]]
    assert any(
        f in driver_names
        for f in ["dst_queue_depth_curr", "queue_depth_delta", "src_cpu_load_curr", "payload_mb"]
    )


def test_predict_alias(client):
    """Verify POST /predict works as an alias to /diagnose."""
    event = generate_single_event(is_anomaly=False, seed=12)
    response = client.post("/predict", json=event)
    assert response.status_code == 200
    assert "hybrid_score" in response.json()


def test_diagnose_batch(client):
    """Verify POST /diagnose/batch evaluates valid batches and rejects oversized batches."""
    events = [
        generate_single_event(is_anomaly=False, seed=13),
        generate_single_event(is_anomaly=True, mode="MODE_A"),
    ]
    response = client.post("/diagnose/batch", json={"events": events})
    assert response.status_code == 200
    data = response.json()
    assert len(data) == 2
    assert all("hybrid_score" in item for item in data)
    cfg = api_main.state.config
    too_many = [events[0]] * (int(cfg["api"]["max_batch_size"]) + 1)
    invalid = client.post("/diagnose/batch", json={"events": too_many})
    assert invalid.status_code == 422


def test_prometheus_metrics(client):
    """Verify GET /metrics exposes Prometheus metrics."""
    response = client.get("/metrics")
    assert response.status_code == 200
    content = response.text
    assert "gridpulse_requests_total" in content
    assert "gridpulse_inference_latency_seconds" in content
    assert "gridpulse_hybrid_score_summary" in content


def test_audit_log_written_and_verified(client):
    """Verify generated entries pass hash-chain verification."""
    log_path = api_main.state.audit_logger.log_path
    assert log_path.exists()
    assert log_path.stat().st_size > 0
    is_valid, count, err = AuditLogger.verify_chain(log_path)
    assert is_valid is True
    assert count >= 3
    assert err is None
