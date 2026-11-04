"""Unit tests for SHAP feature attribution, ChromaDB Runbook RAG, and Audit Logger."""

import json
import hashlib
from pathlib import Path
import numpy as np
import pytest

from src.data.feature_engineering import FEATURE_NAMES, TelemetryFeatureExtractor
from src.data.telemetry_generator import generate_single_event, generate_telemetry_dataset
from src.models.lgbm_detector import LGBMAnomalyDetector
from src.xai_rag.audit_logger import AuditLogger
from src.xai_rag.runbook_rag import RunbookRAGEngine
from src.xai_rag.shap_attributor import ShapAttributor


class StubEncoder:
    def encode(self, texts, **kwargs):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append([value + 1 for value in digest])
        return np.asarray(vectors, dtype=np.float32)


class StubExplainer:
    def shap_values(self, features, check_additivity=False):
        values = np.arange(features.shape[1], dtype=np.float32)
        return np.tile(values, (features.shape[0], 1))


class StubCollection:
    def __init__(self):
        self.rows = []

    def add(self, ids, embeddings, metadatas, documents):
        self.rows.extend(zip(ids, embeddings, metadatas, documents))

    def get(self):
        return {"ids": [row[0] for row in self.rows]}

    def query(self, query_embeddings, n_results, include):
        query = np.asarray(query_embeddings[0], dtype=np.float32)
        query_norm = np.linalg.norm(query) or 1.0
        ranked = []
        for _, embedding, metadata, document in self.rows:
            vector = np.asarray(embedding, dtype=np.float32)
            cosine = float(np.dot(query, vector) / (query_norm * (np.linalg.norm(vector) or 1.0)))
            ranked.append((2.0 * (1.0 - cosine), metadata, document))
        ranked.sort(key=lambda row: row[0])
        ranked = ranked[:n_results]
        return {
            "documents": [[row[2] for row in ranked]],
            "metadatas": [[row[1] for row in ranked]],
            "distances": [[row[0] for row in ranked]],
        }


class StubVectorClient:
    def delete_collection(self, name):
        return None

    def create_collection(self, name, metadata):
        return StubCollection()


@pytest.fixture(scope="module")
def trained_lgbm_and_extractor():
    """Fixture providing trained LightGBM model and feature extractor."""
    df = generate_telemetry_dataset(n_samples=250, seed=42)
    extractor = TelemetryFeatureExtractor(eps=1e-4, use_scaler=False)
    X = extractor.transform_df(df).values
    y = df["is_anomaly"].values

    detector = LGBMAnomalyDetector(n_estimators=25, num_leaves=15, max_depth=4)
    detector.fit(X, y)
    return detector, extractor


def test_shap_attributor(trained_lgbm_and_extractor):
    """Verify SHAP attributor returns top-k features with valid directions and impacts."""
    detector, extractor = trained_lgbm_and_extractor
    attributor = ShapAttributor(
        lgbm_model=detector.get_model(),
        feature_names=FEATURE_NAMES,
        top_k=4,
        explainer=StubExplainer(),
    )

    event = generate_single_event(is_anomaly=True, mode="MODE_A")
    feat_df = extractor.transform_single(event)
    attributions = attributor.explain_instance(feat_df.values[0], observed_dict=event)

    assert len(attributions) == 4
    for attr in attributions:
        assert attr["feature"] in FEATURE_NAMES
        assert isinstance(attr["shap_impact"], float)
        assert attr["impact_units"] == "log-odds"
        assert attr["direction"] in ("+", "-")
        assert attr["abs_impact"] >= 0.0

    # Ensure ranked in descending order of absolute impact
    assert attributions[0]["abs_impact"] >= attributions[1]["abs_impact"]


def test_runbook_rag_indexing_and_retrieval(tmp_path):
    """Verify ChromaDB runbook RAG indexes actionable sections and retrieves procedures."""
    rag_engine = RunbookRAGEngine(
        embedding_model_name="all-MiniLM-L6-v2",
        collection_name="test_runbooks_collection",
        top_k=2,
        encoder=StubEncoder(),
        client=StubVectorClient(),
    )
    indexed_count = rag_engine.index_runbooks()
    assert indexed_count >= 18
    split_chunks = rag_engine._split_action_text(
        "\n".join(f"- Recovery check {index}: verify queue and service state." for index in range(60))
    )
    assert len(split_chunks) > 1
    assert all(chunk.startswith("- Recovery check") for chunk in split_chunks)

    # Test retrieval for Mode A: XRootD Buffer Saturation
    sample_drivers = [
        {"feature": "dst_queue_depth_curr", "shap_impact": 0.45, "direction": "+"},
        {"feature": "payload_mb", "shap_impact": 0.35, "direction": "+"},
    ]
    recs = rag_engine.retrieve_recommendations(
        protocol_type="XROOTD_STREAM",
        shap_drivers=sample_drivers,
        bottleneck_regime="NOMINAL",
        top_k=2,
    )

    assert len(recs) == 2
    for r in recs:
        assert "runbook_id" in r
        assert "section_title" in r
        assert 0.0 <= r["similarity_score"] <= 1.0
        assert len(r["recommended_action"]) > 0
        assert r["section_title"] in {
            "Diagnostic Verification",
            "Remediation & Recovery Procedure",
            "Prevention & Hardening",
        }
        assert "Metadata" not in r["recommended_action"]
        assert "## Metadata" not in r["full_section_snippet"]
        assert "## Symptom Signature" not in r["full_section_snippet"]


def test_audit_logger_hash_chain_and_tamper_detection(tmp_path):
    """Verify SHA-256 hash chaining, log appending, and tamper resistance detection."""
    log_file = tmp_path / "test_audit.jsonl"
    logger = AuditLogger(log_path=log_file, deterministic_mode=True)

    # Append 3 diagnostic log entries
    logger.log_diagnostic_event(
        event_id="evt_001",
        source_node="NODE_T1_001",
        target_node="SE_POOL_001",
        hybrid_score=0.88,
        is_anomaly=True,
        severity="CRITICAL",
        top_shap_drivers=["dst_queue_depth_curr", "payload_mb"],
        retrieved_runbook_ids=["RB-GRID-001"],
        window_step=7,
    )

    logger.log_diagnostic_event(
        event_id="evt_002",
        source_node="NODE_T1_002",
        target_node="SE_POOL_002",
        hybrid_score=0.15,
        is_anomaly=False,
        severity="NOMINAL",
        top_shap_drivers=["src_cpu_load_curr"],
        retrieved_runbook_ids=["RB-GRID-005"],
        window_step=8,
    )

    logger.log_diagnostic_event(
        event_id="evt_003",
        source_node="NODE_T1_005",
        target_node="NODE_T1_006",
        hybrid_score=0.92,
        is_anomaly=True,
        severity="CRITICAL",
        top_shap_drivers=["protocol_rpc"],
        retrieved_runbook_ids=["RB-GRID-002"],
        window_step=9,
    )

    repeat_file = tmp_path / "repeat_audit.jsonl"
    repeat_logger = AuditLogger(log_path=repeat_file, deterministic_mode=True)
    repeat_logger.log_diagnostic_event(
        event_id="evt_001",
        source_node="NODE_T1_001",
        target_node="SE_POOL_001",
        hybrid_score=0.88,
        is_anomaly=True,
        severity="CRITICAL",
        top_shap_drivers=["dst_queue_depth_curr", "payload_mb"],
        retrieved_runbook_ids=["RB-GRID-001"],
        window_step=7,
    )
    assert log_file.read_text(encoding="utf-8").splitlines()[0] == repeat_file.read_text(encoding="utf-8").splitlines()[0]

    # Verify uncorrupted chain
    is_valid, count, err = AuditLogger.verify_chain(log_file)
    assert is_valid is True
    assert count == 3
    assert err is None

    # Tamper with the second entry
    lines = log_file.read_text(encoding="utf-8").strip().split("\n")
    second_entry = json.loads(lines[1])
    second_entry["hybrid_score"] = 0.99  # Tampered score
    lines[1] = json.dumps(second_entry)
    tampered_log = tmp_path / "tampered_audit.jsonl"
    tampered_log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Verify tamper is flagged immediately
    is_valid_tampered, count_tampered, err_msg = AuditLogger.verify_chain(tampered_log)
    assert is_valid_tampered is False
    assert "Tamper detected" in err_msg or "Hash break" in err_msg
