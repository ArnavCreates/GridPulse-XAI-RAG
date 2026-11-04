# GridPulse-XAI-RAG

[![CI Test Suite](https://img.shields.io/badge/pytest-20%20passed-brightgreen.svg)](#test-suite-and-verification)
[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](#installation)
[![Inference Device](https://img.shields.io/badge/device-CPU%20optimized-orange.svg)](#latency-telemetry)
[![Embedding Model](https://img.shields.io/badge/embeddings-all--MiniLM--L6--v2-purple.svg)](#local-runbook-rag-engine)
[![Observability](https://img.shields.io/badge/metrics-Prometheus%20native-red.svg)](#api-endpoints-and-observability)

**GridPulse-XAI-RAG** is a self-contained project exploring anomaly attribution and runbook retrieval for distributed cloud and compute-cluster telemetry. It combines tabular gradient boosting (`LightGBM`), temporal reconstruction (`PyTorch LSTM Autoencoder`), and graph analysis (`NetworkX`) for routing cycles and fan-in/fan-out bottlenecks. `SHAP` `TreeExplainer` provides feature attributions, while local CPU `ChromaDB` retrieval uses `all-MiniLM-L6-v2` to find relevant runbooks. An asynchronous `FastAPI` service exposes the diagnostic endpoints and a `Prometheus` `/metrics` endpoint. The benchmark uses synthetic telemetry.

The service accepts events describing compute nodes, storage elements, and network links. For each event, it combines tabular, temporal, and graph signals, returns signed feature attributions, retrieves relevant runbooks, and records the result in a tamper-evident hash chain. The same API exports request and latency metrics through Prometheus.

---

## Table of Contents

- [System Architecture](#system-architecture)
- [Distributed Telemetry Schema](#distributed-telemetry-schema)
- [Tri-Modal Ensemble Methodology](#tri-modal-ensemble-methodology)
- [Explainable AI (SHAP Attributions)](#explainable-ai-shap-attributions)
- [Local Runbook RAG Engine](#local-runbook-rag-engine)
- [Cryptographic Audit Logger](#cryptographic-audit-logger)
- [Empirical Benchmark Results](#empirical-benchmark-results)
- [API Endpoints and Observability](#api-endpoints-and-observability)
- [Repository Structure](#repository-structure)
- [Installation and Setup](#installation-and-setup)
- [Reproduction and Usage](#reproduction-and-usage)
- [Test Suite and Verification](#test-suite-and-verification)

---

## System Architecture

GridPulse combines three complementary anomaly detection paradigms into a calibrated convex ensemble, accompanied by interpretability and automated operational response components:

```
                            +-----------------------------------+
                            | Distributed Grid Telemetry Stream |
                            +-----------------+-----------------+
                                              |
                                              v
                            +-----------------------------------+
                            | Feature Extractor (14 Channels)   |
                            +--------+-----------------+--------+
                                     |                 |
            +------------------------+                 +------------------------+
            |                                  |                                |
            v                                  v                                v
+-----------------------+          +-----------------------+        +-----------------------+
| LightGBM Tabular      |          | PyTorch LSTM-AE       |        | NetworkX Graph        |
| Gradient Booster      |          | Temporal Sequence     |        | Topology Analyzer     |
+-----------+-----------+          +-----------+-----------+        +-----------+-----------+
            |                                  |                                |
            | Score s_lgbm                     | Score s_lstm                   | Score s_graph
            |                                  |                                |
            +------------------------+         |         +----------------------+
                                     |         |         |
                                     v         v         v
                            +-----------------------------------+
                            | Hybrid Convex Combiner (s_hybrid) |
                            +-----------------+-----------------+
                                              |
                         [ If Anomaly Flagged / Above Threshold ]
                                              |
                         +--------------------+--------------------+
                         |                                         |
                         v                                         v
            +-------------------------+               +-------------------------+
            | SHAP TreeExplainer      |               | ChromaDB Vector RAG     |
            | Signed Feature Impacts  |               | all-MiniLM-L6-v2 Embeds |
            +------------+------------+               +------------+------------+
                         |                                         |
                         +--------------------+--------------------+
                                              |
                                              v
                            +-----------------------------------+
                            | SHA-256 Tamper-Evident Hash Chain |
                            | JSONL Diagnostic Audit Log        |
                            +-----------------+-----------------+
                                              |
                                              v
                            +-----------------------------------+
                            | Async FastAPI & Prometheus Exporter|
                            +-----------------------------------+
```

---

## Distributed Telemetry Schema

The input data plane consumes a 10-column distributed telemetry schema capturing protocol metadata, volumetric throughput, compute saturation, and queue states:

| Column Name | Data Type | Physical Description |
| :--- | :--- | :--- |
| `window_step` | Integer | Discrete time window or synchronization epoch counter |
| `protocol_type` | Categorical | Grid protocol: `XROOTD_STREAM`, `GRID_FTP`, `HEARTBEAT_SYNC`, `RPC_CONTROL`, `CACHE_REPLICATION` |
| `payload_mb` | Float | Transfer payload volume in megabytes |
| `source_node` | Categorical | Originating node identifier (e.g., `NODE_T1_001` .. `NODE_T1_060`) |
| `src_cpu_load_prev` | Float | Source node CPU load percentage at previous step ($t - 1$) |
| `src_cpu_load_curr` | Float | Source node CPU load percentage at current step ($t$) |
| `target_node` | Categorical | Destination node or storage pool (e.g., `SE_POOL_001` .. `SE_POOL_060`) |
| `dst_queue_depth_prev`| Float | Target node I/O pending queue depth at previous step ($t - 1$) |
| `dst_queue_depth_curr`| Float | Target node I/O pending queue depth at current step ($t$) |
| `is_anomaly` | Integer | Ground-truth anomaly label ($0 = \text{nominal}, 1 = \text{anomaly}$) |

### Engineered Kinematic & Surge Features (14 Channels)

From these 10 raw attributes, `TelemetryFeatureExtractor` computes 14 engineered analytical features:

1. `cpu_load_delta`: Current minus previous source CPU load
2. `queue_depth_delta`: Current minus previous destination queue depth
3. `queue_surge_ratio`: Current-to-previous queue depth ratio
4. `cpu_load_ratio`: Current-to-previous source CPU load ratio
5. `bandwidth_pressure`: Payload multiplied by current CPU fraction
6. `queue_to_payload_ratio`: Current queue depth divided by payload
7. `payload_mb`: Raw payload magnitude
8. `src_cpu_load_curr`: Current source CPU load
9. `dst_queue_depth_curr`: Current destination queue depth
10. `protocol_xrootd`: One-hot indicator for XRootD streams
11. `protocol_gridftp`: One-hot indicator for GridFTP transfers
12. `protocol_heartbeat`: One-hot indicator for heartbeat synchronization
13. `protocol_rpc`: One-hot indicator for RPC control traffic
14. `protocol_replication`: One-hot indicator for cache replication

The temporal model builds six-event windows independently per `source_node`, sorted by `window_step`. Short histories are left-padded with the earliest available event; windows never include another node's events.

---

## Tri-Modal Ensemble Methodology

The detection architecture processes distributed telemetry across three complementary, orthogonal failure regimes:

1. **LightGBM Tabular Gradient Booster (Mode A Specialist ~35% of anomalies):**
   - Detects extreme metric boundaries, instantaneous queue spikes (`dst_queue_depth_curr`), and severe compute surges (`src_cpu_load_curr > 90%`, `payload_mb > 280 MB`).
   - Single-row LightGBM only partially detects Modes B, C, and D, where individual row marginals remain strictly within nominal bounds.
   - Outputs calibrated tabular probability $s_{\text{lgbm}} \in [0, 1]$.

2. **PyTorch LSTM Autoencoder (Mode B Specialist ~30% of anomalies):**
   - Architecture: Sequence-to-sequence autoencoder with continuous feature standardization $\frac{X - \mu}{\sigma + \epsilon}$, 1-layer LSTM encoder ($6 \to 32 \to 16$), latent bottleneck, and LSTM decoder ($16 \to 32 \to 6$).
   - Trained strictly on nominal baseline sliding sequences ($L = 6$).
   - Detects stealth temporal sequence drift and multi-step sawtooth oscillations where marginal values look completely nominal on any single event but exhibit non-stationary sequence drift across the 6-step window.
   - Outputs standardized reconstruction anomaly score $s_{\text{lstm}} \in [0, 1]$.

3. **NetworkX Directed Graph Topology Analyzer (Modes C & D Specialist ~35% of anomalies):**
   - Tracks dynamic multi-hop routing interactions across the 60-node grid cluster over discrete temporal windows.
   - **Mode C (Stealth Routing Cycles ~18%):** Executes bounded cycle detection ($3 \le \text{length} \le 4$), isolating directed routing loops ($A \to B \to C \to A$) that bypass tabular thresholds.
   - **Mode D (Stealth Fan-In Incast ~17%):** Measures dynamic in-degree distributions to identify synchronized fan-in bursts ($\ge 7$ distinct compute nodes simultaneously saturating one storage pool).
   - Incorporates degree congestion and volumetric edge priors, outputting topological anomaly score $s_{\text{graph}} \in [0, 1]$.

4. **Fitted Logistic Stacking Combiner:**
   A logistic meta-learner is fitted on the training partition with the three component scores and their maximum:
   $$
   s_{\text{stack}}(x) = \sigma\left(w_0 + w_1 s_{\text{lgbm}}(x) + w_2 s_{\text{lstm}}(x) + w_3 s_{\text{graph}}(x) + w_4 \max(s_{\text{lgbm}}, s_{\text{lstm}}, s_{\text{graph}})\right)
   $$
   The configured prior blend then combines the stacker with the normalized component-weighted score: $s_{\text{hybrid}} = 0.8s_{\text{stack}} + 0.2(0.45s_{\text{lgbm}} + 0.30s_{\text{lstm}} + 0.25s_{\text{graph}})$. The alert threshold is selected using the training scores. The current seeded run fitted $(w_0, w_1, w_2, w_3, w_4) = (-7.78, 13.75, 5.65, 13.25, -4.09)$ and threshold $0.26$.
   
   Severity classification:
   - $s_{\text{hybrid}} \ge 0.80 \implies \textbf{CRITICAL}$
   - $s_{\text{hybrid}} \ge 0.65 \implies \textbf{HIGH}$
   - $s_{\text{hybrid}} \ge 0.50 \implies \textbf{MEDIUM}$
   - $s_{\text{hybrid}} < 0.50 \implies \textbf{NOMINAL}$

---

## Explainable AI (SHAP Attributions)

GridPulse incorporates per-alert local explanations using `shap.TreeExplainer`:
- Calculates signed feature contributions $\phi_i$ in **log-odds units** for each inference request.
- Selects the top-4 absolute drivers ($|\phi_i|$), returning their observed values, signed impacts, and direction (`+` indicating push toward anomaly, `-` indicating push toward nominal).
- Makes alerts easier to inspect by showing the strongest measured feature contributions (for example, a queue increase or a CPU-load change).

---

## Local Runbook RAG Engine

Operational responses are automated using a vector retrieval engine:
- **Vector Store:** ChromaDB running locally on CPU.
- **Embedding Model:** `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors, fully offline).
- **Knowledge Base:** 6 structured markdown runbooks (`runbooks/`) indexed as 18 actionable procedure chunks (3 per runbook). Metadata and symptom signature headings are excluded from retrieved chunks; runbook ID, failure mode, and severity remain attached as structured context:
  1. `01_xrootd_buffer_saturation.md` (RB-GRID-001: Mode A Queue Overflow)
  2. `02_routing_cycle_storm.md` (RB-GRID-002: Mode C Routing Loops)
  3. `03_fan_in_incast_congestion.md` (RB-GRID-003: Mode D Incast Bottlenecks)
  4. `04_fan_out_replication_flood.md` (RB-GRID-004: Mode D Outbound Bursts)
  5. `05_worker_cpu_thermal_stall.md` (RB-GRID-005: Mode B Progressive Drift)
  6. `06_heartbeat_desync_partition.md` (RB-GRID-006: Consensus Partitions)
- Generates context-aware diagnostic queries combining top-4 signed SHAP features, temporal sequence drift status, and graph topological regimes, then returns the top-2 distinct runbooks with actionable procedure text.

---

## Cryptographic Audit Logger

Every diagnostic decision is logged to a tamper-evident audit ledger (`logs/diagnostic_audit.jsonl`):
- Each record contains: timestamp, event ID, anonymized source and target identifiers, composite score, severity, top SHAP drivers, retrieved runbooks, and previous record SHA-256 hash.
- Current block hash is computed as:
  $$
  \text{Hash}_k = \text{SHA-256}(\text{CanonicalJSON}(\text{Record}_k))
  $$
- The record includes the prior hash. `AuditLogger.verify_chain()` checks continuity and detects altered entries.

---

## Empirical Benchmark Results

Evaluated on a self-contained 15,000-event synthetic distributed cloud & compute cluster telemetry benchmark across 60 nodes with 4 orthogonal failure modes (`Mode A` tabular surge, `Mode B` stealth temporal sequence drift, `Mode C` directed routing cycle, `Mode D` fan-in incast congestion). All metrics are empirical, evaluated on the held-out test partition of 3,750 events, and saved directly in [`assets/benchmark_metrics.json`](assets/benchmark_metrics.json). The topology analyzer builds separate train and test graphs, and the temporal model forms node-specific chronological windows in both partitions.

### Component Ablation Comparison

| Component Model | ROC-AUC | PR-AUC | F1-Score | Precision | Recall | Primary Specialization |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **LightGBM (Tabular)** | 0.91024 | 0.77355 | 0.66995 | 0.96226 | 0.51385 | Mode A Tabular Resource Surge |
| **LSTM Autoencoder (Temporal)** | 0.78005 | 0.28306 | 0.37792 | 0.30060 | 0.50882 | Mode B Stealth Sequence Drift |
| **NetworkX Topology (Graph)** | 0.86670 | 0.77745 | 0.68760 | 1.00000 | 0.52393 | Modes C & D Cycles & Incast |
| **Stacked Logistic Hybrid Ensemble** | **0.96662** | **0.88245** | **0.82797** | **0.93082** | **0.74559** | Combined detector scores |

On this synthetic test split, the hybrid ensemble has the highest ROC-AUC, PR-AUC, and F1-score among the evaluated components. The individual models show different strengths across the failure modes, as reflected in the table.

### Runbook RAG Retrieval Benchmarks

Evaluated across 100 test anomalies spanning all 4 injected failure modes against operational recovery runbooks `RB-GRID-001` through `RB-GRID-006`:

| Retrieval Metric | Score | Evaluation Scope |
| :--- | :---: | :--- |
| **Top-1 Runbook Accuracy** | **100.00%** | Correct runbook family ranked #1 across the benchmark's four modes |
| **Top-2 Runbook Accuracy** | **100.00%** | A valid recovery runbook retrieved in Top-2 |

### CPU Latency Telemetry

Measured on the local CPU (mean of 20 benchmark runs; timings vary with host and load):

| Pipeline Stage | Mean Execution Time |
| :--- | :---: |
| **Tri-Modal Inference** (LightGBM + LSTM-AE + NetworkX) | 8.84 ms |
| **SHAP TreeExplainer Attribution** (Top-4 channels) | 5.70 ms |
| **ChromaDB Vector Runbook Retrieval** (Cosine similarity) | 79.19 ms |
| **Complete End-to-End Diagnostic Request** | **93.72 ms** |

### Visual Deliverables

- **Model Ablation Curves:** [`assets/model_ablation_roc_pr.png`](assets/model_ablation_roc_pr.png) displays ROC and Precision-Recall ablation curves across all individual detectors and the hybrid ensemble.
- **SHAP & Topology Diagnostics:** [`assets/shap_and_topology_diagnostics.png`](assets/shap_and_topology_diagnostics.png) illustrates signed feature importance distributions and cluster routing topologies.

---

## API Endpoints and Observability

The microservice exposes an asynchronous FastAPI application with the following endpoints:

| Method | Path | Summary | Description |
| :--- | :--- | :--- | :--- |
| `GET` | `/` | Root Info | Service health status, version, and 10-column schema summary |
| `GET` | `/health` | Health Check | Validates model readiness and indexed runbook counts |
| `GET` | `/metrics` | Prometheus | Real-time Prometheus metrics exporter |
| `POST`| `/diagnose` | Single Diagnosis | Processes 1 telemetry event; returns scores, SHAP, and RAG |
| `POST`| `/predict` | Predict Alias | Alias pointing to `/diagnose` |
| `POST`| `/diagnose/batch` | Batch Diagnosis | High-throughput batch processing for distributed telemetry |

`/diagnose` appends each event to that source node's six-step history before temporal scoring. In deterministic mode, event IDs and timestamps derive from the event payload and `window_step`. Batch size, CORS origins, Prometheus names, histogram buckets, and the audit log path come from `config/config.yaml`.

### Prometheus Observability Metrics

- `gridpulse_requests_total`: Counter tracking total processed requests partitioned by endpoint and status (`success`, `error`).
- `gridpulse_anomalies_detected_total`: Counter tracking anomaly detections partitioned by severity (`CRITICAL`, `HIGH`, `MEDIUM`) and bottleneck regime.
- `gridpulse_inference_latency_seconds`: Histogram measuring end-to-end inference and RAG latency across standard duration buckets.
- `gridpulse_hybrid_score_summary`: Summary tracking the distribution and quantiles of composite anomaly scores.

---

## Repository Structure

```
GridPulse-XAI-RAG/
├── .github/
│   └── workflows/
│       └── ci.yml                          # Python 3.12 install and pytest workflow
├── assets/
│   ├── benchmark_metrics.json              # Empirical evaluation metrics & CPU latencies
│   ├── model_ablation_roc_pr.png           # ROC and PR ablation curves (4 detectors)
│   ├── sample_diagnostic_report.json       # Example JSON payload from /diagnose
│   └── shap_and_topology_diagnostics.png   # SHAP attribution summary & cluster topology
├── config/
│   └── config.yaml                         # 114 leaf parameters across 10 top-level sections
├── runbooks/
│   ├── 01_xrootd_buffer_saturation.md      # RB-GRID-001 (Mode A: Queue Saturation)
│   ├── 02_routing_cycle_storm.md           # RB-GRID-002 (Mode C: 4-Hop Routing Storms)
│   ├── 03_fan_in_incast_congestion.md      # RB-GRID-003 (Mode D: Fan-In Incast)
│   ├── 04_fan_out_replication_flood.md     # RB-GRID-004 (Mode D: Fan-Out Burst)
│   ├── 05_worker_cpu_thermal_stall.md      # RB-GRID-005 (Mode B: Progressive Drift)
│   └── 06_heartbeat_desync_partition.md    # RB-GRID-006 (Consensus & Network Partitions)
├── scripts/
│   └── run_pipeline_and_benchmark.py       # Full benchmark generation pipeline script
├── src/
│   ├── __init__.py
│   ├── config_loader.py                    # YAML configuration parser & validator
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py                         # FastAPI microservice with Prometheus exporter
│   │   └── schemas.py                      # Pydantic v2 schemas for events and diagnostics
│   ├── data/
│   │   ├── __init__.py
│   │   ├── feature_engineering.py          # 14-channel kinematic and surge feature extractor
│   │   └── telemetry_generator.py          # 10-column distributed telemetry simulator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── hybrid_ensemble.py              # Calibrated convex combiner & evaluator
│   │   ├── lgbm_detector.py                # LightGBM tabular detector
│   │   ├── lstm_autoencoder.py             # PyTorch sequence-to-sequence autoencoder
│   │   └── topology_analyzer.py            # NetworkX graph topology and cycle analyzer
│   └── xai_rag/
│       ├── __init__.py
│       ├── audit_logger.py                 # SHA-256 tamper-evident cryptographic logger
│       ├── runbook_rag.py                  # Local ChromaDB vector runbook engine
│       └── shap_attributor.py              # SHAP TreeExplainer signed local attribution
├── tests/
│   ├── conftest.py                         # Pytest configuration & path imports
│   ├── test_config_and_schema.py           # Configuration and schema unit tests (4 tests)
│   ├── test_fastapi_prometheus.py          # FastAPI, Prometheus, and audit tests (8 tests)
│   ├── test_models_and_topology.py         # LightGBM, LSTM, and NetworkX tests (5 tests)
│   └── test_shap_and_rag.py                # SHAP, ChromaDB, and hash chain tests (3 tests)
├── .gitignore                              # Comprehensive clean exclusions
├── LICENSE                                 # MIT License
├── pyproject.toml                          # Modern Python packaging configuration
├── requirements.txt                        # Minimum-version dependencies
└── README.md                               # Project documentation
```

---

## Installation and Setup

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Local CPU environment (no GPU required)

### Setup Environment
```bash
# Navigate to project root
cd GridPulse-XAI-RAG

# Create and activate virtual environment
python -m venv .venv
# On Linux/macOS:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install the declared minimum-version dependencies
pip install -r requirements.txt
```

---

## Reproduction and Usage

### 1. Run Pipeline and Benchmark
To train models on a synthetic 15,000-event telemetry dataset, evaluate ablation metrics, execute SHAP attributions and ChromaDB vector queries, and regenerate all visual assets:
```bash
python scripts/run_pipeline_and_benchmark.py
```
This updates `assets/benchmark_metrics.json`, `assets/sample_diagnostic_report.json`, `assets/model_ablation_roc_pr.png`, and `assets/shap_and_topology_diagnostics.png`.

### 2. Launch the FastAPI Microservice
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
Interactive Swagger API documentation will be accessible at `http://localhost:8000/docs`.

### 3. Example Telemetry Diagnosis Request
Send a telemetry event to `/diagnose`:
```bash
curl -X POST "http://localhost:8000/diagnose" \
  -H "Content-Type: application/json" \
  -d '{
    "window_step": 142,
    "protocol_type": "XROOTD_STREAM",
    "payload_mb": 512.45,
    "source_node": "NODE_T1_014",
    "src_cpu_load_prev": 48.0,
    "src_cpu_load_curr": 96.5,
    "target_node": "SE_POOL_007",
    "dst_queue_depth_prev": 12.0,
    "dst_queue_depth_curr": 88.5,
    "is_anomaly": 1
  }'
```

The generated benchmark report uses this shape:
```json
{
  "event_id": "evt_diag_benchmark_001",
  "timestamp_utc": "2026-01-01T00:02:22.000+00:00",
  "is_anomaly": true,
  "severity": "CRITICAL",
  "hybrid_score": 0.9928,
  "model_scores": {
    "lightgbm": 0.9198,
    "lstm_autoencoder": 1.0,
    "networkx_topology": 1.0
  },
  "top_shap_drivers": [
    {"feature": "payload_mb", "observed_value": 512.45, "shap_impact": 3.5215, "impact_units": "log-odds", "direction": "+"},
    {"feature": "queue_depth_delta", "observed_value": 76.5, "shap_impact": 0.2721, "impact_units": "log-odds", "direction": "+"}
  ],
  "runbook_recommendations": [
    {
      "runbook_id": "RB-GRID-001",
      "section_title": "Diagnostic Verification",
      "severity": "CRITICAL / HIGH",
      "similarity_score": 0.77,
      "recommended_action": "1. **Queue Gradient Analysis:** Check if dst_queue_depth_curr - dst_queue_depth_prev exceeds threshold within a single telemetry window step."
    }
  ],
  "execution_latency_ms": {"end_to_end_total_ms": 75.72}
}
```

---

## Test Suite and Verification

The repository contains an automated test suite spanning 20 unit and API route tests across 4 test modules:

```bash
python -m pytest -p no:cacheprovider
```

Execution output (Python 3.12.16, Windows; full suite):
```
============================= test session starts =============================
platform win32 -- Python 3.12.16, pytest-9.2.0, pluggy-1.7.0
rootdir: GridPulse-XAI-RAG
configfile: pyproject.toml

tests/test_config_and_schema.py::test_config_loader_and_leaf_parameters PASSED [  5%]
tests/test_config_and_schema.py::test_telemetry_schema_validation PASSED         [ 10%]
tests/test_config_and_schema.py::test_telemetry_generator_shapes_and_modes PASSED [ 15%]
tests/test_config_and_schema.py::test_feature_engineering_14_channels PASSED    [ 20%]
tests/test_fastapi_prometheus.py::test_root_endpoint PASSED                     [ 25%]
tests/test_fastapi_prometheus.py::test_health_endpoint PASSED                   [ 30%]
tests/test_fastapi_prometheus.py::test_diagnose_nominal_event PASSED             [ 35%]
tests/test_fastapi_prometheus.py::test_diagnose_anomaly_event PASSED             [ 40%]
tests/test_fastapi_prometheus.py::test_predict_alias PASSED                     [ 45%]
tests/test_fastapi_prometheus.py::test_diagnose_batch PASSED                     [ 50%]
tests/test_fastapi_prometheus.py::test_prometheus_metrics PASSED                 [ 55%]
tests/test_fastapi_prometheus.py::test_audit_log_written_and_verified PASSED     [ 60%]
tests/test_models_and_topology.py::test_lgbm_detector PASSED                     [ 65%]
tests/test_models_and_topology.py::test_lstm_autoencoder_architecture PASSED    [ 70%]
tests/test_models_and_topology.py::test_lstm_autoencoder_detector PASSED        [ 75%]
tests/test_models_and_topology.py::test_networkx_topology_analyzer PASSED       [ 80%]
tests/test_models_and_topology.py::test_hybrid_ensemble_combiner PASSED         [ 85%]
tests/test_shap_and_rag.py::test_shap_attributor PASSED                         [ 90%]
tests/test_shap_and_rag.py::test_runbook_rag_indexing_and_retrieval PASSED     [ 95%]
tests/test_shap_and_rag.py::test_audit_logger_hash_chain_and_tamper_detection PASSED [100%]

============================= 20 passed in 6.63s =============================
```

---

## References & Acknowledgments

Feature attribution follows S. M. Lundberg and S.-I. Lee, [“A Unified Approach to Interpreting Model Predictions”](https://papers.nips.cc/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html), NeurIPS 2017. Runbook retrieval uses [Chroma](https://github.com/chroma-core/chroma); the service and telemetry schemas use FastAPI, Pydantic, and Prometheus client libraries.
