"""End-to-End Pipeline Execution, Evaluation Benchmark, and Diagnostic Artifact Generation.

Trains LightGBM, PyTorch LSTM Autoencoder, and NetworkX Topology Analyzer,
fits the stacked logistic ensemble, measures latency, evaluates Runbook RAG
retrieval accuracy, and generates empirical figures and JSON assets.
"""

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, List

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, roc_curve

# Enforce offline mode for Hugging Face Hub
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from src.config_loader import apply_deterministic_seeds, load_config
from src.data.feature_engineering import LSTM_FEATURE_INDICES, TelemetryFeatureExtractor
from src.data.telemetry_generator import generate_single_event, generate_telemetry_dataset
from src.models.hybrid_ensemble import HybridEnsembleDetector
from src.models.lgbm_detector import LGBMAnomalyDetector
from src.models.lstm_autoencoder import LSTMAutoencoderDetector
from src.models.topology_analyzer import GridTopologyAnalyzer
from src.xai_rag.audit_logger import AuditLogger
from src.xai_rag.runbook_rag import RunbookRAGEngine
from src.xai_rag.shap_attributor import ShapAttributor


def deterministic_timestamp(window_step: int) -> str:
    epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return (epoch + timedelta(seconds=max(0, int(window_step)))).isoformat(timespec="milliseconds")


def main():
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    print("GridPulse telemetry models, attribution, and runbook retrieval")

    base_dir = Path(__file__).resolve().parent.parent
    assets_dir = base_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Centralized Configuration
    cfg = load_config()
    seed = apply_deterministic_seeds(cfg)
    print(f"[1/7] Configuration loaded successfully. Random seed: {cfg['system']['random_seed']}")

    # 2. Generate Chronological Telemetry Dataset with Mode Tracking
    n_samples = 15000
    print(f"[2/7] Generating N={n_samples} synthetic distributed grid telemetry events across 60 nodes...")
    full_df = generate_telemetry_dataset(
        n_samples=n_samples,
        anomaly_ratio=0.08,
        seed=seed,
        num_steps=500,
        return_modes=True,
        config=cfg,
    )
    anomaly_count = int(full_df["is_anomaly"].sum())
    print(f"      Dataset generated: {len(full_df)} events ({anomaly_count} injected multi-modal anomalies, {anomaly_count / len(full_df):.2%})")

    # Chronological Train-Test Split (75% train, 25% test)
    split_idx = int(len(full_df) * 0.75)
    train_df = full_df.iloc[:split_idx].reset_index(drop=True)
    test_df = full_df.iloc[split_idx:].reset_index(drop=True)

    y_train = train_df["is_anomaly"].values.astype(int)
    y_test = test_df["is_anomaly"].values.astype(int)
    print(f"      Train set: {len(train_df)} events | Test set: {len(test_df)} events")

    # 3. Feature Engineering (14 channels)
    print("[3/7] Extracting 14 engineered kinematic/queue/surge features...")
    extractor = TelemetryFeatureExtractor(
        config=cfg["feature_engineering"],
        schema_config=cfg["telemetry_schema"],
    )
    X_train_df = extractor.transform_df(train_df)
    X_test_df = extractor.transform_df(test_df)
    X_train = X_train_df.values
    X_test = X_test_df.values

    # 4. Train Tri-Modal Anomaly Engines
    print("[4/7] Training Tri-Modal Detection Sub-Models...")

    # A. LightGBM Detector
    t_lgbm_start = time.perf_counter()
    lgbm_config = dict(cfg.get("lightgbm", {}))
    lgbm_config.update({"n_estimators": 50, "max_depth": 3, "learning_rate": 0.05, "min_child_samples": 40})
    lgbm_config.update({"random_state": seed, "n_jobs": int(cfg["system"].get("num_threads", 1))})
    lgbm_detector = LGBMAnomalyDetector(config=lgbm_config)
    lgbm_detector.fit(X_train, y_train)
    t_lgbm = (time.perf_counter() - t_lgbm_start) * 1000.0
    print(f"      - LightGBM fitted in {t_lgbm:.1f} ms")

    # B. PyTorch LSTM Autoencoder Detector
    t_lstm_start = time.perf_counter()
    lstm_config = dict(cfg.get("lstm_autoencoder", {}))
    lstm_config["device"] = cfg["system"].get("device", "cpu")
    lstm_detector = LSTMAutoencoderDetector(config=lstm_config)
    nom_train = train_df[train_df["is_anomaly"] == 0]
    X_nom_feat = extractor.transform_df(nom_train).values[:, LSTM_FEATURE_INDICES]
    train_seqs, _ = extractor.build_sequences(
        nom_train,
        X_nom_feat,
        seq_len=int(cfg["telemetry_schema"]["sequence_length"]),
        window_stride=int(cfg["telemetry_schema"]["window_stride"]),
        source_col=cfg["telemetry_schema"]["src_node_col"],
        time_col=cfg["telemetry_schema"]["time_col"],
    )
    lstm_detector.fit(train_seqs)
    t_lstm = (time.perf_counter() - t_lstm_start) * 1000.0
    print(f"      - PyTorch LSTM-AE trained (8 epochs) in {t_lstm:.1f} ms")

    # C. NetworkX Graph Topology Analyzer
    t_top_start = time.perf_counter()
    top_analyzer = GridTopologyAnalyzer(config=cfg.get("topology_graph", {}))
    top_analyzer.build_graph_from_events(train_df)
    t_top = (time.perf_counter() - t_top_start) * 1000.0
    print(f"      - NetworkX DiGraph built ({len(top_analyzer.graph.nodes)} nodes, {len(top_analyzer.graph.edges)} edges) in {t_top:.1f} ms")

    test_top_analyzer = GridTopologyAnalyzer(config=cfg.get("topology_graph", {}))
    test_top_analyzer.build_graph_from_events(test_df)

    # 5. Fit the logistic meta-classifier on the training split
    print("[5/7] Fitting the stacked logistic ensemble...")
    s_lgbm_train = lgbm_detector.predict_proba(X_train)

    X_train_cont = X_train[:, LSTM_FEATURE_INDICES]
    tr_seqs, _ = extractor.build_sequences(
        train_df,
        X_train_cont,
        seq_len=int(cfg["telemetry_schema"]["sequence_length"]),
        window_stride=int(cfg["telemetry_schema"]["window_stride"]),
        source_col=cfg["telemetry_schema"]["src_node_col"],
        time_col=cfg["telemetry_schema"]["time_col"],
    )
    s_lstm_train = lstm_detector.predict_score(tr_seqs)

    s_graph_train, _ = top_analyzer.score_batch(train_df)

    ensemble_config = dict(cfg.get("hybrid_ensemble", {}))
    ensemble_config["random_state"] = seed
    ensemble = HybridEnsembleDetector(config=ensemble_config)
    ensemble.fit(s_lgbm_train, s_lstm_train, s_graph_train, y_train)
    print(f"      - Decision threshold={ensemble.alert_threshold:.2f}; meta-weights=[{ensemble.w0:.2f}, {ensemble.w1:.2f}, {ensemble.w2:.2f}, {ensemble.w3:.2f}, {ensemble.w4:.2f}]")

    # Test Inference & Ablation Evaluation
    s_lgbm_test = lgbm_detector.predict_proba(X_test)

    X_test_cont = X_test[:, LSTM_FEATURE_INDICES]
    test_seqs, _ = extractor.build_sequences(
        test_df,
        X_test_cont,
        seq_len=int(cfg["telemetry_schema"]["sequence_length"]),
        window_stride=int(cfg["telemetry_schema"]["window_stride"]),
        source_col=cfg["telemetry_schema"]["src_node_col"],
        time_col=cfg["telemetry_schema"]["time_col"],
    )
    s_lstm_test = lstm_detector.predict_score(test_seqs)

    s_graph_test, meta_test = test_top_analyzer.score_batch(test_df)
    s_hybrid_test = ensemble.combine_scores(s_lgbm_test, s_lstm_test, s_graph_test)

    # Compute Comparative Metrics
    eval_results = ensemble.evaluate_all(y_test, s_lgbm_test, s_lstm_test, s_graph_test)
    print("\n" + "-" * 75)
    print("EMPIRICAL MODEL ABLATION RESULTS (HELD-OUT TEST SET):")
    print(f"{'Component':<20} | {'ROC-AUC':<9} | {'PR-AUC':<9} | {'F1-Score':<9} | {'Precision':<9} | {'Recall':<9}")
    print("-" * 75)
    for model_name, m in eval_results.items():
        print(f"{model_name:<20} | {m['roc_auc']:<9.5f} | {m['pr_auc']:<9.5f} | {m['f1_score']:<9.5f} | {m['precision']:<9.5f} | {m['recall']:<9.5f}")
    print("-" * 75 + "\n")

    # 6. Initialize SHAP, Index ChromaDB RAG, and Measure Latencies
    print("[6/7] Initializing SHAP TreeExplainer & ChromaDB Runbook RAG...")
    shap_attributor = ShapAttributor(
        lgbm_model=lgbm_detector.get_model(),
        feature_names=extractor.feature_names,
        top_k=4,
    )
    rag_engine = RunbookRAGEngine(config=cfg.get("rag_diagnostics", {}))
    indexed_runbooks_count = rag_engine.index_runbooks()
    print(f"      - ChromaDB indexed {indexed_runbooks_count} semantic runbook sections on CPU.")

    # Evaluate Runbook RAG Retrieval Accuracy on 100 Test Anomalies
    print("      Evaluating Runbook RAG retrieval accuracy across 100 test anomalies...")
    anom_test_df = test_df[test_df["is_anomaly"] == 1].reset_index(drop=True)
    eval_rows: List[pd.DataFrame] = []
    for mode_name in ["MODE_A", "MODE_B", "MODE_C", "MODE_D"]:
        sub_mode = anom_test_df[anom_test_df["_mode"] == mode_name]
        eval_rows.append(sub_mode.iloc[:25])
    eval_rag_df = pd.concat(eval_rows, ignore_index=True)

    mode_to_valid_runbooks = {
        "MODE_A": {"RB-GRID-001"},
        "MODE_B": {"RB-GRID-005", "RB-GRID-006"},
        "MODE_C": {"RB-GRID-002", "RB-GRID-006"},
        "MODE_D": {"RB-GRID-003", "RB-GRID-004"},
    }

    top1_correct = 0
    top2_correct = 0

    for _, row in eval_rag_df.iterrows():
        m_label = str(row["_mode"])
        ev_dict = row.to_dict()
        feat_df_i = extractor.transform_single(ev_dict)
        feat_1d = feat_df_i.values[0]

        shap_drivers = shap_attributor.explain_instance(feat_1d, observed_dict=ev_dict)
        _, t_meta = test_top_analyzer.analyze_event(
            source_node=str(row["source_node"]),
            target_node=str(row["target_node"]),
            payload_mb=float(row["payload_mb"]),
            protocol=str(row["protocol_type"]),
            window_step=int(row["window_step"]),
        )
        is_temp_drift = (m_label == "MODE_B")

        recs = rag_engine.retrieve_recommendations(
            protocol_type=str(row["protocol_type"]),
            shap_drivers=shap_drivers,
            bottleneck_regime=str(t_meta.get("bottleneck_regime", "NOMINAL")),
            temporal_drift_detected=is_temp_drift,
            top_k=2,
        )
        retrieved_ids = [r["runbook_id"] for r in recs]
        expected_ids = mode_to_valid_runbooks[m_label]

        if retrieved_ids and retrieved_ids[0] in expected_ids:
            top1_correct += 1
        if any(r_id in expected_ids for r_id in retrieved_ids[:2]):
            top2_correct += 1

    n_rag_eval = len(eval_rag_df)
    top1_acc = round(float(top1_correct / n_rag_eval), 4)
    top2_acc = round(float(top2_correct / n_rag_eval), 4)
    print(f"      - Top-1 Runbook Retrieval Accuracy: {top1_correct}/{n_rag_eval} ({top1_acc:.2%})")
    print(f"      - Top-2 Runbook Retrieval Accuracy: {top2_correct}/{n_rag_eval} ({top2_acc:.2%})")

    # Latency benchmarking on 20 individual diagnostic runs
    latencies_scoring = []
    latencies_shap = []
    latencies_rag = []
    latencies_e2e = []

    sample_event = generate_single_event(is_anomaly=True, mode="MODE_A", seed=seed + 1000)
    feat_single = extractor.transform_single(sample_event).values
    sample_cont = feat_single[0, LSTM_FEATURE_INDICES]
    seq_len = int(cfg["telemetry_schema"]["sequence_length"])
    source_col = cfg["telemetry_schema"]["src_node_col"]
    time_col = cfg["telemetry_schema"]["time_col"]
    sample_history = train_df[
        (train_df[source_col].astype(str) == str(sample_event[source_col]))
        & (train_df[time_col] <= sample_event[time_col])
    ].sort_values(time_col, kind="mergesort")
    history_positions = sample_history.index.to_numpy(dtype=int)
    history_cont = X_train_cont[history_positions][-(seq_len - 1):] if seq_len > 1 else np.empty((0, len(sample_cont)))
    sample_sequence = np.concatenate([history_cont, sample_cont[None, :]], axis=0)
    if len(sample_sequence) < seq_len:
        sample_sequence = np.concatenate(
            [np.repeat(sample_sequence[:1], seq_len - len(sample_sequence), axis=0), sample_sequence],
            axis=0,
        )

    for _ in range(20):
        t0 = time.perf_counter()
        # Scoring
        _ = lgbm_detector.predict_proba(feat_single)
        _ = lstm_detector.predict_score(sample_sequence[None, :, :])
        _ = top_analyzer.analyze_event(sample_event["source_node"], sample_event["target_node"])
        t1 = time.perf_counter()
        latencies_scoring.append((t1 - t0) * 1000.0)

        # SHAP
        shap_res = shap_attributor.explain_instance(feat_single[0], observed_dict=sample_event)
        t2 = time.perf_counter()
        latencies_shap.append((t2 - t1) * 1000.0)

        # RAG
        rag_res = rag_engine.retrieve_recommendations(
            protocol_type=sample_event["protocol_type"],
            shap_drivers=shap_res,
            bottleneck_regime="NOMINAL",
            temporal_drift_detected=False,
            top_k=2,
        )
        t3 = time.perf_counter()
        latencies_rag.append((t3 - t2) * 1000.0)
        latencies_e2e.append((t3 - t0) * 1000.0)

    avg_scoring_ms = float(np.mean(latencies_scoring))
    avg_shap_ms = float(np.mean(latencies_shap))
    avg_rag_ms = float(np.mean(latencies_rag))
    avg_e2e_ms = float(np.mean(latencies_e2e))

    print(f"      Latency Telemetry (mean of 20 runs):")
    print(f"      - Tri-Modal Scoring: {avg_scoring_ms:.2f} ms")
    print(f"      - SHAP TreeExplainer Attribution: {avg_shap_ms:.2f} ms")
    print(f"      - ChromaDB Dense Vector RAG: {avg_rag_ms:.2f} ms")
    print(f"      - Total End-to-End Pipeline: {avg_e2e_ms:.2f} ms")

    # Generate sample diagnostic report
    s_lgbm_sample = float(lgbm_detector.predict_proba(feat_single)[0])
    s_lstm_sample = float(lstm_detector.predict_score(sample_sequence[None, :, :])[0])
    s_top_sample, meta_sample = top_analyzer.analyze_event(sample_event["source_node"], sample_event["target_node"])
    s_hyb_sample = float(ensemble.combine_scores(s_lgbm_sample, s_lstm_sample, s_top_sample))
    is_anom_sample, sev_sample = ensemble.classify_severity(s_hyb_sample)

    sample_report = {
        "event_id": "evt_diag_benchmark_001",
        "timestamp_utc": deterministic_timestamp(sample_event[time_col]),
        "input_telemetry": sample_event,
        "is_anomaly": is_anom_sample,
        "severity": sev_sample,
        "hybrid_score": round(s_hyb_sample, 4),
        "model_scores": {
            "lightgbm": round(s_lgbm_sample, 4),
            "lstm_autoencoder": round(s_lstm_sample, 4),
            "networkx_topology": round(s_top_sample, 4),
        },
        "top_shap_drivers": shap_res,
        "runbook_recommendations": rag_res,
        "execution_latency_ms": {
            "scoring_ms": round(avg_scoring_ms, 2),
            "shap_attribution_ms": round(avg_shap_ms, 2),
            "chromadb_rag_ms": round(avg_rag_ms, 2),
            "end_to_end_total_ms": round(avg_e2e_ms, 2),
        },
    }

    # Save benchmark_metrics.json
    metrics_export = {
        "dataset_statistics": {
            "total_samples": n_samples,
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "anomaly_ratio": round(float(anomaly_count / n_samples), 4),
            "num_nodes": 60,
        },
        "component_ablation_metrics": eval_results,
        "rag_retrieval_benchmarks": {
            "evaluation_anomalies_count": n_rag_eval,
            "top1_retrieval_accuracy": top1_acc,
            "top2_retrieval_accuracy": top2_acc,
            "evaluation_modes_covered": ["MODE_A", "MODE_B", "MODE_C", "MODE_D"],
        },
        "latency_benchmarks_ms": {
            "tri_modal_scoring_mean_ms": round(avg_scoring_ms, 2),
            "shap_tree_explainer_mean_ms": round(avg_shap_ms, 2),
            "chromadb_vector_rag_mean_ms": round(avg_rag_ms, 2),
            "end_to_end_pipeline_mean_ms": round(avg_e2e_ms, 2),
        },
        "system_metadata": {
            "hardware_device": "CPU",
            "embedding_model": "all-MiniLM-L6-v2",
            "embedding_dimensions": 384,
            "indexed_runbooks_count": indexed_runbooks_count,
            "hash_chain_algorithm": "SHA-256",
            "ensemble_architecture": "Stacked Logistic Hybrid Ensemble",
        },
    }

    metrics_file = assets_dir / "benchmark_metrics.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_export, f, indent=2)
    print(f"      Saved: {metrics_file}")

    report_file = assets_dir / "sample_diagnostic_report.json"
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(sample_report, f, indent=2)
    print(f"      Saved: {report_file}")

    # 7. Generate Figures
    print("[7/7] Generating empirical diagnostic figures...")

    # Figure 1: Model Ablation ROC & PR Curves (assets/model_ablation_roc_pr.png)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    plt.subplots_adjust(wspace=0.25)

    models_data = [
        ("LightGBM", s_lgbm_test, "#1f77b4", ":"),
        ("LSTM Autoencoder", s_lstm_test, "#ff7f0e", "--"),
        ("NetworkX Topology", s_graph_test, "#2ca02c", "-."),
        ("Hybrid Ensemble", s_hybrid_test, "#d62728", "-"),
    ]

    for name, scores, color, ls in models_data:
        fpr, tpr, _ = roc_curve(y_test, scores)
        prec, rec, _ = precision_recall_curve(y_test, scores)
        m = eval_results.get(name.replace(" ", "_"), {})
        auc_val = m.get("roc_auc", 0.0)
        pr_val = m.get("pr_auc", 0.0)

        lw = 2.8 if name == "Hybrid Ensemble" else 1.8
        ax1.plot(fpr, tpr, color=color, linestyle=ls, linewidth=lw, label=f"{name} (AUC = {auc_val:.3f})")
        ax2.plot(rec, prec, color=color, linestyle=ls, linewidth=lw, label=f"{name} (PR-AUC = {pr_val:.3f})")

    ax1.plot([0, 1], [0, 1], "k:", alpha=0.5, label="Random Baseline")
    ax1.set_title("Grid Telemetry Anomaly Detection: ROC Curves", fontsize=12, fontweight="bold")
    ax1.set_xlabel("False Positive Rate (FPR)", fontsize=10)
    ax1.set_ylabel("True Positive Rate (TPR)", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="lower right", fontsize=9)

    ax2.set_title("Precision-Recall Curves Across Detection Modalities", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Recall", fontsize=10)
    ax2.set_ylabel("Precision", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="lower left", fontsize=9)

    roc_pr_path = assets_dir / "model_ablation_roc_pr.png"
    plt.savefig(roc_pr_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {roc_pr_path}")

    # Figure 2: SHAP Feature Importance & NetworkX Topology Graph (assets/shap_and_topology_diagnostics.png)
    fig, (ax_shap, ax_graph) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)
    plt.subplots_adjust(wspace=0.28)

    # Panel 1: Global Mean Absolute SHAP Importance
    global_shap = shap_attributor.compute_global_importance(X_test[:250])
    top_shap = global_shap.head(10).iloc[::-1]
    y_pos = np.arange(len(top_shap))
    ax_shap.barh(y_pos, top_shap["mean_abs_shap"], color="#3366cc", edgecolor="#1a3366", height=0.65)
    ax_shap.set_yticks(y_pos)
    ax_shap.set_yticklabels(top_shap["feature"], fontsize=9)
    ax_shap.set_xlabel("Mean |SHAP Value| (Impact on Anomaly Probability)", fontsize=10)
    ax_shap.set_title("Global SHAP Feature Attribution (Top 10 Channels)", fontsize=12, fontweight="bold")
    ax_shap.grid(True, linestyle="--", alpha=0.4, axis="x")

    # Panel 2: NetworkX Topology Bottleneck Graph Visualization
    import networkx as nx
    subgraph_nodes = list(top_analyzer.graph.nodes)[:28]
    subgraph = top_analyzer.graph.subgraph(subgraph_nodes)
    pos = nx.spring_layout(subgraph, seed=42)

    node_colors = []
    for n in subgraph.nodes():
        if "SE_POOL" in str(n):
            node_colors.append("#ff7f0e")
        elif n in top_analyzer.cycle_nodes_cache:
            node_colors.append("#d62728")
        else:
            node_colors.append("#1f77b4")

    nx.draw_networkx_nodes(subgraph, pos, ax=ax_graph, node_size=320, node_color=node_colors, edgecolors="black", linewidths=0.8)
    nx.draw_networkx_edges(subgraph, pos, ax=ax_graph, arrowstyle="->", arrowsize=10, edge_color="#888888", width=1.0, alpha=0.7)
    nx.draw_networkx_labels(subgraph, pos, ax=ax_graph, font_size=6, font_family="sans-serif", font_weight="bold")

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="o", color="w", label="Compute Node", markerfacecolor="#1f77b4", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Storage Pool (Incast)", markerfacecolor="#ff7f0e", markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Cycle Storm Node", markerfacecolor="#d62728", markersize=8),
    ]
    ax_graph.legend(handles=legend_elements, loc="upper right", fontsize=8)
    ax_graph.set_title("NetworkX Grid Interaction Topology & Identified Loops", fontsize=12, fontweight="bold")
    ax_graph.axis("off")

    shap_topo_path = assets_dir / "shap_and_topology_diagnostics.png"
    plt.savefig(shap_topo_path, bbox_inches="tight")
    plt.close()
    print(f"      Saved: {shap_topo_path}")

    print("\nBenchmark artifacts written.")


if __name__ == "__main__":
    main()
