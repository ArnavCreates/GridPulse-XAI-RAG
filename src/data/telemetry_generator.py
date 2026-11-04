"""Synthetic distributed cloud & compute cluster telemetry generator.

Generates 10-column telemetry events conforming to distributed cloud & compute
cluster architectures with realistic multi-modal anomaly injections:
- Mode A: Tabular Resource Surge (queue depth & CPU surge)
- Mode B: Stealth Temporal Sequence Drift & Oscillation (6-step sliding episode)
- Mode C: Stealth Directed Routing Cycles (3-hop & 4-hop feedback loops)
- Mode D: Stealth Fan-In Incast (congested multi-source synchronized bursts)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from ..config_loader import load_config

PROTOCOLS = list(load_config()["telemetry_schema"]["allowed_protocols"])


def _build_node_pool(num_compute: int = 40, num_storage: int = 20) -> Tuple[List[str], List[str]]:
    """Build identifiers for compute nodes and storage element pools."""
    compute_nodes = [f"NODE_T1_{i:03d}" for i in range(1, num_compute + 1)]
    storage_nodes = [f"SE_POOL_{i:03d}" for i in range(1, num_storage + 1)]
    return compute_nodes, storage_nodes


def sample_nominal_event(
    rng: np.random.Generator,
    compute_nodes: List[str],
    storage_nodes: List[str],
    window_step: int,
) -> Dict[str, Any]:
    """Sample a single nominal event with realistic grid feature distributions."""
    p_val = rng.random()
    if p_val < 0.40:
        proto = "XROOTD_STREAM"
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes))
        payload = float(rng.exponential(scale=20.0) + 2.0)
    elif p_val < 0.65:
        proto = "GRID_FTP"
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes))
        payload = float(rng.exponential(scale=18.0) + 2.0)
    elif p_val < 0.80:
        proto = "CACHE_REPLICATION"
        src = str(rng.choice(storage_nodes))
        dst = str(rng.choice(compute_nodes))
        payload = float(rng.exponential(scale=14.0) + 1.0)
    elif p_val < 0.90:
        proto = "RPC_CONTROL"
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes))
        payload = float(rng.uniform(0.5, 4.0))
    else:
        proto = "HEARTBEAT_SYNC"
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes))
        payload = float(rng.uniform(0.2, 1.8))

    payload = float(np.clip(payload, 0.2, 55.0))
    cpu_prev = float(rng.uniform(25.0, 60.0))
    cpu_curr = float(np.clip(cpu_prev + rng.normal(0.0, 2.0), 20.0, 68.0))
    q_prev = float(rng.uniform(4.0, 16.0))
    q_curr = float(np.clip(q_prev + rng.normal(0.0, 1.2), 3.0, 20.0))

    return {
        "window_step": int(window_step),
        "protocol_type": str(proto),
        "payload_mb": round(payload, 4),
        "source_node": str(src),
        "src_cpu_load_prev": round(cpu_prev, 2),
        "src_cpu_load_curr": round(cpu_curr, 2),
        "target_node": str(dst),
        "dst_queue_depth_prev": round(q_prev, 2),
        "dst_queue_depth_curr": round(q_curr, 2),
        "is_anomaly": 0,
    }


def generate_telemetry_dataset(
    n_samples: int = 15000,
    anomaly_ratio: float = 0.08,
    seed: int = 42,
    num_steps: int = 500,
    return_modes: bool = False,
    config: Optional[Dict[str, Any]] = None,
) -> pd.DataFrame:
    """Generate a chronological dataset of distributed grid telemetry.

    Args:
        n_samples: Total number of telemetry events to generate (default 15,000).
        anomaly_ratio: Proportion of injected anomalies (~0.08).
        seed: Random seed for reproducibility.
        num_steps: Number of discrete time windows (default 500).
        return_modes: If True, includes internal '_mode' column for diagnostics.

    Returns:
        pd.DataFrame conforming to 10 standard columns (or 11 if return_modes=True).
    """
    rng = np.random.default_rng(seed)
    compute_nodes, storage_nodes = _build_node_pool()

    target_anomalies = max(1, int(n_samples * anomaly_ratio))
    target_nominal = n_samples - target_anomalies

    records: List[Dict[str, Any]] = []

    # 1. Nominal streams (6-event reporting bursts per node session)
    nom_generated = 0
    while nom_generated < target_nominal:
        w = (nom_generated // 30) % num_steps + 1
        stream_len = min(6, target_nominal - nom_generated)
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes))
        proto = str(
            rng.choice(
                ["XROOTD_STREAM", "GRID_FTP", "CACHE_REPLICATION", "RPC_CONTROL", "HEARTBEAT_SYNC"],
                p=[0.40, 0.25, 0.15, 0.10, 0.10],
            )
        )

        c_state = float(rng.uniform(28.0, 56.0))
        q_state = float(rng.uniform(5.0, 15.0))
        p_base = float(rng.uniform(10.0, 35.0))

        for _ in range(stream_len):
            prev_c = c_state
            c_state = float(np.clip(prev_c + rng.normal(0.0, 1.8), 20.0, 68.0))
            prev_q = q_state
            q_state = float(np.clip(prev_q + rng.normal(0.0, 1.0), 3.0, 20.0))
            p_val = float(np.clip(p_base + rng.normal(0.0, 2.5), 0.5, 55.0))

            rec = {
                "window_step": int(w),
                "protocol_type": proto,
                "payload_mb": round(p_val, 4),
                "source_node": str(src),
                "src_cpu_load_prev": round(prev_c, 2),
                "src_cpu_load_curr": round(c_state, 2),
                "target_node": str(dst),
                "dst_queue_depth_prev": round(prev_q, 2),
                "dst_queue_depth_curr": round(q_state, 2),
                "is_anomaly": 0,
            }
            if return_modes:
                rec["_mode"] = "NOMINAL"
            records.append(rec)
            nom_generated += 1

    n_mode_a = int(target_anomalies * 0.35)
    n_mode_b = int(target_anomalies * 0.30)
    n_mode_c = int(target_anomalies * 0.18)
    n_mode_d = target_anomalies - (n_mode_a + n_mode_b + n_mode_c)

    # 2. Mode A: Extreme Tabular Resource Surge (LightGBM specializes)
    for _ in range(n_mode_a):
        w = int(rng.integers(1, num_steps + 1))
        src = str(rng.choice(compute_nodes))
        dst = str(rng.choice(storage_nodes[:6]))
        payload = float(rng.uniform(280.0, 750.0))
        src_cpu_prev = float(rng.uniform(45.0, 65.0))
        src_cpu_curr = float(rng.uniform(91.0, 99.8))
        dst_queue_prev = float(rng.uniform(10.0, 22.0))
        dst_queue_curr = float(dst_queue_prev + rng.uniform(50.0, 105.0))

        rec = {
            "window_step": w,
            "protocol_type": "XROOTD_STREAM",
            "payload_mb": round(payload, 4),
            "source_node": str(src),
            "src_cpu_load_prev": round(src_cpu_prev, 2),
            "src_cpu_load_curr": round(src_cpu_curr, 2),
            "target_node": str(dst),
            "dst_queue_depth_prev": round(dst_queue_prev, 2),
            "dst_queue_depth_curr": round(dst_queue_curr, 2),
            "is_anomaly": 1,
        }
        if return_modes:
            rec["_mode"] = "MODE_A"
        records.append(rec)

    # 3. Mode B: Stealth Temporal Sequence Sawtooth (Contiguous 6-step episode)
    n_mode_b_episodes = max(1, n_mode_b // 6)
    for _ in range(n_mode_b_episodes):
        w = int(rng.integers(1, max(2, num_steps - 5)))
        drift_node = str(rng.choice(compute_nodes))
        drift_target = str(rng.choice(storage_nodes))
        base_c = float(rng.uniform(34.0, 50.0))
        base_q = float(rng.uniform(7.0, 13.0))

        for s in range(6):
            sign = 1.0 if (s % 2 == 0) else -1.0
            proto = str(rng.choice(["XROOTD_STREAM", "GRID_FTP", "CACHE_REPLICATION"], p=[0.50, 0.35, 0.15]))
            cpu_curr = float(np.clip(base_c + sign * 9.0, 20.0, 65.0))
            cpu_prev = float(np.clip(cpu_curr - sign * rng.uniform(0.5, 2.5), 20.0, 65.0))
            q_curr = float(np.clip(base_q - sign * 4.0, 3.0, 18.0))
            q_prev = float(np.clip(q_curr + sign * rng.uniform(0.4, 1.8), 3.0, 18.0))
            p_load = float(rng.uniform(14.0, 35.0))

            rec = {
                "window_step": w + s,
                "protocol_type": proto,
                "payload_mb": round(p_load, 4),
                "source_node": str(drift_node),
                "src_cpu_load_prev": round(cpu_prev, 2),
                "src_cpu_load_curr": round(cpu_curr, 2),
                "target_node": str(drift_target),
                "dst_queue_depth_prev": round(q_prev, 2),
                "dst_queue_depth_curr": round(q_curr, 2),
                "is_anomaly": 1,
            }
            if return_modes:
                rec["_mode"] = "MODE_B"
            records.append(rec)

    # 4. Mode C: Stealth Routing Cycles (3-hop and 4-hop loops)
    cycle_templates = [
        ["NODE_T1_005", "NODE_T1_006", "NODE_T1_007", "NODE_T1_005"],
        ["NODE_T1_010", "NODE_T1_011", "NODE_T1_012", "NODE_T1_010"],
        ["NODE_T1_015", "NODE_T1_016", "NODE_T1_017", "NODE_T1_018", "NODE_T1_015"],
        ["NODE_T1_020", "NODE_T1_021", "NODE_T1_022", "NODE_T1_020"],
    ]
    mode_c_count = 0
    while mode_c_count < n_mode_c:
        w = int(rng.integers(1, num_steps + 1))
        tmpl = cycle_templates[int(rng.integers(0, len(cycle_templates)))]
        for i in range(len(tmpl) - 1):
            if mode_c_count >= n_mode_c:
                break
            u, v = tmpl[i], tmpl[i + 1]
            rec = sample_nominal_event(rng, compute_nodes, storage_nodes, w)
            rec["source_node"] = u
            rec["target_node"] = v
            rec["is_anomaly"] = 1
            if return_modes:
                rec["_mode"] = "MODE_C"
            records.append(rec)
            mode_c_count += 1

    # 5. Mode D: Stealth Fan-In Incast (>= 7 distinct flows hitting same SE in window w)
    mode_d_count = 0
    incast_targets = ["SE_POOL_003", "SE_POOL_007", "SE_POOL_011"]
    while mode_d_count < n_mode_d:
        w = int(rng.integers(1, num_steps + 1))
        target_se = str(incast_targets[int(rng.integers(0, len(incast_targets)))])
        burst_size = min(8, n_mode_d - mode_d_count)
        sources = rng.choice(compute_nodes, size=burst_size, replace=False)
        for src in sources:
            rec = sample_nominal_event(rng, compute_nodes, storage_nodes, w)
            rec["source_node"] = str(src)
            rec["target_node"] = target_se
            rec["is_anomaly"] = 1
            if return_modes:
                rec["_mode"] = "MODE_D"
            records.append(rec)
            mode_d_count += 1

    df = pd.DataFrame(records)
    schema = (config or load_config()).get("telemetry_schema", {})
    allowed_protocols = list(schema.get("allowed_protocols", PROTOCOLS))
    if allowed_protocols:
        df["protocol_type"] = df["protocol_type"].where(
            df["protocol_type"].isin(allowed_protocols), allowed_protocols[0]
        )
    for column, minimum_key, maximum_key in (
        ("payload_mb", "min_payload_mb", "max_payload_mb"),
        ("src_cpu_load_prev", "min_cpu_load", "max_cpu_load"),
        ("src_cpu_load_curr", "min_cpu_load", "max_cpu_load"),
        ("dst_queue_depth_prev", "min_queue_depth", "max_queue_depth"),
        ("dst_queue_depth_curr", "min_queue_depth", "max_queue_depth"),
    ):
        if minimum_key in schema and maximum_key in schema:
            df[column] = df[column].clip(float(schema[minimum_key]), float(schema[maximum_key]))
    if len(df) > n_samples:
        df_anom = df[df["is_anomaly"] == 1]
        keep_nom_count = max(0, n_samples - len(df_anom))
        df_nom = df[df["is_anomaly"] == 0].iloc[:keep_nom_count]
        df = pd.concat([df_nom, df_anom], ignore_index=True)
    elif len(df) < n_samples:
        diff = n_samples - len(df)
        pad_rows = df[df["is_anomaly"] == 0].iloc[:diff].copy()
        df = pd.concat([df, pad_rows], ignore_index=True)

    # Stable sort by window_step preserves Mode B 6-step blocks
    df = df.sort_values(by=["window_step"], kind="mergesort").reset_index(drop=True)
    return df


def generate_single_event(
    is_anomaly: bool = False,
    mode: str = "MODE_A",
    seed: Optional[int] = None,
) -> Dict[str, Any]:
    """Generate a single sample 10-column telemetry event for API testing/inference."""
    rng = np.random.default_rng(seed)
    if not is_anomaly:
        return {
            "window_step": int(rng.integers(1, 500)),
            "protocol_type": "XROOTD_STREAM",
            "payload_mb": round(float(rng.uniform(10.0, 45.0)), 4),
            "source_node": "NODE_T1_014",
            "src_cpu_load_prev": 42.50,
            "src_cpu_load_curr": 44.10,
            "target_node": "SE_POOL_007",
            "dst_queue_depth_prev": 8.00,
            "dst_queue_depth_curr": 9.20,
            "is_anomaly": 0,
        }
    else:
        if mode == "MODE_A":
            return {
                "window_step": 142,
                "protocol_type": "XROOTD_STREAM",
                "payload_mb": 512.4500,
                "source_node": "NODE_T1_014",
                "src_cpu_load_prev": 48.00,
                "src_cpu_load_curr": 96.50,
                "target_node": "SE_POOL_007",
                "dst_queue_depth_prev": 12.00,
                "dst_queue_depth_curr": 88.50,
                "is_anomaly": 1,
            }
        elif mode == "MODE_B":
            return {
                "window_step": 188,
                "protocol_type": "GRID_FTP",
                "payload_mb": 24.5000,
                "source_node": "NODE_T1_008",
                "src_cpu_load_prev": 42.00,
                "src_cpu_load_curr": 54.00,
                "target_node": "SE_POOL_012",
                "dst_queue_depth_prev": 9.50,
                "dst_queue_depth_curr": 6.00,
                "is_anomaly": 1,
            }
        elif mode == "MODE_C":
            return {
                "window_step": 204,
                "protocol_type": "RPC_CONTROL",
                "payload_mb": 22.1000,
                "source_node": "NODE_T1_005",
                "src_cpu_load_prev": 38.00,
                "src_cpu_load_curr": 42.20,
                "target_node": "NODE_T1_006",
                "dst_queue_depth_prev": 8.00,
                "dst_queue_depth_curr": 10.00,
                "is_anomaly": 1,
            }
        else:  # MODE_D
            return {
                "window_step": 310,
                "protocol_type": "GRID_FTP",
                "payload_mb": 28.0000,
                "source_node": "NODE_T1_022",
                "src_cpu_load_prev": 35.00,
                "src_cpu_load_curr": 39.00,
                "target_node": "SE_POOL_003",
                "dst_queue_depth_prev": 11.00,
                "dst_queue_depth_curr": 14.00,
                "is_anomaly": 1,
            }
