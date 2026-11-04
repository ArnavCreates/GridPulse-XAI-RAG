"""NetworkX directed routing-cycle and fan-in/fan-out bottleneck topology analyzer.

Builds a directed multigraph / DiGraph over source-target interactions,
detects feedback routing loops of length 3 <= L <= 4, computes degree and PageRank
centrality, and emits structural diagnostics and calibrated anomaly scores.
"""

from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
import numpy as np
import pandas as pd


class GridTopologyAnalyzer:
    """Directed graph topology analyzer for distributed grid communication flows."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_cycle_length: int = 4,
        fan_in_threshold: int = 7,
        fan_out_threshold: int = 9,
        cycle_penalty: float = 0.85,
        bottleneck_penalty: float = 0.75,
        pagerank_alpha: float = 0.85,
    ):
        if config:
            self.max_cycle_length = config.get("max_cycle_length", max_cycle_length)
            self.fan_in_threshold = config.get("fan_in_threshold", fan_in_threshold)
            self.fan_out_threshold = config.get("fan_out_threshold", fan_out_threshold)
            self.cycle_penalty = config.get("cycle_penalty_weight", cycle_penalty)
            self.bottleneck_penalty = config.get("bottleneck_penalty_weight", bottleneck_penalty)
            self.pagerank_alpha = config.get("pagerank_alpha", pagerank_alpha)
        else:
            self.max_cycle_length = max_cycle_length
            self.fan_in_threshold = fan_in_threshold
            self.fan_out_threshold = fan_out_threshold
            self.cycle_penalty = cycle_penalty
            self.bottleneck_penalty = bottleneck_penalty
            self.pagerank_alpha = pagerank_alpha

        self.graph = nx.DiGraph()
        self.window_graphs: Dict[int, nx.DiGraph] = {}
        self.window_cycles: Dict[int, List[List[str]]] = {}
        self.cycle_nodes_cache: set[str] = set()
        self.cycle_edges_cache: set[Tuple[str, str]] = set()
        self.cycle_paths_cache: List[List[str]] = []
        self.pagerank_scores: Dict[str, float] = {}

    def build_graph_from_events(
        self,
        events_df: pd.DataFrame,
        decay_existing: bool = False,
    ) -> "GridTopologyAnalyzer":
        """Build/update directed interaction graphs from telemetry events."""
        if not decay_existing:
            self.graph.clear()
            self.window_graphs.clear()
            self.window_cycles.clear()
            self.cycle_nodes_cache.clear()
            self.cycle_edges_cache.clear()
            self.cycle_paths_cache.clear()

        # Build overall graph
        for _, row in events_df.iterrows():
            u = str(row["source_node"])
            v = str(row["target_node"])
            payload = float(row.get("payload_mb", 1.0))

            if self.graph.has_edge(u, v):
                self.graph[u][v]["weight"] += 1.0
                self.graph[u][v]["payload_mb"] += payload
            else:
                self.graph.add_edge(u, v, weight=1.0, payload_mb=payload)

        # Build per-window graphs if window_step is present
        if "window_step" in events_df.columns:
            for w, group in events_df.groupby("window_step"):
                w_int = int(w)
                gw = nx.DiGraph()
                for _, r in group.iterrows():
                    u = str(r["source_node"])
                    v = str(r["target_node"])
                    p_load = float(r.get("payload_mb", 1.0))
                    if gw.has_edge(u, v):
                        gw[u][v]["weight"] += 1.0
                        gw[u][v]["payload_mb"] += p_load
                    else:
                        gw.add_edge(u, v, weight=1.0, payload_mb=p_load)
                self.window_graphs[w_int] = gw

                # Bounded cycle detection (lengths 3 to max_cycle_length)
                cycles_w: List[List[str]] = []
                try:
                    for c in nx.simple_cycles(gw, length_bound=self.max_cycle_length):
                        if 3 <= len(c) <= self.max_cycle_length:
                            cycles_w.append(c)
                            for n in c:
                                self.cycle_nodes_cache.add(n)
                            for i in range(len(c)):
                                self.cycle_edges_cache.add((c[i], c[(i + 1) % len(c)]))
                        if len(cycles_w) >= 30:
                            break
                except Exception:
                    pass
                self.window_cycles[w_int] = cycles_w

        self._analyze_global_topology()
        return self

    def _analyze_global_topology(self) -> None:
        """Analyze cycles and centralities over the overall graph."""
        if len(self.graph.edges) <= 200:
            try:
                for c in nx.simple_cycles(self.graph, length_bound=self.max_cycle_length):
                    if 3 <= len(c) <= self.max_cycle_length:
                        self.cycle_paths_cache.append(c)
                        for n in c:
                            self.cycle_nodes_cache.add(n)
                        for i in range(len(c)):
                            self.cycle_edges_cache.add((c[i], c[(i + 1) % len(c)]))
                    if len(self.cycle_paths_cache) >= 100:
                        break
            except Exception:
                pass

        # Compute PageRank
        if len(self.graph) > 0:
            try:
                self.pagerank_scores = nx.pagerank(
                    self.graph,
                    alpha=self.pagerank_alpha,
                    max_iter=100,
                    tol=1e-5,
                    weight="weight",
                )
            except Exception:
                self.pagerank_scores = {n: 1.0 / len(self.graph) for n in self.graph.nodes()}
        else:
            self.pagerank_scores = {}

    def analyze_event(
        self,
        source_node: str,
        target_node: str,
        payload_mb: float = 1.0,
        protocol: str = "XROOTD_STREAM",
        window_step: Optional[int] = None,
    ) -> Tuple[float, Dict[str, Any]]:
        """Evaluate topological anomaly score and metadata for an interaction."""
        u = str(source_node)
        v = str(target_node)

        # Select target graph: window-specific if available, else global
        target_g = self.window_graphs.get(int(window_step), self.graph) if window_step is not None else self.graph
        w_cycles = self.window_cycles.get(int(window_step), self.cycle_paths_cache) if window_step is not None else self.cycle_paths_cache

        # Degree calculations in the active interaction context
        in_degree = target_g.in_degree(v) if target_g.has_node(v) else 0
        out_degree = target_g.out_degree(u) if target_g.has_node(u) else 0

        # Cycle membership: check if edge u -> v is in any cycle of length >= 3
        is_in_cycle = False
        matching_cycle: List[str] = []
        for p in w_cycles:
            for i in range(len(p)):
                if p[i] == u and p[(i + 1) % len(p)] == v:
                    is_in_cycle = True
                    matching_cycle = p
                    break
            if is_in_cycle:
                break

        if not is_in_cycle and (u, v) in self.cycle_edges_cache:
            is_in_cycle = True
            for p in self.cycle_paths_cache:
                if u in p and v in p:
                    matching_cycle = p
                    break

        # Bottleneck regimes
        is_incast = in_degree >= self.fan_in_threshold
        is_fanout = out_degree >= self.fan_out_threshold

        regime = "NOMINAL"
        if is_in_cycle and (is_incast or is_fanout):
            regime = "COMBINED_BOTTLENECK"
        elif is_in_cycle:
            regime = "ROUTING_CYCLE_LOOP"
        elif is_incast:
            regime = "FAN_IN_INCAST"
        elif is_fanout:
            regime = "FAN_OUT_FLOOD"

        # Topological Anomaly scoring
        score = 0.0
        if is_in_cycle:
            score += self.cycle_penalty
        if is_incast:
            score += self.bottleneck_penalty
        if is_fanout:
            score += self.bottleneck_penalty

        # Prior based on degree congestion and edge traffic volume
        score += min(0.30, (in_degree + out_degree) / 20.0)
        if payload_mb > 120.0:
            score += min(0.35, (payload_mb - 100.0) / 500.0)

        norm_score = float(np.clip(score, 0.0, 1.0))

        metadata = {
            "in_routing_cycle": bool(is_in_cycle),
            "cycle_path": matching_cycle,
            "target_fan_in_degree": int(in_degree),
            "source_fan_out_degree": int(out_degree),
            "bottleneck_regime": str(regime),
            "target_pagerank": round(float(self.pagerank_scores.get(v, 0.0)), 6),
            "source_pagerank": round(float(self.pagerank_scores.get(u, 0.0)), 6),
        }

        return norm_score, metadata

    def score_batch(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Score a batch of telemetry events with temporal window awareness."""
        scores = []
        meta_list = []
        has_window = "window_step" in df.columns

        for _, row in df.iterrows():
            w_step = int(row["window_step"]) if has_window else None
            s, m = self.analyze_event(
                source_node=str(row["source_node"]),
                target_node=str(row["target_node"]),
                payload_mb=float(row.get("payload_mb", 1.0)),
                protocol=str(row.get("protocol_type", "XROOTD_STREAM")),
                window_step=w_step,
            )
            scores.append(s)
            meta_list.append(m)
        return np.array(scores, dtype=np.float32), meta_list
