"""Pydantic v2 schemas for GridPulse-XAI-RAG telemetry ingest and diagnostic responses."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, field_validator

from ..config_loader import load_config

_CONFIG = load_config()
_SCHEMA_CONFIG = _CONFIG["telemetry_schema"]
_API_CONFIG = _CONFIG["api"]


class TelemetryEventInput(BaseModel):
    """Input payload representing a single 10-column distributed grid telemetry event."""

    window_step: int = Field(..., ge=1, description="Discrete telemetry time-window step")
    protocol_type: str = Field(..., description="Transfer protocol category")
    payload_mb: float = Field(
        ...,
        ge=_SCHEMA_CONFIG["min_payload_mb"],
        le=_SCHEMA_CONFIG["max_payload_mb"],
        description="Packet burst or transfer volume in MB",
    )
    source_node: str = Field(..., min_length=2, description="Source compute node ID")
    src_cpu_load_prev: float = Field(
        ...,
        ge=_SCHEMA_CONFIG["min_cpu_load"],
        le=_SCHEMA_CONFIG["max_cpu_load"],
        description="Source CPU load percentage prior",
    )
    src_cpu_load_curr: float = Field(
        ...,
        ge=_SCHEMA_CONFIG["min_cpu_load"],
        le=_SCHEMA_CONFIG["max_cpu_load"],
        description="Source CPU load percentage current",
    )
    target_node: str = Field(..., min_length=2, description="Target storage/aggregator node ID")
    dst_queue_depth_prev: float = Field(
        ...,
        ge=_SCHEMA_CONFIG["min_queue_depth"],
        le=_SCHEMA_CONFIG["max_queue_depth"],
        description="Destination queue depth prior",
    )
    dst_queue_depth_curr: float = Field(
        ...,
        ge=_SCHEMA_CONFIG["min_queue_depth"],
        le=_SCHEMA_CONFIG["max_queue_depth"],
        description="Destination queue depth current",
    )
    is_anomaly: Optional[int] = Field(default=None, description="Optional ground-truth anomaly label")
    event_id: Optional[str] = Field(default=None, description="Optional caller event identifier")

    @field_validator("protocol_type")
    @classmethod
    def validate_protocol_type(cls, value: str) -> str:
        allowed = _SCHEMA_CONFIG["allowed_protocols"]
        if value not in allowed:
            raise ValueError(f"protocol_type must be one of {allowed}")
        return value


class TelemetryBatchInput(BaseModel):
    """Batch input payload containing multiple telemetry events."""

    events: List[TelemetryEventInput] = Field(
        ...,
        min_length=1,
        max_length=_API_CONFIG["max_batch_size"],
    )


class ShapDriver(BaseModel):
    """Individual SHAP-attributed root-cause telemetry feature."""

    feature: str
    observed_value: float
    shap_impact: float
    impact_units: str = "log-odds"
    direction: str = Field(..., pattern=r"^[+-]$")


class TopologyMetadata(BaseModel):
    """NetworkX structural graph topology diagnostics."""

    in_routing_cycle: bool
    cycle_path: List[str] = Field(default_factory=list)
    target_fan_in_degree: int
    source_fan_out_degree: int
    bottleneck_regime: str
    target_pagerank: Optional[float] = 0.0
    source_pagerank: Optional[float] = 0.0


class RunbookRecommendation(BaseModel):
    """ChromaDB RAG operational recovery runbook procedure."""

    runbook_id: str
    section_title: str
    severity: str
    failure_mode: str
    similarity_score: float
    recommended_action: str
    full_section_snippet: Optional[str] = None


class ModelScores(BaseModel):
    """Individual component and hybrid anomaly probability scores."""

    lgbm_score: float
    lstm_ae_score: float
    topology_score: float
    hybrid_score: float


class DiagnosticResponse(BaseModel):
    """Complete diagnostic report with XAI root-cause attribution and RAG recommendations."""

    event_id: str
    timestamp_utc: str
    is_anomaly: bool
    severity: str
    hybrid_score: float
    model_scores: ModelScores
    top_shap_drivers: List[ShapDriver]
    topology_metadata: TopologyMetadata
    runbook_recommendations: List[RunbookRecommendation]
    execution_time_ms: float


class HealthResponse(BaseModel):
    """Service health and readiness check response."""

    status: str
    version: str
    models_loaded: Dict[str, bool]
    runbooks_indexed_count: int
    device: str


class RootResponse(BaseModel):
    """Root service information and schema summary."""

    service: str
    version: str
    status: str
    schema_columns: List[str]
    endpoints: List[str]
