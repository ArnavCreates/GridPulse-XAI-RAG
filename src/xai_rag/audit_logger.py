"""Deterministic SHA-256 hash-chained JSONL diagnostic audit logger.

Logs every /diagnose event with anonymized node digests, tri-modal scores,
SHAP drivers, and cryptographic hash chaining for tamper-evident records.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GENESIS_HASH = "0" * 64


class AuditLogger:
    """Cryptographic hash-chained audit logger for diagnostic events."""

    def __init__(
        self,
        log_path: Optional[str | Path] = None,
        deterministic_mode: bool = False,
    ):
        self.deterministic_mode = bool(deterministic_mode)
        if log_path is None:
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.log_path = base_dir / "logs" / "diagnostic_audit.jsonl"
        else:
            self.log_path = Path(log_path)

        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_hash = self._get_latest_hash()

    def _get_latest_hash(self) -> str:
        """Read the hash of the last entry in the log file, or return GENESIS_HASH."""
        if not self.log_path.exists() or self.log_path.stat().st_size == 0:
            return GENESIS_HASH

        last_line = ""
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    last_line = line.strip()

        if not last_line:
            return GENESIS_HASH

        try:
            entry = json.loads(last_line)
            return entry.get("sha256_chain_hash", GENESIS_HASH)
        except Exception:
            return GENESIS_HASH

    @staticmethod
    def hash_node_id(node_id: str) -> str:
        """Create SHA-256 anonymized digest for a node identifier."""
        return hashlib.sha256(node_id.encode("utf-8")).hexdigest()[:16]

    def log_diagnostic_event(
        self,
        event_id: str,
        source_node: str,
        target_node: str,
        hybrid_score: float,
        is_anomaly: bool,
        severity: str,
        top_shap_drivers: List[str],
        retrieved_runbook_ids: List[str],
        timestamp_utc: Optional[str] = None,
        window_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Record and append an audit event with chained hash."""
        if timestamp_utc is not None:
            ts = timestamp_utc
        elif self.deterministic_mode:
            epoch = datetime(2026, 1, 1, tzinfo=timezone.utc)
            ts = (epoch + timedelta(seconds=max(0, int(window_step or 0)))).isoformat(timespec="milliseconds")
        else:
            ts = datetime.now(timezone.utc).isoformat()
        src_hash = self.hash_node_id(source_node)
        dst_hash = self.hash_node_id(target_node)

        payload_to_hash = {
            "prev_hash": self.last_hash,
            "timestamp_utc": ts,
            "event_id": str(event_id),
            "source_node_hash": src_hash,
            "target_node_hash": dst_hash,
            "hybrid_score": round(float(hybrid_score), 4),
            "is_anomaly": bool(is_anomaly),
            "severity": str(severity),
            "top_shap_drivers": top_shap_drivers,
            "retrieved_runbook_ids": retrieved_runbook_ids,
        }

        # Canonical deterministic JSON string representation
        canonical_str = json.dumps(payload_to_hash, sort_keys=True)
        current_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

        entry = dict(payload_to_hash)
        entry["sha256_chain_hash"] = current_hash

        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")

        self.last_hash = current_hash
        return entry

    @staticmethod
    def verify_chain(log_path: str | Path) -> Tuple[bool, int, Optional[str]]:
        """Verify the cryptographic hash-chain integrity of a JSONL log file.

        Returns:
            Tuple of (is_valid: bool, verified_entries_count: int, error_message: Optional[str])
        """
        p = Path(log_path)
        if not p.exists():
            return False, 0, "Log file does not exist"

        prev_hash = GENESIS_HASH
        count = 0

        with open(p, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line_str = line.strip()
                if not line_str:
                    continue

                try:
                    entry = json.loads(line_str)
                except Exception as e:
                    return False, count, f"JSON parse error at line {line_no}: {e}"

                claimed_prev = entry.get("prev_hash")
                claimed_current = entry.get("sha256_chain_hash")

                if claimed_prev != prev_hash:
                    return (
                        False,
                        count,
                        f"Hash break at line {line_no}: claimed prev {claimed_prev} != expected {prev_hash}",
                    )

                # Recompute hash
                payload = {
                    "prev_hash": claimed_prev,
                    "timestamp_utc": entry["timestamp_utc"],
                    "event_id": entry["event_id"],
                    "source_node_hash": entry["source_node_hash"],
                    "target_node_hash": entry["target_node_hash"],
                    "hybrid_score": entry["hybrid_score"],
                    "is_anomaly": entry["is_anomaly"],
                    "severity": entry["severity"],
                    "top_shap_drivers": entry["top_shap_drivers"],
                    "retrieved_runbook_ids": entry["retrieved_runbook_ids"],
                }
                canonical_str = json.dumps(payload, sort_keys=True)
                expected_hash = hashlib.sha256(canonical_str.encode("utf-8")).hexdigest()

                if claimed_current != expected_hash:
                    return (
                        False,
                        count,
                        f"Tamper detected at line {line_no}: hash {claimed_current} != recomputed {expected_hash}",
                    )

                prev_hash = claimed_current
                count += 1

        return True, count, None
