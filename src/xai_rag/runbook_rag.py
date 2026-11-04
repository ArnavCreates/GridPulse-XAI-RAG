"""Local CPU ChromaDB + SentenceTransformers (all-MiniLM-L6-v2) Runbook RAG Engine.

Indexes technical operational recovery runbooks, executes dense vector retrieval
guided by SHAP root-cause features and topology regimes, and synthesizes recovery actions.
"""

import os
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

from pathlib import Path
import re
from typing import Any, Dict, List, Optional


class RunbookRAGEngine:
    """RAG engine indexing operational runbooks into ChromaDB for contextual diagnosis."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "gridpulse_runbooks",
        top_k: int = 2,
        runbooks_dir: Optional[str | Path] = None,
        encoder: Any = None,
        client: Any = None,
    ):
        if config:
            self.model_name = config.get("embedding_model_name", embedding_model_name)
            self.collection_name = config.get("collection_name", collection_name)
            self.top_k = int(config.get("top_k_runbooks", top_k))
            self.similarity_metric = str(config.get("similarity_metric", "cosine"))
            self.min_relevance_score = float(config.get("min_relevance_score", 0.0))
            self.chunk_size_chars = int(config.get("chunk_size_chars", 800))
            self.chunk_overlap_chars = int(config.get("chunk_overlap_chars", 100))
            self.query_instruction = str(config.get("query_instruction", ""))
            self.excluded_header_sections = {
                str(name).strip().casefold()
                for name in config.get("excluded_header_sections", ["Metadata", "Symptom Signature"])
            }
            cfg_dir = config.get("runbooks_dir", "runbooks")
            self.runbooks_dir = Path(cfg_dir)
        else:
            self.model_name = embedding_model_name
            self.collection_name = collection_name
            self.top_k = top_k
            self.similarity_metric = "cosine"
            self.min_relevance_score = 0.0
            self.chunk_size_chars = 800
            self.chunk_overlap_chars = 100
            self.query_instruction = ""
            self.excluded_header_sections = {"metadata", "symptom signature"}
            self.runbooks_dir = Path(runbooks_dir or "runbooks")

        if not self.runbooks_dir.is_absolute():
            # Resolve relative to package root
            base_dir = Path(__file__).resolve().parent.parent.parent
            self.runbooks_dir = base_dir / self.runbooks_dir

        # Prefer the local cache, but download the model when this is a fresh host.
        if encoder is None:
            from sentence_transformers import SentenceTransformer

            try:
                encoder = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                    local_files_only=True,
                )
            except Exception:
                encoder = SentenceTransformer(
                    self.model_name,
                    device="cpu",
                    local_files_only=False,
                )
        self.encoder = encoder

        # Initialize Ephemeral / Local ChromaDB Client
        if client is None:
            import chromadb

            client = chromadb.Client()
        self.client = client
        # Reset or create collection
        try:
            self.client.delete_collection(name=self.collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.similarity_metric},
        )
        self.is_indexed = False

    def _parse_runbook_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Parse markdown runbook into logical semantic chunks."""
        text = file_path.read_text(encoding="utf-8")

        # Extract Runbook ID and Severity from Metadata section
        runbook_id_match = re.search(r"Runbook ID:\*\*\s*([A-Z0-9\-]+)", text)
        runbook_id = runbook_id_match.group(1) if runbook_id_match else file_path.stem

        severity_match = re.search(r"Severity Level:\*\*\s*([A-Za-z /]+)", text)
        severity = severity_match.group(1).strip() if severity_match else "HIGH"

        failure_match = re.search(r"Target Failure Mode:\*\*\s*([^\r\n]+)", text)
        failure_mode = failure_match.group(1).strip() if failure_match else "Distributed Anomaly"

        sections: Dict[str, str] = {}
        parts = re.split(r"(?m)^##\s+", text)
        for section in parts[1:]:
            lines = section.strip().splitlines()
            if lines:
                sections[lines[0].strip()] = "\n".join(lines[1:]).strip()

        header_context = f"Runbook {runbook_id}. Failure mode: {failure_mode}. Severity: {severity}."

        actionable_titles = (
            "Diagnostic Verification",
            "Remediation & Recovery Procedure",
            "Prevention & Hardening",
        )
        chunks = []
        for title in actionable_titles:
            if title.casefold() in self.excluded_header_sections:
                continue
            body = sections.get(title, "")
            if not body:
                continue
            for chunk_index, action_part in enumerate(self._split_action_text(body)):
                text_chunk = f"{header_context}\n\n## {title}\n{action_part}".strip()
                slug = re.sub(r"[^a-z0-9]+", "_", title.casefold()).strip("_")
                chunks.append({
                    "chunk_id": f"{runbook_id}_{slug}_{chunk_index}",
                    "runbook_id": runbook_id,
                    "section_title": title,
                    "severity": severity,
                    "failure_mode": failure_mode,
                    "text": text_chunk,
                    "source_file": file_path.name,
                })

        return chunks

    def _split_action_text(self, text: str) -> List[str]:
        if len(text) <= self.chunk_size_chars:
            return [text]
        pieces: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size_chars)
            if end < len(text):
                boundary = text.rfind("\n", start, end)
                if boundary > start:
                    end = boundary
            pieces.append(text[start:end].strip())
            if end >= len(text):
                break
            overlap_start = max(start + 1, end - self.chunk_overlap_chars)
            next_line = text.find("\n", overlap_start, end)
            if next_line >= 0:
                start = next_line + 1
            else:
                next_space = text.find(" ", overlap_start, end)
                start = next_space + 1 if next_space >= 0 else end
        return pieces

    def index_runbooks(self) -> int:
        """Parse, embed, and index all markdown runbooks in the target directory."""
        if not self.runbooks_dir.exists():
            raise FileNotFoundError(f"Runbooks directory not found at: {self.runbooks_dir}")

        md_files = sorted(list(self.runbooks_dir.glob("*.md")))
        if not md_files:
            raise FileNotFoundError(f"No markdown runbooks found in: {self.runbooks_dir}")

        all_chunks: List[Dict[str, Any]] = []
        for f in md_files:
            chunks = self._parse_runbook_file(f)
            all_chunks.extend(chunks)

        texts = [c["text"] for c in all_chunks]
        ids = [c["chunk_id"] for c in all_chunks]
        metadatas = [
            {
                "runbook_id": c["runbook_id"],
                "section_title": c["section_title"],
                "severity": c["severity"],
                "failure_mode": c["failure_mode"],
                "source_file": c["source_file"],
            }
            for c in all_chunks
        ]

        # Generate 384-dim embeddings on CPU
        embeddings = self.encoder.encode(texts, show_progress_bar=False, device="cpu")
        embeddings_list = embeddings.tolist()

        self.collection.add(
            ids=ids,
            embeddings=embeddings_list,
            metadatas=metadatas,
            documents=texts,
        )
        self.is_indexed = True
        return len(all_chunks)

    def formulate_query(
        self,
        protocol_type: str,
        shap_drivers: List[Dict[str, Any]],
        bottleneck_regime: str,
        temporal_drift_detected: bool = False,
    ) -> str:
        """Synthesize semantic search query from tri-modal alert features."""
        top_driver_strs = [
            f"{d['feature']} ({d['direction']}{d['shap_impact']})"
            for d in shap_drivers[:4]
        ]
        drivers_text = ", ".join(top_driver_strs)
        query_parts = ["Distributed computing grid telemetry anomaly."]
        if self.query_instruction:
            query_parts.append(self.query_instruction)
        if temporal_drift_detected:
            query_parts.append(
                "Mode B: stealth temporal sequence drift and oscillation detected across sliding window. "
                "CPU thermal stall and worker load oscillation."
            )
        if bottleneck_regime in ("ROUTING_CYCLE_LOOP", "COMBINED_BOTTLENECK"):
            query_parts.append(
                "Mode C: network topology routing cycle loop storm detected. "
                "In routing cycle: True. Topology regime: ROUTING_CYCLE_LOOP."
            )
        elif bottleneck_regime in ("FAN_IN_INCAST", "FAN_OUT_FLOOD"):
            query_parts.append(
                f"Mode D: network topology incast congestion detected. "
                f"Topology regime: {bottleneck_regime}. Multi-source synchronized transfer burst."
            )
        elif not temporal_drift_detected:
            query_parts.append(
                "Mode A: tabular resource surge and queue saturation. Queue buffer overflow pressure."
            )

        query_parts.append(f"Protocol: {protocol_type}.")
        query_parts.append(f"Primary SHAP attribution drivers: {drivers_text}.")
        return " ".join(query_parts)

    def retrieve_recommendations(
        self,
        protocol_type: str,
        shap_drivers: List[Dict[str, Any]],
        bottleneck_regime: str,
        temporal_drift_detected: bool = False,
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve top matching runbook procedures and mitigation actions."""
        if not self.is_indexed:
            self.index_runbooks()

        k = top_k or self.top_k
        query_text = self.formulate_query(
            protocol_type,
            shap_drivers,
            bottleneck_regime,
            temporal_drift_detected=temporal_drift_detected,
        )
        query_embedding = self.encoder.encode([query_text], device="cpu").tolist()

        # Query multiple candidates to deduplicate across distinct runbooks
        n_fetch = len(self.collection.get()["ids"])
        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=n_fetch,
            include=["documents", "metadatas", "distances"],
        )

        recommendations = []
        seen_runbooks = set()
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if results.get("metadatas") else [{}] * len(docs)
            distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, distances):
                rb_id = meta.get("runbook_id", "RB-GRID-UNKNOWN")
                if rb_id in seen_runbooks:
                    continue
                seen_runbooks.add(rb_id)

                if self.similarity_metric == "cosine":
                    similarity = max(0.0, 1.0 - (dist / 2.0))
                else:
                    similarity = 1.0 / (1.0 + max(0.0, float(dist)))
                similarity = round(similarity, 4)
                if similarity < self.min_relevance_score:
                    continue

                cli_blocks = re.findall(r"```bash(.*?)```", doc, re.DOTALL)
                mitigation_snippet = cli_blocks[0].strip() if cli_blocks else ""
                section_title = meta.get("section_title", "Operational Recovery")
                action_body = doc.split(f"## {section_title}\n", 1)[-1].strip()
                recommended_action = mitigation_snippet or action_body

                recommendations.append({
                    "runbook_id": rb_id,
                    "section_title": section_title,
                    "severity": meta.get("severity", "HIGH"),
                    "failure_mode": meta.get("failure_mode", "Grid Anomaly"),
                    "source_file": meta.get("source_file", ""),
                    "similarity_score": similarity,
                    "recommended_action": recommended_action,
                    "full_section_snippet": doc[:450],
                })

                if len(recommendations) >= k:
                    break

        return recommendations
