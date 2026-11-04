"""Configuration loader and validator for GridPulse-XAI-RAG.

Loads centralized configuration from YAML, validates structural sections,
and computes parameter counts.
"""

from pathlib import Path
import random
from typing import Any, Dict
import yaml


def count_leaf_parameters(data: Any) -> int:
    """Recursively count scalar leaf parameters in a nested dictionary/list structure."""
    if isinstance(data, dict):
        total = 0
        for _, value in data.items():
            total += count_leaf_parameters(value)
        return total
    elif isinstance(data, list):
        # A list of primitives counts as 1 parameter collection, or count elements if nested dicts
        if data and isinstance(data[0], dict):
            return sum(count_leaf_parameters(item) for item in data)
        return 1
    else:
        return 1


def load_config(config_path: str | Path | None = None) -> Dict[str, Any]:
    """Load and validate the YAML configuration.

    Args:
        config_path: Path to config.yaml. Defaults to 'config/config.yaml' relative to package root.

    Returns:
        Nested configuration dictionary.

    Raises:
        FileNotFoundError: If configuration file is missing.
        ValueError: If structure is invalid or required leaf parameters are insufficient.
    """
    if config_path is None:
        # Locate config relative to this file's parent directory
        base_dir = Path(__file__).resolve().parent.parent
        resolved_path = base_dir / "config" / "config.yaml"
    else:
        resolved_path = Path(config_path).resolve()

    if not resolved_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {resolved_path}")

    with open(resolved_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if not isinstance(config, dict):
        raise ValueError("Invalid configuration file: root must be a YAML mapping.")

    # Validate mandatory top-level sections
    mandatory_sections = [
        "system",
        "telemetry_schema",
        "feature_engineering",
        "lightgbm",
        "lstm_autoencoder",
        "topology_graph",
        "hybrid_ensemble",
        "shap_xai",
        "rag_diagnostics",
        "api",
    ]
    for section in mandatory_sections:
        if section not in config:
            raise ValueError(f"Missing mandatory configuration section: '{section}'")

    leaf_count = count_leaf_parameters(config)
    if leaf_count < 60:
        raise ValueError(
            f"Configuration contains only {leaf_count} parameters, expected at least 60."
        )

    return config


def apply_deterministic_seeds(config: Dict[str, Any] | None = None) -> int:
    """Apply the configured random seeds and CPU thread limit."""
    if config is None:
        config = load_config()

    system = config.get("system", {})
    seed = int(system.get("random_seed", 42))
    random.seed(seed)

    import numpy as np
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, int(system.get("num_threads", 1))))
    deterministic = bool(system.get("deterministic_mode", False))
    torch.use_deterministic_algorithms(deterministic, warn_only=True)
    return seed
