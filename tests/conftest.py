"""Pytest configuration and environment fixtures for GridPulse-XAI-RAG."""

import os
from pathlib import Path
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import apply_deterministic_seeds, load_config

TEST_CONFIG = load_config()
TEST_THREADS = str(TEST_CONFIG["system"].get("num_threads", 1))
os.environ["OMP_NUM_THREADS"] = TEST_THREADS
os.environ["MKL_NUM_THREADS"] = TEST_THREADS
os.environ["OPENBLAS_NUM_THREADS"] = TEST_THREADS
apply_deterministic_seeds(TEST_CONFIG)

# Enforce offline mode for Hugging Face Hub during testing
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
