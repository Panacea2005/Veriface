import os
from pathlib import Path
from typing import Literal

MODE: Literal["heur", "onnx"] = os.getenv("MODE", "heur")

# Paths
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / "models"
STORE_DIR = BASE_DIR / "store"
STORE_DIR.mkdir(exist_ok=True)
REGISTRY_PATH = STORE_DIR / "registry.json"

# CORS
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000,http://localhost:3001").split(",")

# Config files
THRESHOLDS_PATH = BASE_DIR / "core" / "thresholds.yaml"
PREPROCESS_PATH = BASE_DIR / "core" / "preprocess.json"
LABEL_MAP_PATH = BASE_DIR / "core" / "label_map.json"
METRICS_PATH = BASE_DIR / "core" / "metrics.json"

