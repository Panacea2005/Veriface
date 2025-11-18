import os
from pathlib import Path
from typing import Literal

# Load environment variables from backend/.env if present
try:
    from dotenv import load_dotenv  # type: ignore
    _ENV_PATH = Path(__file__).parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(dotenv_path=_ENV_PATH, override=False)
except Exception:
    # dotenv is optional; ignore if not installed
    pass

MODE: Literal["heur", "onnx"] = os.getenv("MODE", "heur")
SIMILARITY_METRIC: Literal["cosine", "euclidean"] = os.getenv("SIMILARITY_METRIC", "cosine")  # backend-wide default

# Model selection: "A", "B", or "deepface" (default: "A")
# - "A": Use Model A (modelA_best.pth)
# - "B": Use Model B (modelB_best.pth)
# - "deepface": Use DeepFace ArcFace (ignores DEEPFACE_ONLY)
_model_type_raw = os.getenv("MODEL_TYPE", "A").lower()
if _model_type_raw not in ["a", "b", "deepface"]:
    _model_type_raw = "a"
MODEL_TYPE: Literal["A", "B", "deepface"] = _model_type_raw if _model_type_raw == "deepface" else _model_type_raw.upper()  # type: ignore

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
LABEL_MAP_PATH = BASE_DIR / "core" / "label_map.json"

