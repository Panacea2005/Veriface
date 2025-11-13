"""Utility script to sanity-check the ArcFace embedding model weights.

This script validates that the configured weights file can be deserialized
and produces a 512-D output with a dummy forward pass.
"""
from __future__ import annotations

import sys
from pathlib import Path
import os

import numpy as np
import torch

try:  # torch < 2.6 does not expose add_safe_globals
    from torch.serialization import add_safe_globals  # type: ignore
except ImportError:  # pragma: no cover - fallback for old torch
    add_safe_globals = None  # type: ignore

# Ensure project root is on sys.path when running as standalone script
BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.pipelines.arcface_model import get_model

MODELS_DIR = Path(__file__).resolve().parent.parent / "app" / "models"
default_weights = MODELS_DIR / "modelA_best.pth"
weights_env = Path(os.environ.get("MODEL_WEIGHTS_PATH", str(default_weights)))

if not weights_env.is_absolute():
    # Interpret relative paths relative to backend root
    weights_env = (Path(__file__).resolve().parent.parent / weights_env).resolve()

print(f"[INFO] Models directory: {MODELS_DIR}")
print(f"[INFO] Weights path: {weights_env}")

# Allow numpy scalar objects when loading legacy checkpoints
if add_safe_globals is not None:
    try:
        add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass

path = weights_env
if not path.exists():
    print(f"[ERROR] Weights file not found at: {path}")
    sys.exit(1)

size_mb = path.stat().st_size / 1024 / 1024
print(f"[OK] Weights FOUND ({size_mb:.1f} MB) at {path}")
print(f"[INFO] Loading weights...")

try:
    model = get_model(input_size=[112, 112], num_layers=100, mode="ir")

    # Torch >= 2.6 defaults weights_only=True; force False for legacy checkpoints
    try:
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.6
        checkpoint = torch.load(str(path), map_location="cpu")

    if isinstance(checkpoint, dict) and not ("state_dict" in checkpoint or "model" in checkpoint):
        state_dict = checkpoint
    else:
        state_dict = checkpoint.get("state_dict", checkpoint.get("model", checkpoint))

    model.load_state_dict(state_dict, strict=False)

    model.eval()
    with torch.no_grad():
        output = model(torch.randn(1, 3, 112, 112))
    if output.shape[-1] == 512:
        print(f"[OK] Output dim {output.shape[-1]} (expected 512)")
        sys.exit(0)
    else:
        print(f"[ERROR] Unexpected output dim {output.shape[-1]} (expected 512)")
        sys.exit(2)

except Exception as exc:  # pragma: no cover - diagnostic output only
    msg = str(exc).replace("\n", " ")[:200]
    print(f"[ERROR] Failed to load/infer - {msg}")
    sys.exit(3)

sys.exit(0)
