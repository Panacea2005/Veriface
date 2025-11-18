import argparse
import json
import math
from itertools import combinations
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def load_registry(registry_path: Path) -> Dict[str, List[List[float]]]:
    """Load registry JSON and return mapping user_id -> list of embeddings."""
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found at {registry_path}")

    with registry_path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    result: Dict[str, List[List[float]]] = {}
    for user_id, value in raw.items():
        if isinstance(value, dict):
            embeddings = value.get("embeddings") or value.get("vectors") or value.get("data")
        else:
            embeddings = value
        if embeddings is None:
            embeddings = []
        result[user_id] = embeddings
    return result


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-8
    if denom <= 0:
        return float("nan")
    return float(np.dot(vec_a, vec_b) / denom)


def analyze_registry(
    registry: Dict[str, List[List[float]]],
    highlight_same: float,
    top_cross: int,
) -> Tuple[List[str], List[Tuple[float, str, str]]]:
    per_user_logs: List[str] = []
    cross_user_pairs: List[Tuple[float, str, str]] = []

    for user_id, vectors in registry.items():
        arr = np.array(vectors, dtype=np.float32)
        count = len(arr)
        if count == 0:
            per_user_logs.append(f"{user_id}: no embeddings saved")
            continue
        norms = np.linalg.norm(arr, axis=1)
        intramax = 0.0
        trimask = ""
        if count > 1:
            sims = (arr @ arr.T) / ((norms[:, None] + 1e-8) * (norms[None, :] + 1e-8))
            upper = sims - np.eye(count)
            intramax = float(upper.max())
            trimask = " ⚠" if intramax >= highlight_same else ""
        per_user_logs.append(
            f"{user_id}: {count} embeddings | norm mu={norms.mean():.4f} sigma={norms.std():.4f} | "
            f"max self-cos={intramax:.4f}{trimask}"
        )

    for (user_a, vecs_a), (user_b, vecs_b) in combinations(registry.items(), 2):
        arr_a = [np.array(v, dtype=np.float32) for v in vecs_a]
        arr_b = [np.array(v, dtype=np.float32) for v in vecs_b]
        if not arr_a or not arr_b:
            continue
        max_sim = -1.0
        for va in arr_a:
            for vb in arr_b:
                sim = cosine_similarity(va, vb)
                if math.isnan(sim):
                    continue
                if sim > max_sim:
                    max_sim = sim
        if max_sim >= 0:
            cross_user_pairs.append((max_sim, user_a, user_b))

    cross_user_pairs.sort(reverse=True, key=lambda x: x[0])
    return per_user_logs, cross_user_pairs[:top_cross]


def main():
    parser = argparse.ArgumentParser(
        description="Inspect registry embeddings for duplicate or overlapping users."
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=None,
        help="Path to registry.json (defaults to app.core.config.REGISTRY_PATH)",
    )
    parser.add_argument(
        "--highlight-same",
        type=float,
        default=0.98,
        help="Highlight intra-user similarities above this cosine threshold (default 0.98).",
    )
    parser.add_argument(
        "--top-cross",
        type=int,
        default=5,
        help="Number of highest cross-user similarity pairs to display (default 5).",
    )
    args = parser.parse_args()

    if args.registry is None:
        import sys

        ROOT = Path(__file__).resolve().parents[1]
        if str(ROOT) not in sys.path:
            sys.path.insert(0, str(ROOT))
        from app.core.config import REGISTRY_PATH

        registry_path = REGISTRY_PATH
    else:
        registry_path = args.registry

    registry = load_registry(registry_path)
    per_user_logs, cross_pairs = analyze_registry(registry, args.highlight_same, args.top_cross)

    print(f"Registry file: {registry_path}")
    print(f"Total users: {len(registry)}\n")
    print("Per-user stats:")
    for line in per_user_logs:
        print(f"  - {line}")

    if cross_pairs:
        print(f"\nTop {len(cross_pairs)} cross-user cosine similarities:")
        for rank, (sim, ua, ub) in enumerate(cross_pairs, 1):
            flag = " ⚠" if sim >= args.highlight_same else ""
            print(f"  {rank:02d}. cos={sim:.4f} between {ua} and {ub}{flag}")
    else:
        print("\nNo cross-user pairs found.")


if __name__ == "__main__":
    main()

