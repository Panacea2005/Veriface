from fastapi import APIRouter, Query
from typing import Literal, List, Dict, Any
from fastapi import HTTPException
import numpy as np

from app.pipelines.registry import FaceRegistry

router = APIRouter()
registry = FaceRegistry()


def _pca_project_2d(vectors: np.ndarray) -> np.ndarray:
    """Project high-dimensional vectors to 2D using PCA via SVD (no sklearn).

    Returns an array of shape (N, 2).
    """
    if vectors.ndim != 2:
        raise ValueError("vectors must be 2D array [N, D]")
    if vectors.shape[0] == 0:
        return np.zeros((0, 2), dtype=np.float32)
    # Center
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    # SVD on covariance-equivalent matrix
    # For numerical stability when D is large relative to N, use economy SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]  # (2, D)
    projected = centered @ components.T  # (N, 2)
    return projected.astype(np.float32)


@router.get("/api/registry")
async def get_registry(
    include_vectors: bool = Query(False, description="Include raw embeddings"),
    project: Literal["none", "pca2d"] = Query("pca2d", description="Projection method"),
    limit_per_user: int = Query(200, ge=1, le=10000, description="Cap embeddings per user for response size")
):
    """Return registry identities and (optionally) embeddings and 2D projection.

    Response structure:
    {
      users: [userId, ...],
      counts: { userId: number },
      embeddings2d: [{ user_id, index, x, y }],
      vectors?: { userId: number[][] }
    }
    """
    data = registry.get_all()
    users: List[str] = list(data.keys())
    counts: Dict[str, int] = {uid: len(v) for uid, v in data.items()}

    # Collect vectors with optional limiting per user to control payload size
    stacked: List[np.ndarray] = []
    labels: List[str] = []
    indices: List[int] = []
    for uid in users:
        user_vecs = data[uid][:limit_per_user]
        if len(user_vecs) == 0:
            continue
        arr = np.asarray(user_vecs, dtype=np.float32)
        stacked.append(arr)
        labels.extend([uid] * arr.shape[0])
        indices.extend(list(range(arr.shape[0])))

    embeddings2d: List[Dict[str, Any]] = []
    if len(stacked) > 0 and project == "pca2d":
        all_vecs = np.vstack(stacked)
        proj = _pca_project_2d(all_vecs)
        for (uid, idx, p) in zip(labels, indices, proj):
            embeddings2d.append({
                "user_id": uid,
                "index": int(idx),
                "x": float(p[0]),
                "y": float(p[1])
            })

    response: Dict[str, Any] = {
        "users": users,
        "counts": counts,
        "embeddings2d": embeddings2d,
    }

    if include_vectors:
        response["vectors"] = {uid: data[uid][:limit_per_user] for uid in users}

    return response


@router.delete("/api/registry/{user_id}")
async def delete_user(user_id: str):
    """Delete an entire user and all embeddings."""
    # Always reload
    _ = registry.get_all()
    removed = registry.remove_user(user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="User not found")
    # Confirm state
    updated = registry.get_all()
    return {"status": "ok", "deleted": user_id, "users_count": len(updated), "total_embeddings": sum(len(v) for v in updated.values())}


@router.delete("/api/registry/{user_id}/{index}")
async def delete_embedding(user_id: str, index: int):
    """Delete a single embedding by index for a user."""
    # Always reload
    _ = registry.get_all()
    ok = registry.remove_embedding(user_id, index)
    if not ok:
        raise HTTPException(status_code=404, detail="Embedding not found")
    updated = registry.get_all()
    return {"status": "ok", "user_id": user_id, "deleted_index": index, "remaining": len(updated.get(user_id, []))}



