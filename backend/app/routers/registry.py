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
    
    # Handle edge cases: if only 1 vector, return [0, 0]
    if vectors.shape[0] == 1:
        return np.zeros((1, 2), dtype=np.float32)
    
    # Center
    mean = vectors.mean(axis=0, keepdims=True)
    centered = vectors - mean
    
    # SVD on covariance-equivalent matrix
    # For numerical stability when D is large relative to N, use economy SVD
    u, s, vt = np.linalg.svd(centered, full_matrices=False)
    
    # Ensure we have at least 2 components
    # If we have fewer components than needed, pad with zeros
    n_components = min(2, vt.shape[0])
    if n_components < 2:
        # Pad vt with zeros to get 2 components
        components = np.zeros((2, vt.shape[1]), dtype=vt.dtype)
        components[:n_components] = vt[:n_components]
    else:
        components = vt[:2]  # (2, D)
    
    projected = centered @ components.T  # (N, 2)
    
    # Ensure output is always (N, 2)
    if projected.shape[1] < 2:
        padded = np.zeros((projected.shape[0], 2), dtype=np.float32)
        padded[:, :projected.shape[1]] = projected
        return padded.astype(np.float32)
    
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
    # Always reload to get latest data
    data = registry.get_all()
    users: List[str] = list(data.keys())
    
    # Ensure counts are accurate - count ALL actual vectors (not limited)
    # Also collect names
    counts: Dict[str, int] = {}
    names: Dict[str, str] = {}
    
    for uid, user_data in data.items():
        # Handle both new format {name, embeddings} and legacy format [[embeddings]]
        if isinstance(user_data, dict) and "embeddings" in user_data:
            counts[uid] = len(user_data["embeddings"])
            names[uid] = user_data.get("name", uid)
        elif isinstance(user_data, list):
            counts[uid] = len(user_data)
            names[uid] = uid  # Legacy format - use user_id as name
        else:
            counts[uid] = 0
            names[uid] = uid

    # Collect vectors with optional limiting per user to control payload size
    stacked: List[np.ndarray] = []
    labels: List[str] = []
    indices: List[int] = []
    for uid in users:
        # Extract embeddings from new or legacy format
        user_data = data[uid]
        if isinstance(user_data, dict) and "embeddings" in user_data:
            user_vecs = user_data["embeddings"][:limit_per_user]
        elif isinstance(user_data, list):
            user_vecs = user_data[:limit_per_user]
        else:
            continue
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
        "names": names,  # Add names mapping
        "counts": counts,  # Total counts (not limited)
        "embeddings2d": embeddings2d,
    }

    if include_vectors:
        # Return limited vectors for display, but counts show total
        vectors_dict: Dict[str, List[List[float]]] = {}
        for uid in users:
            user_data = data.get(uid, [])
            # Extract embeddings from new or legacy format
            if isinstance(user_data, dict) and "embeddings" in user_data:
                user_vectors = user_data["embeddings"]
            elif isinstance(user_data, list):
                user_vectors = user_data
            else:
                user_vectors = []
            if user_vectors and len(user_vectors) > 0:
                # Ensure vectors are lists of lists (not numpy arrays)
                limited = user_vectors[:limit_per_user]
                vectors_dict[uid] = [[float(x) for x in vec] if isinstance(vec, (list, tuple)) else vec.tolist() if hasattr(vec, 'tolist') else vec for vec in limited]
            else:
                vectors_dict[uid] = []
        response["vectors"] = vectors_dict
        # Debug logging
        import sys
        print(f"[DEBUG] Registry: Returning vectors for {len(vectors_dict)} users", file=sys.stderr)
        for uid, vecs in vectors_dict.items():
            print(f"[DEBUG] Registry: User '{uid}': {len(vecs)} vectors", file=sys.stderr)

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


@router.delete("/api/registry")
async def clear_registry():
    """Clear all users and embeddings from registry. Use with caution!"""
    # Force reload before clearing
    _ = registry.get_all()
    deleted_count = registry.clear_all()
    # Verify it's actually cleared
    verify_data = registry.get_all()
    return {
        "status": "ok",
        "message": "All users and embeddings cleared",
        "deleted_users": deleted_count,
        "verified_empty": len(verify_data) == 0,
        "remaining_users": len(verify_data)
    }



