from fastapi import APIRouter
from app.core.config import MODE
from app.pipelines.registry import FaceRegistry
from pathlib import Path

router = APIRouter()
registry = FaceRegistry()

@router.get("/health")
async def health():
    """Health check endpoint."""
    try:
        registry_data = registry.get_all()
        registry_count = len(registry_data)
        total_embeddings = sum(len(vectors) for vectors in registry_data.values())
        registry_path = registry.path
        
        return {
            "status": "ok",
            "mode": MODE,
            "registry": {
                "accessible": registry_path.exists(),
                "path": str(registry_path),
                "users_count": registry_count,
                "total_embeddings": total_embeddings,
                "users": list(registry_data.keys())
            }
        }
    except Exception as e:
        return {
            "status": "ok",
            "mode": MODE,
            "registry": {
                "accessible": False,
                "error": str(e)
            }
        }

