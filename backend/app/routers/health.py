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
        
        # Handle both old and new registry formats
        total_embeddings = 0
        for user_id, data in registry_data.items():
            if isinstance(data, dict) and 'embeddings' in data:
                # New format: {user_id: {name, embeddings}}
                total_embeddings += len(data['embeddings'])
            elif isinstance(data, list):
                # Old format: {user_id: [[embeddings]]}
                total_embeddings += len(data)
        
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

