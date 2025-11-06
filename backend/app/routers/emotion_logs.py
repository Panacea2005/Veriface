from fastapi import APIRouter, Query, HTTPException
from typing import List
from app.core.config import STORE_DIR
import json

router = APIRouter()

@router.get("/api/emotion_logs")
async def get_emotion_logs(limit: int = Query(100, ge=1, le=1000)):
    """Return recent emotion logs from JSONL file."""
    path = STORE_DIR / "emotion_logs.jsonl"
    if not path.exists():
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Get last N lines
        recent = lines[-limit:]
        return [json.loads(line) for line in recent if line.strip()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


