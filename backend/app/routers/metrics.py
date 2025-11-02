from fastapi import APIRouter, HTTPException
from app.core.config import METRICS_PATH
import json

router = APIRouter()

@router.get("/api/roc")
async def get_roc_metrics():
    """
    Get ROC curve metrics (AUC, EER, etc.).
    Returns static metrics from config file.
    Returns 404 if metrics file doesn't exist (no evaluation performed yet).
    """
    try:
        if METRICS_PATH.exists():
            with open(METRICS_PATH) as f:
                metrics = json.load(f)
            # Validate that metrics exist
            if "auc" not in metrics or "accuracy" not in metrics:
                raise HTTPException(status_code=404, detail="Metrics file exists but is invalid")
            return metrics
        else:
            # No metrics file - evaluation hasn't been run yet
            raise HTTPException(status_code=404, detail="No metrics available. Model evaluation required.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

