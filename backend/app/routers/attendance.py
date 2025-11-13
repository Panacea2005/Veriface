from fastapi import APIRouter, Query, HTTPException, Response
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import csv
import io
from app.core.config import STORE_DIR
from app.pipelines.registry import FaceRegistry

router = APIRouter()
registry = FaceRegistry()

ATTENDANCE_LOG_PATH = STORE_DIR / "attendance.jsonl"
USER_METADATA_PATH = STORE_DIR / "user_metadata.json"
COOLDOWN_MINUTES = 5  # Prevent duplicate check-ins within 5 minutes

def _load_attendance_logs(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Load attendance logs from JSONL file."""
    if not ATTENDANCE_LOG_PATH.exists():
        return []
    
    try:
        with open(ATTENDANCE_LOG_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        logs = []
        for line in lines:
            if line.strip():
                try:
                    logs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        
        # Sort by timestamp descending (newest first)
        logs.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
        
        if limit:
            logs = logs[:limit]
        
        return logs
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load attendance logs: {str(e)}")

def _load_user_metadata() -> Dict[str, Dict[str, Any]]:
    """Load user metadata."""
    if not USER_METADATA_PATH.exists():
        return {}
    try:
        with open(USER_METADATA_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _save_user_metadata(metadata: Dict[str, Dict[str, Any]]):
    """Save user metadata."""
    try:
        USER_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(USER_METADATA_PATH, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to save user metadata: {e}", file=__import__('sys').stderr)

def _check_duplicate(user_id: str, cooldown_minutes: int = COOLDOWN_MINUTES) -> Optional[Dict[str, Any]]:
    """Check if user has checked in recently. Returns last check-in if within cooldown."""
    logs = _load_attendance_logs(limit=100)  # Check last 100 records
    user_logs = [log for log in logs if log.get("user_id") == user_id]
    if not user_logs:
        return None
    
    last_log = user_logs[0]  # Already sorted by timestamp desc
    try:
        last_timestamp = datetime.fromisoformat(last_log.get("timestamp", "").replace("Z", "+00:00"))
        now = datetime.utcnow().replace(tzinfo=last_timestamp.tzinfo)
        time_diff = (now - last_timestamp).total_seconds() / 60  # minutes
        
        if time_diff < cooldown_minutes:
            return last_log
    except Exception:
        pass
    
    return None

def _determine_check_type(user_id: str) -> str:
    """
    Determine if this is a check-in or check-out based on last record.
    
    Logic:
    - If user has no records today → check-in
    - If last record today is check-in → check-out
    - If last record today is check-out → check-in
    - If last record is from previous day → check-in (new day starts)
    """
    logs = _load_attendance_logs(limit=100)
    user_logs = [log for log in logs if log.get("user_id") == user_id]
    
    if not user_logs:
        return "check-in"
    
    # Get today's records (sorted by timestamp desc, so first is most recent)
    today = datetime.utcnow().date()
    today_logs = []
    for log in user_logs:
        try:
            log_date = datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")).date()
            if log_date == today:
                today_logs.append(log)
        except Exception:
            continue
    
    # No records today → check-in
    if not today_logs:
        return "check-in"
    
    # Get the most recent record today (first in sorted list)
    last_record = today_logs[0]
    last_type = last_record.get("type", "check-in")
    
    # If last record is check-in → next should be check-out
    # If last record is check-out → next should be check-in
    if last_type == "check-in":
        return "check-out"
    else:
        return "check-in"

def _log_attendance(entry: Dict[str, Any]):
    """Append attendance entry to JSONL file."""
    try:
        ATTENDANCE_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(ATTENDANCE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        # Log error but don't fail the request
        print(f"[WARN] Failed to write attendance log: {e}", file=__import__('sys').stderr)

@router.post("/api/attendance")
async def log_attendance(
    user_id: str,
    match_score: float,
    liveness_score: float,
    emotion_label: Optional[str] = None,
    emotion_confidence: Optional[float] = None,
    force: bool = False
):
    """Log an attendance record. Returns error if duplicate check-in within cooldown period."""
    # Check for duplicate
    if not force:
        duplicate = _check_duplicate(user_id, COOLDOWN_MINUTES)
        if duplicate:
            return {
                "status": "error",
                "error": "duplicate",
                "message": f"User {user_id} already checked in recently. Please wait {COOLDOWN_MINUTES} minutes.",
                "last_check_in": duplicate
            }
    
    # Determine check type
    check_type = _determine_check_type(user_id)
    
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "user_id": user_id,
        "type": check_type,
        "match_score": float(match_score),
        "liveness_score": float(liveness_score),
        "emotion_label": emotion_label,
        "emotion_confidence": emotion_confidence
    }
    _log_attendance(entry)
    return {"status": "ok", "logged": entry, "type": check_type}

@router.get("/api/attendance")
async def get_attendance(
    limit: int = Query(100, ge=1, le=10000, description="Maximum number of records to return"),
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """Get attendance records with optional filtering. Enriches with user names from registry."""
    from app.pipelines.registry import FaceRegistry
    
    logs = _load_attendance_logs(limit=None)  # Load all first for filtering
    
    # Load registry to get user names
    registry = FaceRegistry()
    registry_data = registry.get_all()
    
    # Filter by user_id
    if user_id:
        logs = [log for log in logs if log.get("user_id") == user_id]
    
    # Filter by date range
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) >= from_date]
        except Exception:
            pass
    
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) <= to_date]
        except Exception:
            pass
    
    # Apply limit after filtering
    if limit:
        logs = logs[:limit]
    
    # Enrich logs with user names from registry
    enriched_logs = []
    registry_data = registry.get_all()  # Load registry once
    
    for log in logs:
        log_user_id = log.get("user_id", "")
        user_name = None
        
        # Get name from registry (new format)
        if log_user_id in registry_data:
            user_data = registry_data[log_user_id]
            if isinstance(user_data, dict) and "name" in user_data:
                user_name = user_data["name"]
            elif isinstance(user_data, list):
                # Legacy format - use user_id as name
                user_name = log_user_id
        
        # If no name found, use user_id as fallback
        if not user_name:
            user_name = log_user_id
        
        enriched_log = {
            **log,
            "name": user_name  # Add name field
        }
        enriched_logs.append(enriched_log)
    
    return {
        "records": enriched_logs,
        "count": len(enriched_logs)
    }

@router.get("/api/attendance/stats")
async def get_attendance_stats(
    user_id: Optional[str] = Query(None, description="Filter by user ID"),
    date_from: Optional[str] = Query(None, description="Filter from date (YYYY-MM-DD)"),
    date_to: Optional[str] = Query(None, description="Filter to date (YYYY-MM-DD)")
):
    """Get attendance statistics."""
    logs = _load_attendance_logs(limit=None)
    
    # Apply filters
    if user_id:
        logs = [log for log in logs if log.get("user_id") == user_id]
    
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) >= from_date]
        except Exception:
            pass
    
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) <= to_date]
        except Exception:
            pass
    
    # Calculate statistics
    total_records = len(logs)
    unique_users = len(set(log.get("user_id") for log in logs if log.get("user_id")))
    
    # Group by date
    by_date: Dict[str, int] = {}
    for log in logs:
        timestamp = log.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                date_key = dt.strftime("%Y-%m-%d")
                by_date[date_key] = by_date.get(date_key, 0) + 1
            except Exception:
                pass
    
    # Group by user
    by_user: Dict[str, int] = {}
    for log in logs:
        uid = log.get("user_id")
        if uid:
            by_user[uid] = by_user.get(uid, 0) + 1
    
    # Group by emotion
    by_emotion: Dict[str, int] = {}
    for log in logs:
        emotion = log.get("emotion_label", "unknown")
        by_emotion[emotion] = by_emotion.get(emotion, 0) + 1
    
    # Group by check type
    by_type: Dict[str, int] = {}
    check_ins = 0
    check_outs = 0
    for log in logs:
        check_type = log.get("type", "check-in")
        by_type[check_type] = by_type.get(check_type, 0) + 1
        if check_type == "check-in":
            check_ins += 1
        elif check_type == "check-out":
            check_outs += 1
    
    # Daily trends (last 30 days)
    daily_trends: Dict[str, int] = {}
    today = datetime.utcnow().date()
    for i in range(30):
        date_key = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        daily_trends[date_key] = by_date.get(date_key, 0)
    
    return {
        "total_records": total_records,
        "total_entries": total_records,  # Alias for clarity
        "unique_users": unique_users,
        "check_ins": check_ins,
        "check_outs": check_outs,
        "by_date": by_date,
        "by_user": by_user,
        "by_emotion": by_emotion,
        "by_type": by_type,
        "daily_trends": daily_trends
    }

@router.get("/api/attendance/export")
async def export_attendance(
    format: str = Query("csv", regex="^(csv|json)$"),
    user_id: Optional[str] = Query(None),
    date_from: Optional[str] = Query(None),
    date_to: Optional[str] = Query(None)
):
    """Export attendance records as CSV or JSON."""
    logs = _load_attendance_logs(limit=None)
    
    # Apply filters
    if user_id:
        logs = [log for log in logs if log.get("user_id") == user_id]
    
    if date_from:
        try:
            from_date = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) >= from_date]
        except Exception:
            pass
    
    if date_to:
        try:
            to_date = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
            logs = [log for log in logs if datetime.fromisoformat(log.get("timestamp", "").replace("Z", "+00:00")) <= to_date]
        except Exception:
            pass
    
    # Load user metadata for enrichment
    metadata = _load_user_metadata()
    
    if format == "csv":
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=[
            "timestamp", "user_id", "user_name", "department", "type",
            "match_score", "liveness_score", "emotion_label", "emotion_confidence"
        ])
        writer.writeheader()
        
        for log in logs:
            user_id = log.get("user_id", "")
            user_meta = metadata.get(user_id, {})
            writer.writerow({
                "timestamp": log.get("timestamp", ""),
                "user_id": user_id,
                "user_name": user_meta.get("name", ""),
                "department": user_meta.get("department", ""),
                "type": log.get("type", "check-in"),
                "match_score": f"{log.get('match_score', 0):.4f}",
                "liveness_score": f"{log.get('liveness_score', 0):.4f}",
                "emotion_label": log.get("emotion_label", ""),
                "emotion_confidence": f"{log.get('emotion_confidence', 0):.4f}" if log.get("emotion_confidence") else ""
            })
        
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename=attendance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"}
        )
    else:  # json
        # Enrich with metadata
        enriched_logs = []
        for log in logs:
            user_id = log.get("user_id", "")
            user_meta = metadata.get(user_id, {})
            enriched_log = log.copy()
            enriched_log["user_name"] = user_meta.get("name")
            enriched_log["department"] = user_meta.get("department")
            enriched_logs.append(enriched_log)
        
        return Response(
            content=json.dumps(enriched_logs, indent=2),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=attendance_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"}
        )

@router.get("/api/users/metadata")
async def get_user_metadata():
    """Get all user metadata."""
    return _load_user_metadata()

@router.post("/api/users/metadata")
async def set_user_metadata(
    user_id: str,
    name: Optional[str] = None,
    department: Optional[str] = None,
    email: Optional[str] = None
):
    """Set user metadata."""
    metadata = _load_user_metadata()
    if user_id not in metadata:
        metadata[user_id] = {}
    
    if name is not None:
        metadata[user_id]["name"] = name
    if department is not None:
        metadata[user_id]["department"] = department
    if email is not None:
        metadata[user_id]["email"] = email
    
    _save_user_metadata(metadata)
    return {"status": "ok", "user_id": user_id, "metadata": metadata[user_id]}

@router.delete("/api/users/metadata/{user_id}")
async def delete_user_metadata(user_id: str):
    """Delete user metadata."""
    metadata = _load_user_metadata()
    if user_id in metadata:
        del metadata[user_id]
        _save_user_metadata(metadata)
        return {"status": "ok", "deleted": user_id}
    raise HTTPException(status_code=404, detail="User metadata not found")

