"""
Emotion Analytics API for workplace sentiment analysis and mental health insights.
"""
from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import json
from pathlib import Path
from collections import defaultdict, Counter

router = APIRouter(prefix="/emotion-analytics", tags=["Emotion Analytics"])

ATTENDANCE_FILE = Path(__file__).parent.parent / "store" / "attendance.jsonl"


def _load_attendance_data(days: int = 7) -> List[Dict]:
    """Load attendance data from last N days."""
    if not ATTENDANCE_FILE.exists():
        return []
    
    cutoff = datetime.utcnow() - timedelta(days=days)
    # Make cutoff timezone-naive for comparison
    cutoff = cutoff.replace(tzinfo=None)
    data = []
    
    with open(ATTENDANCE_FILE, "r") as f:
        for line in f:
            try:
                record = json.loads(line.strip())
                # Parse timestamp and make timezone-naive
                timestamp_str = record.get("timestamp", "")
                timestamp = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                # Convert to naive datetime (remove timezone info)
                timestamp = timestamp.replace(tzinfo=None)
                
                if timestamp >= cutoff:
                    data.append(record)
            except (ValueError, KeyError) as e:
                # Skip invalid records
                continue
    
    return data


@router.get("/trends/hourly")
async def get_hourly_emotion_trends(days: int = Query(default=7, ge=1, le=30)):
    """
    Get emotion distribution by hour of day (aggregated over last N days).
    
    Use case: Identify peak stress hours, optimal meeting times.
    """
    data = _load_attendance_data(days)
    
    hourly_data = defaultdict(lambda: defaultdict(int))
    
    for record in data:
        if not record.get("emotion_label"):
            continue
        
        timestamp = datetime.fromisoformat(record["timestamp"].replace("Z", "+00:00"))
        hour = timestamp.hour
        emotion = record["emotion_label"]
        
        hourly_data[hour][emotion] += 1
        hourly_data[hour]["_total"] += 1
    
    # Format output
    trends = []
    for hour in sorted(hourly_data.keys()):
        emotions = hourly_data[hour]
        total = emotions.pop("_total", 1)
        
        trends.append({
            "hour": hour,
            "total_records": total,
            "emotions": {k: round(v / total * 100, 1) for k, v in emotions.items()},
            "dominant_emotion": max(emotions.items(), key=lambda x: x[1])[0] if emotions else "neutral"
        })
    
    return {
        "period": f"last_{days}_days",
        "hourly_trends": trends
    }


@router.get("/user/{user_id}/profile")
async def get_user_emotion_profile(user_id: str, days: int = Query(default=7, ge=1, le=30)):
    """
    Get emotion profile for specific user over last N days.
    
    Use case: Mental health monitoring, burnout detection.
    """
    data = _load_attendance_data(days)
    
    user_records = [r for r in data if r.get("user_id") == user_id and r.get("emotion_label")]
    
    if not user_records:
        raise HTTPException(status_code=404, detail=f"No emotion data found for user {user_id}")
    
    emotions = [r["emotion_label"] for r in user_records]
    emotion_counts = Counter(emotions)
    total = len(emotions)
    
    # Calculate metrics
    distribution = {k: round(v / total * 100, 1) for k, v in emotion_counts.items()}
    dominant = emotion_counts.most_common(1)[0][0]
    
    # Concern flags
    concerns = []
    if distribution.get("sad", 0) > 40:
        concerns.append("high_sadness_ratio")
    if distribution.get("angry", 0) > 30:
        concerns.append("high_anger_ratio")
    if distribution.get("fear", 0) > 25:
        concerns.append("high_anxiety_ratio")
    if distribution.get("neutral", 0) > 80:
        concerns.append("low_emotional_expression")
    
    # Wellness score (0-100)
    positive = distribution.get("happy", 0) + distribution.get("surprise", 0)
    negative = distribution.get("sad", 0) + distribution.get("angry", 0) + distribution.get("fear", 0)
    wellness_score = max(0, min(100, 50 + (positive - negative)))
    
    return {
        "user_id": user_id,
        "period": f"last_{days}_days",
        "total_records": total,
        "emotion_distribution": distribution,
        "dominant_emotion": dominant,
        "wellness_score": round(wellness_score, 1),
        "concern_flags": concerns,
        "recommendation": _get_recommendation(wellness_score, concerns)
    }


@router.get("/department/sentiment")
async def get_department_sentiment(days: int = Query(default=7, ge=1, le=30)):
    """
    Compare emotion sentiment across departments.
    
    Use case: Identify departments needing support, resource allocation.
    """
    data = _load_attendance_data(days)
    
    dept_data = defaultdict(lambda: defaultdict(int))
    
    for record in data:
        dept = record.get("department", "Unknown")
        emotion = record.get("emotion_label")
        
        if not emotion:
            continue
        
        dept_data[dept][emotion] += 1
        dept_data[dept]["_total"] += 1
    
    # Calculate sentiment per department
    results = []
    for dept, emotions in dept_data.items():
        total = emotions.pop("_total", 1)
        
        positive = emotions.get("happy", 0) + emotions.get("surprise", 0)
        negative = emotions.get("sad", 0) + emotions.get("angry", 0) + emotions.get("fear", 0)
        
        happiness = round(positive / total * 100, 1)
        stress = round(negative / total * 100, 1)
        wellness = round(50 + (positive - negative) / total * 100, 1)
        
        results.append({
            "department": dept,
            "total_records": total,
            "happiness_percentage": happiness,
            "stress_percentage": stress,
            "wellness_score": max(0, min(100, wellness)),
            "emotion_distribution": {k: round(v / total * 100, 1) for k, v in emotions.items()}
        })
    
    # Sort by wellness score
    results.sort(key=lambda x: x["wellness_score"], reverse=True)
    
    return {
        "period": f"last_{days}_days",
        "departments": results
    }


@router.get("/anomalies")
async def detect_emotion_anomalies(days: int = Query(default=7, ge=1, le=30)):
    """
    Detect unusual emotion patterns that may indicate issues.
    
    Use case: Early intervention, proactive mental health support.
    """
    data = _load_attendance_data(days)
    
    # Group by user
    user_emotions = defaultdict(list)
    for record in data:
        if record.get("emotion_label"):
            user_emotions[record["user_id"]].append({
                "emotion": record["emotion_label"],
                "timestamp": record["timestamp"]
            })
    
    anomalies = []
    
    for user_id, emotions in user_emotions.items():
        if len(emotions) < 5:  # Need minimum data
            continue
        
        emotion_counts = Counter([e["emotion"] for e in emotions])
        total = len(emotions)
        
        # Check for prolonged negative emotions
        negative = emotion_counts.get("sad", 0) + emotion_counts.get("angry", 0) + emotion_counts.get("fear", 0)
        if negative / total > 0.6:
            anomalies.append({
                "type": "prolonged_negative",
                "user_id": user_id,
                "severity": "high" if negative / total > 0.8 else "medium",
                "details": f"{round(negative/total*100, 1)}% negative emotions over {days} days",
                "timestamp": emotions[-1]["timestamp"]
            })
        
        # Check for consistently angry
        if emotion_counts.get("angry", 0) / total > 0.4:
            anomalies.append({
                "type": "high_anger",
                "user_id": user_id,
                "severity": "high",
                "details": f"{round(emotion_counts['angry']/total*100, 1)}% angry emotions detected",
                "timestamp": emotions[-1]["timestamp"]
            })
    
    return {
        "period": f"last_{days}_days",
        "anomalies": anomalies,
        "total_anomalies": len(anomalies)
    }


def _get_recommendation(wellness_score: float, concerns: List[str]) -> str:
    """Generate recommendation based on wellness score and concerns."""
    if wellness_score >= 70:
        return "Employee showing positive emotional state. Continue current practices."
    elif wellness_score >= 50:
        return "Employee showing moderate emotional state. Consider check-in conversation."
    elif concerns:
        return f"Employee showing signs of distress ({', '.join(concerns)}). Recommend HR intervention or mental health support."
    else:
        return "Employee showing low emotional wellness. Immediate manager follow-up recommended."
