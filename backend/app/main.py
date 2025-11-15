from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.core.config import CORS_ORIGINS
from app.routers import health, register, verify, registry as registry_router
from app.routers import emotion_logs
from app.routers import emotion_rt
from app.routers import liveness_rt
from app.routers import attendance
from app.routers import emotion_analytics
import traceback
import sys
import os

app = FastAPI(title="Veriface API", version="1.0.0")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(health.router)
app.include_router(register.router)
app.include_router(verify.router)
app.include_router(registry_router.router)
app.include_router(emotion_logs.router)
app.include_router(emotion_rt.router)
app.include_router(liveness_rt.router)
app.include_router(attendance.router)
app.include_router(emotion_analytics.router)

@app.on_event("startup")
async def startup_event():
    """Initialize API server."""
    print("Starting Veriface API...", file=sys.stderr)
    # Models are lazy-loaded when first used

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler to catch all unhandled errors."""
    tb = traceback.format_exc()
    print("=" * 80, file=sys.stderr)
    print(f"UNHANDLED ERROR: {type(exc).__name__}", file=sys.stderr)
    print(f"Path: {request.url.path}", file=sys.stderr)
    print(f"Error: {str(exc)}", file=sys.stderr)
    print("-" * 80, file=sys.stderr)
    print(tb, file=sys.stderr)
    print("=" * 80, file=sys.stderr)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}: {str(exc)}"}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

