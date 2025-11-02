@echo off
REM Veriface Backend - Complete Setup and Run Script

echo ========================================
echo Veriface Backend Setup ^& Run
echo ========================================
echo.

REM Step 1: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python first.
    pause
    exit /b 1
)

REM Step 2: Check dependencies
echo Checking dependencies...
pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies!
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed
) else (
    echo [OK] Dependencies already installed
)

REM Step 3: Set environment variables
echo.
echo Setting environment variables...
set MODE=onnx
set CORS_ORIGINS=http://localhost:3000,http://localhost:3001
echo [OK] MODE=%MODE%
echo [OK] CORS_ORIGINS=%CORS_ORIGINS%

REM Step 4: Create necessary directories
if not exist "app\store" mkdir app\store
if not exist "app\models" mkdir app\models

REM Step 5: Check and convert ONNX models if needed
echo.
echo Checking ONNX models compatibility...
python scripts\convert_onnx_version.py >nul 2>&1
if errorlevel 1 (
    echo [WARN] Model conversion check failed, but continuing...
) else (
    echo [OK] Models checked/comverted
)

REM Step 6: Run server
echo.
echo ========================================
echo Starting Veriface Backend
echo Mode: %MODE%
echo URL: http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

uvicorn app.main:app --reload --port 8000 --host 0.0.0.0

pause
