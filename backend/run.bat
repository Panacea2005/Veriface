@echo off
REM Veriface Backend - Complete Setup and Run Script
REM This script works in both CMD and PowerShell

setlocal enabledelayedexpansion

echo ========================================
echo Veriface Backend Setup ^& Run
echo ========================================
echo.

REM Step 1: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found! Please install Python first.
    echo.
    pause
    exit /b 1
)
python --version
echo.

REM Step 2: Check and install dependencies
echo Checking dependencies...
echo.

REM Ensure latest pip/setuptools/wheel
python -m pip install --upgrade pip setuptools wheel >nul 2>&1

REM Check critical packages
set MISSING_DEPS=0
python -m pip show fastapi >nul 2>&1
if errorlevel 1 (
    echo [MISSING] fastapi
    set MISSING_DEPS=1
)
python -m pip show deepface >nul 2>&1
if errorlevel 1 (
    echo [MISSING] deepface - Required for anti-spoofing
    set MISSING_DEPS=1
)
python -m pip show numpy >nul 2>&1
if errorlevel 1 (
    echo [MISSING] numpy
    set MISSING_DEPS=1
)
python -m pip show opencv-python >nul 2>&1
if errorlevel 1 (
    echo [MISSING] opencv-python
    set MISSING_DEPS=1
)
python -m pip show tensorflow >nul 2>&1
if errorlevel 1 (
    echo [MISSING] tensorflow - Optional but recommended for DeepFace
    set MISSING_DEPS=1
)
python -m pip show keras >nul 2>&1
if errorlevel 1 (
    echo [MISSING] keras - Required for DeepFace backends
    set MISSING_DEPS=1
)
python -m pip show tf-keras >nul 2>&1
if errorlevel 1 (
    echo [MISSING] tf-keras - Required for DeepFace backends
    set MISSING_DEPS=1
)
python -m pip show torch >nul 2>&1
if errorlevel 1 (
    echo [MISSING] torch - Required for PyTorch embedding models
    set MISSING_DEPS=1
)

if !MISSING_DEPS!==1 (
    echo.
    echo Installing/updating all dependencies from requirements.txt...
    echo This may take a few minutes, especially for TensorFlow and DeepFace...
    echo.
    echo Removing conflicting JAX packages if present...
    python -m pip uninstall -y jax jaxlib >nul 2>&1
    echo Ensuring ml-dtypes compatibility for TF...
    python -m pip install --upgrade ml-dtypes==0.5.0 >nul 2>&1
    python -m pip install --no-cache-dir -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Failed to install some dependencies!
        echo Please check the error messages above.
        echo.
        echo You can try installing manually:
        echo   pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
    echo Forcing compatible DeepFace + TensorFlow stack...
    python -m pip install --upgrade --no-cache-dir tensorflow==2.17.0 tf-keras==2.17.0 keras==3.4.1 deepface==0.0.95 ml-dtypes==0.5.0
    echo.
    echo [OK] All dependencies installed/updated successfully
) else (
    echo [OK] All critical dependencies are installed
    echo.
    echo Checking if all packages from requirements.txt are up to date...
    python -m pip install -r requirements.txt --upgrade --quiet
    if errorlevel 0 (
        echo [OK] Dependencies are up to date
    )
)

REM Verify DeepFace import works (use one-liner for CMD/PowerShell compatibility)
echo.
echo Verifying DeepFace installation...
python -c "import importlib,sys; m=importlib.import_module('deepface'); import os; print('[OK] DeepFace import successful, version:', getattr(m,'__version__','unknown')); print('[OK] Python exe:', sys.executable); print('[OK] DeepFace path:', getattr(m,'__file__','unknown'))"
if errorlevel 1 (
    echo.
    echo [ERROR] DeepFace import failed. Please check dependency output above.
    pause
    exit /b 1
)

echo.
echo Verifying TensorFlow / Keras backend...
python -c "import tensorflow as tf; import tensorflow.keras as K; print('[OK] TensorFlow:', getattr(tf,'__version__','unknown')); print('[OK] Keras available')" 2>nul
if errorlevel 1 (
    echo [WARN] TensorFlow/Keras verification failed. Trying to reinstall minimal stack...
    python -m pip uninstall -y jax jaxlib >nul 2>&1
    python -m pip install --upgrade --no-cache-dir ml-dtypes==0.5.0 tensorflow==2.17.0 tf-keras==2.17.0 keras==3.4.1
)

REM Step 3: Set environment variables
echo.
echo Setting environment variables...
set MODE=heur
set CORS_ORIGINS=http://localhost:3000,http://localhost:3001
set PYTHONUNBUFFERED=1
REM Enforce PyTorch Model A usage (fail fast if missing)
set REQUIRE_TORCH=1
set REQUIRE_MODEL_A=1
if "%MODEL_WEIGHTS_PATH%"=="" (
    set MODEL_WEIGHTS_PATH=app\models\modelA_best.pth
)
if "%SIMILARITY_METRIC%"=="" (
    set SIMILARITY_METRIC=cosine
)
echo [OK] MODE=%MODE%
echo [OK] CORS_ORIGINS=%CORS_ORIGINS%
echo [OK] MODEL_WEIGHTS_PATH=%MODEL_WEIGHTS_PATH%
echo [OK] SIMILARITY_METRIC=%SIMILARITY_METRIC%

REM Step 4: Create necessary directories
echo.
echo Creating necessary directories...
if not exist "app\store" (
    mkdir app\store
    echo [OK] Created app\store directory
) else (
    echo [OK] app\store directory exists
)
if not exist "app\models" (
    mkdir app\models
    echo [OK] Created app\models directory
) else (
    echo [OK] app\models directory exists
)

REM Step 5: Verify PyTorch model files (require Model A)
echo.
echo Verifying PyTorch embedding model files...
if not exist "%MODEL_WEIGHTS_PATH%" (
    echo [ERROR] Missing required model: %MODEL_WEIGHTS_PATH%
    echo Please place your ArcFace weights ^(512-D backbone^) at the configured path or update MODEL_WEIGHTS_PATH.
    echo.
    pause
    exit /b 1
)

python scripts\verify_models.py
if errorlevel 1 (
    echo.
    echo [ERROR] Model verification failed. Check the message above and MODEL_WEIGHTS_PATH.
    pause
    exit /b 1
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

uvicorn app.main:app --reload --port 8000 --host 0.0.0.0 --log-level debug

pause
