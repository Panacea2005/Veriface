@echo off
REM Veriface Backend - Complete Setup and Run Script
REM This script works in both CMD and PowerShell

setlocal enabledelayedexpansion

REM ========================================
REM MODEL CONFIGURATION
REM ========================================
REM Model A (PyTorch trained - Triplet Loss + iResNet100)
REM   - Set MODEL_TYPE=A
REM   - Requires modelA_best.pth in app/models/
REM   - Preprocessing: (pixel - 127.5) / 128.0 (matches notebook exactly)
REM   - ALWAYS RUNS (required)
REM
REM Model B (PyTorch trained - Sub-Center ArcFace + R100)
REM   - Set MODEL_TYPE=B
REM   - Requires modelB_best.pth in app/models/
REM   - Preprocessing: (pixel - 127.5) / 128.0 (matches notebook exactly)
REM   - ALWAYS RUNS (required)
REM
REM Internal configuration (not exposed in API)

REM ========================================
REM CHANGE THESE LINES TO CONFIGURE MODELS:
REM ========================================
set MODEL_TYPE=A
set ENABLE_DEEPFACE=1

echo ========================================
echo Veriface Backend Setup ^& Run
echo Using: Model %MODEL_TYPE%
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
    echo [MISSING] tensorflow - Required for anti-spoofing
    set MISSING_DEPS=1
)
python -m pip show keras >nul 2>&1
if errorlevel 1 (
    echo [MISSING] keras - Required for anti-spoofing
    set MISSING_DEPS=1
)
python -m pip show tf-keras >nul 2>&1
if errorlevel 1 (
    echo [MISSING] tf-keras - Required for anti-spoofing
    set MISSING_DEPS=1
)
python -m pip show torch >nul 2>&1
if errorlevel 1 (
    echo [INFO] torch - Required for Model A/B
)

if !MISSING_DEPS!==1 (
    echo.
    echo Installing/updating all dependencies from requirements.txt...
    echo This may take a few minutes, especially for TensorFlow...
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
    echo Forcing compatible TensorFlow stack...
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

REM Verify installation (use one-liner for CMD/PowerShell compatibility)
echo.
echo Verifying installation...
python -c "import importlib,sys; m=importlib.import_module('deepface'); import os; print('[OK] Import successful, version:', getattr(m,'__version__','unknown')); print('[OK] Python exe:', sys.executable); print('[OK] Module path:', getattr(m,'__file__','unknown'))"
if errorlevel 1 (
    echo.
    echo [ERROR] Import failed. Please check dependency output above.
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
REM MODEL_TYPE is already set at the top of the script
if "%SIMILARITY_METRIC%"=="" (
    set SIMILARITY_METRIC=cosine
)
REM Internal model weights configuration
if "%MODEL_TORCH_WEIGHT%"=="" (
    set MODEL_TORCH_WEIGHT=0.15
)
if "%MODEL_DEEPFACE_WEIGHT%"=="" (
    set MODEL_DEEPFACE_WEIGHT=0.85
)
REM Set model weights path based on MODEL_TYPE
if "%MODEL_TYPE%"=="A" (
    set MODEL_WEIGHTS_PATH=app\models\modelA_best.pth
    set REQUIRE_MODEL_A=1
    set REQUIRE_MODEL_B=0
) else if "%MODEL_TYPE%"=="B" (
    set MODEL_WEIGHTS_PATH=app\models\modelB_best.pth
    set REQUIRE_MODEL_A=0
    set REQUIRE_MODEL_B=1
    set BACKBONE_MODE=ir
    set BACKBONE_LAYERS=100
) else (
    echo [ERROR] Invalid MODEL_TYPE: %MODEL_TYPE%. Must be A or B.
    pause
    exit /b 1
)
REM PyTorch model is always required
    set REQUIRE_TORCH=1
REM Legacy configuration (kept for backward compatibility)
set DEEPFACE_ONLY=0

echo [OK] MODE=%MODE%
echo [OK] CORS_ORIGINS=%CORS_ORIGINS%
echo [OK] MODEL_TYPE=%MODEL_TYPE%
echo [OK] SIMILARITY_METRIC=%SIMILARITY_METRIC%
echo [OK] Embedding Model: Model %MODEL_TYPE%
    echo [OK] Model Weights: %MODEL_WEIGHTS_PATH%
    echo [OK] Preprocessing: Notebook training normalization

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

REM Step 5: Verify model files
    echo.
    echo Verifying PyTorch model files - Model %MODEL_TYPE%...
    if not exist "%MODEL_WEIGHTS_PATH%" (
        echo [ERROR] Missing required model: %MODEL_WEIGHTS_PATH%
        echo Please place your Model %MODEL_TYPE% checkpoint at the configured path.
        echo Expected path: %MODEL_WEIGHTS_PATH%
        echo.
        pause
        exit /b 1
    )
    echo [OK] Model %MODEL_TYPE% checkpoint found: %MODEL_WEIGHTS_PATH%
    
REM Internal verification (silent)

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
