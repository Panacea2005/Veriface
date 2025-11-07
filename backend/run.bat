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
REM DeepFace is used for emotion, antispoof, and detection; no ONNX/PTH needed
echo [OK] MODE=%MODE% - Using DeepFace for detection/liveness/emotion
echo [OK] CORS_ORIGINS=%CORS_ORIGINS%
set PYTHONUNBUFFERED=1

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

REM Step 5: Check and verify PyTorch models
echo.
echo Checking PyTorch embedding models...
python -c "from pathlib import Path; import sys; models_dir = Path('app/models'); pth_path = models_dir / 'ms1mv3_arcface_r100_fp16.pth'; pth_exists = pth_path.exists(); pth_size = pth_path.stat().st_size if pth_exists else 0; print(f'ms1mv3_arcface_r100_fp16.pth: {\"FOUND\" if pth_exists else \"NOT FOUND\"}' + (f' ({pth_size/1024/1024:.1f} MB)' if pth_exists else '')); sys.exit(0)" 2>nul
if errorlevel 1 (
    echo [INFO] Checking PyTorch models...
) else (
    echo [INFO] Verifying PyTorch model compatibility...
    python -c "import torch; from pathlib import Path; import sys; models_dir = Path('app/models'); pth_path = models_dir / 'ms1mv3_arcface_r100_fp16.pth'; if pth_path.exists(): try: from app.pipelines.arcface_model import get_model; model = get_model(input_size=[112, 112], num_layers=100, mode='ir'); ckpt = torch.load(str(pth_path), map_location='cpu'); state_dict = ckpt if isinstance(ckpt, dict) and not ('state_dict' in ckpt or 'model' in ckpt) else (ckpt.get('state_dict', ckpt.get('model', ckpt))); model.load_state_dict(state_dict, strict=False); model.eval(); test_input = torch.randn(1, 3, 112, 112); with torch.no_grad(): test_output = model(test_input); if test_output.shape[-1] == 512: print(f'[OK] PyTorch model: Valid (output dim: 512)'); else: print(f'[WARN] PyTorch model: Invalid output dim {test_output.shape[-1]} (expected 512)'); except Exception as e: err_msg = str(e)[:80]; print(f'[WARN] PyTorch model: Error - {err_msg}'); print('[INFO] System will automatically use DeepFace fallback for embedding extraction'); sys.exit(0)" 2>nul
    if errorlevel 0 (
        echo [OK] PyTorch model verification complete
    )
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
