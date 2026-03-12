@echo off
setlocal enabledelayedexpansion
title Digital Witness — Setup

echo.
echo ============================================================
echo   Digital Witness — Automatic Environment Setup
echo ============================================================
echo.

:: ── 1. Check Python ──────────────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found.
    echo         Download Python 3.10 or 3.11 from https://www.python.org/downloads/
    echo         Make sure to tick "Add Python to PATH" during install.
    pause & exit /b 1
)

for /f "tokens=2" %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo [OK] Python %PYVER% found

:: ── 2. Check version is 3.10+ ────────────────────────────────
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PYMAJ=%%a
    set PYMIN=%%b
)
if %PYMAJ% LSS 3 (
    echo [ERROR] Python 3.10 or higher required. You have %PYVER%
    pause & exit /b 1
)
if %PYMAJ% EQU 3 if %PYMIN% LSS 10 (
    echo [ERROR] Python 3.10 or higher required. You have %PYVER%
    pause & exit /b 1
)

:: ── 3. Create virtual environment ────────────────────────────
if exist .venv (
    echo [OK] Virtual environment already exists — skipping creation
) else (
    echo [..] Creating virtual environment...
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause & exit /b 1
    )
    echo [OK] Virtual environment created
)

:: ── 4. Upgrade pip ───────────────────────────────────────────
echo [..] Upgrading pip...
.venv\Scripts\python -m pip install --upgrade pip --quiet
echo [OK] pip upgraded

:: ── 5. Detect GPU / CUDA ─────────────────────────────────────
echo.
echo [..] Detecting GPU...
set CUDA_MAJOR=0
nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [INFO] No NVIDIA GPU detected — will install CPU-only PyTorch
) else (
    for /f "tokens=*" %%L in ('nvidia-smi 2^>nul') do (
        echo %%L | findstr /i "CUDA Version" >nul
        if not errorlevel 1 set CUDA_LINE=%%L
    )
    for /f "tokens=3 delims=: " %%v in ("!CUDA_LINE!") do (
        for /f "tokens=1 delims=." %%m in ("%%v") do set CUDA_MAJOR=%%m
    )
    echo [OK] NVIDIA GPU found — CUDA !CUDA_MAJOR!.x
)

:: ── 6. Install PyTorch (GPU or CPU) ──────────────────────────
echo.
echo [..] Installing PyTorch...
if !CUDA_MAJOR! GEQ 12 (
    echo      GPU mode: CUDA 12.x support
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121 --quiet
) else if !CUDA_MAJOR! EQU 11 (
    echo      GPU mode: CUDA 11.x support
    .venv\Scripts\pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118 --quiet
) else (
    echo      CPU mode: no GPU detected
    .venv\Scripts\pip install torch torchvision --quiet
)
if errorlevel 1 (
    echo [ERROR] PyTorch installation failed. Check your internet connection.
    pause & exit /b 1
)
echo [OK] PyTorch installed

:: ── 7. Install remaining packages ────────────────────────────
echo.
echo [..] Installing remaining packages from requirements.txt...
.venv\Scripts\pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERROR] Package installation failed.
    pause & exit /b 1
)
echo [OK] All packages installed

:: ── 8. Install Jupyter kernel ─────────────────────────────────
echo.
echo [..] Installing Jupyter kernel support...
.venv\Scripts\pip install ipykernel --quiet
if errorlevel 1 (
    echo [ERROR] ipykernel install failed.
    pause & exit /b 1
)
.venv\Scripts\python -m ipykernel install --user --name=digitalwitness --display-name "Digital Witness (.venv)"
if errorlevel 1 (
    echo [ERROR] Kernel registration failed.
    pause & exit /b 1
)
echo [OK] Jupyter kernel registered as "Digital Witness (.venv)"

:: ── 9. Verify ────────────────────────────────────────────────
echo.
echo [..] Verifying installation...
.venv\Scripts\python -c "import torch; gpu=torch.cuda.is_available(); name=torch.cuda.get_device_name(0) if gpu else 'CPU only'; print(f'[OK] PyTorch {torch.__version__}  |  GPU: {gpu}  |  Device: {name}')"
.venv\Scripts\python -c "import cv2, ultralytics, streamlit; print('[OK] OpenCV, Ultralytics, Streamlit — all OK')"

:: ── 10. Done ──────────────────────────────────────────────────
echo.
echo ============================================================
echo   Setup complete!
echo.
echo   In VSCode: open the notebook, click the kernel picker
echo   (top-right), and select "Digital Witness (.venv)"
echo.
echo   To activate the environment manually:
echo     .venv\Scripts\activate
echo.
echo   To run the app:
echo     .venv\Scripts\activate
echo     streamlit run app.py
echo ============================================================
echo.
pause