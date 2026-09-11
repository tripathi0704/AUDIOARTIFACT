@echo off
setlocal enabledelayedexpansion
title AudioArtifact - Automated Environment Setup

echo ================================================================
echo         AudioArtifact v2.0 - Environment Setup Script
echo ================================================================
echo.

REM 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not added to your system PATH!
    echo.
    echo Please install Python 3.10, 3.11, or 3.12 (64-bit) from:
    echo https://www.python.org/downloads/
    echo.
    echo IMPORTANT: Make sure to check the box:
    echo   [x] "Add python.exe to PATH"
    echo during the Python installation!
    echo.
    pause
    exit /b 1
)

echo [OK] Python detected:
python --version
echo.

REM 2. Check for copied/broken venv
if exist "venv\" (
    echo [CHECK] Existing 'venv' folder detected. Verifying validity...
    
    REM Test if venv's python works on this machine
    venv\Scripts\python.exe -c "import sys" >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo [WARNING] The existing 'venv' was copied from another computer or is invalid!
        echo           Virtual environments cannot be copied across PCs because paths are hardcoded.
        echo [ACTION]  Removing invalid 'venv' folder...
        rmdir /s /q "venv"
        echo [OK] Cleaned old venv.
        echo.
    ) else (
        echo [OK] Existing virtual environment is valid.
    )
)

REM 3. Create fresh venv if not present
if not exist "venv\" (
    echo [1/4] Creating new virtual environment in .\venv ...
    python -m venv venv
    if !errorlevel! neq 0 (
        echo [ERROR] Failed to create virtual environment!
        echo Please ensure you have write permissions in this folder.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
) else (
    echo [1/4] Using existing .\venv
)
echo.

REM 4. Upgrade pip
echo [2/4] Upgrading pip...
venv\Scripts\python.exe -m pip install --upgrade pip --quiet
echo [OK] Pip upgraded.
echo.

REM 5. Install PyTorch CPU first (faster and lighter for Windows)
echo [3/4] Installing PyTorch (CPU build) and dependencies...
echo       (This may take 1-3 minutes depending on your internet connection)
venv\Scripts\pip.exe install torch torchaudio --index-url https://download.pytorch.org/whl/cpu --quiet
if !errorlevel! neq 0 (
    echo [NOTICE] PyTorch direct wheel install failed, will fallback to standard pip install...
)

REM 6. Install remaining requirements
echo [4/4] Installing project requirements from requirements.txt...
venv\Scripts\pip.exe install -r requirements.txt --quiet
if !errorlevel! neq 0 (
    echo [ERROR] Dependency installation failed! Please check your internet connection.
    pause
    exit /b 1
)
echo [OK] All dependencies successfully installed!
echo.

REM 7. Verify core imports
echo Verifying installation...
venv\Scripts\python.exe -c "import fastapi, uvicorn, streamlit, librosa, xgboost, torch, silero_vad; print('All core modules verified successfully!')"
if !errorlevel! neq 0 (
    echo [WARNING] One or more modules could not be verified. Please check logs.
) else (
    echo.
    echo ================================================================
    echo        SETUP COMPLETE! AudioArtifact is ready to use!
    echo ================================================================
    echo.
    echo To launch the application:
    echo   - Double-click 'start.bat'
    echo   OR run manually:
    echo     Backend:  uvicorn backend.main:app --reload --port 8000
    echo     Frontend: streamlit run frontend\app.py
    echo.
)

REM 8. Check FFmpeg
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo [RECOMMENDATION] FFmpeg was not detected in PATH.
    echo   WAV files will work fine out-of-the-box.
    echo   To analyze MP3 files, install FFmpeg via:
    echo     winget install Gyan.FFmpeg
    echo.
)

pause
