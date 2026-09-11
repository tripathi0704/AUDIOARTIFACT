@echo off
setlocal enabledelayedexpansion
title AudioArtifact Launcher

echo ================================================================
echo                   AudioArtifact v2.0 Launcher
echo ================================================================
echo.

REM 1. Verify that venv exists
if not exist "%~dp0venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment 'venv' not found!
    echo.
    echo Please run 'setup.bat' first to automatically configure the environment.
    echo.
    pause
    exit /b 1
)

REM 2. Verify that venv works on this specific computer
"%~dp0venv\Scripts\python.exe" -c "import sys" >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] The virtual environment 'venv' appears to be broken or was
    echo         copied directly from another computer with different paths!
    echo.
    echo [SOLUTION] Run 'setup.bat' to automatically rebuild a fresh environment.
    echo.
    pause
    exit /b 1
)

REM 3. Check model file
if not exist "%~dp0model\deepfake_detector.pkl" (
    echo [WARNING] Model file 'model\deepfake_detector.pkl' not found!
    echo Backend will start, but you may need to run 'python train_model.py'
    echo to generate the trained model.
    echo.
)

echo Starting Backend (FastAPI on http://127.0.0.1:8000)...
start "AudioArtifact - Backend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && uvicorn backend.main:app --reload --port 8000"

echo Waiting for backend service to initialize...
timeout /t 4 /nobreak >nul

echo Starting Frontend (Streamlit on http://localhost:8501)...
start "AudioArtifact - Frontend" cmd /k "cd /d %~dp0 && call venv\Scripts\activate && streamlit run frontend\app.py"

echo.
echo ================================================================
echo Both servers have been launched in separate windows!
echo   - Backend API : http://127.0.0.1:8000 (Swagger docs: /docs)
echo   - Frontend UI : http://localhost:8501
echo ================================================================
echo.
echo You can minimize or close this launcher window at any time.
pause
