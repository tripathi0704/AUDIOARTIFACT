
@echo off
title AudioArtifact Launcher

echo Starting AudioArtifact...
echo.

REM Start backend (FastAPI) in a new window
start "AudioArtifact - Backend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && uvicorn backend.main:app --reload --port 8000"

REM Wait a few seconds so the backend is up before the frontend tries to call it
timeout /t 5 /nobreak >nul

REM Start frontend (Streamlit) in a new window
start "AudioArtifact - Frontend" cmd /k "cd /d %~dp0 && venv\Scripts\activate && streamlit run frontend\app.py"

echo Both servers are starting in separate windows.
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://localhost:8501
echo.
echo Close this window anytime - it does not affect the servers.
pause
