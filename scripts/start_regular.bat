@echo off
rem Starts the regular local demo: backend (port 8000) and frontend (port 5173) in two windows, then opens the site.
rem Close any other backend first (two backends cannot share the local Qdrant store). Stop with Ctrl+C in each window.
cd /d "%~dp0.."
start "MedOrchestrate backend" cmd /k "venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"
start "MedOrchestrate frontend" cmd /k "cd frontend && npm run dev"
echo Waiting for the servers to start (the first start loads the models, about a minute)...
timeout /t 25 /nobreak >nul
start "" http://localhost:5173
echo Run one warm-up case before presenting. Close the two server windows to stop.
