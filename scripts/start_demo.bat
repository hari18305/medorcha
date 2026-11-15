@echo off
rem Starts the single-address demo server on http://localhost:8000 (site + API behind one shared password).
rem Usage: scripts\start_demo.bat  (asks for the password), then run a tunnel (see DEPLOY_DEMO.md).
cd /d "%~dp0.."
if "%DEMO_PASSWORD%"=="" set /p DEMO_PASSWORD=Choose the demo password to share with judges: 
if not exist frontend\dist\index.html (
  echo Building the website once...
  pushd frontend
  set VITE_API_BASE_URL=/api
  call npm run build
  popd
)
venv\Scripts\python.exe -m uvicorn backend.demo_app:app --host 127.0.0.1 --port 8000
