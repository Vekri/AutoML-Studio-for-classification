@echo off
title AutoML Studio (React + API)
echo Starting AutoML Studio...
echo.
echo [1/2] Building frontend if needed...
cd /d "%~dp0frontend"
if not exist "dist\index.html" (
  call npm install
  call npm run build
)
cd /d "%~dp0"
echo.
echo [2/2] Starting API + UI on http://localhost:7860
python -m pip install -r backend\requirements.txt -q
python -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
pause
