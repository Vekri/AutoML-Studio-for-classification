@echo off
title AutoML Studio
cd /d "%~dp0"
echo Installing Python packages...
python -m pip install -r backend\requirements.txt -q
echo.
if not exist "frontend\dist\index.html" (
  echo Building frontend...
  cd frontend
  call npm install
  call npm run build
  cd ..
)
echo Starting http://localhost:7860
start http://localhost:7860
python -m uvicorn backend.main:app --host 0.0.0.0 --port 7860
pause
